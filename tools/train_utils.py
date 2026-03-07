import os
import torch
import torch.nn as nn
import numpy as np
import random
from collections import OrderedDict
from einops import rearrange
from torchvision.transforms import Normalize
import torchvision
from omegaconf import OmegaConf

from losses.fm import FlowMatching

# Global multiprocessing variables
_rank = 0  
_sync_device = None  
_sync_called = False  

def init_multiprocessing(rank, sync_device):
    global _rank, _sync_device
    assert not _sync_called
    _rank = rank
    _sync_device = sync_device

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())
    for name, buffer in model_buffers.items():
        name = name.replace("module.", "")
        if buffer.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)
        else:
            ema_buffers[name].copy_(buffer)

def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2)
    if mse == 0: return 100.0
    return 10 * torch.log10(1.0 / mse).item()

class FMSamplingWrapper(nn.Module):
    def __init__(self, dit_model):
        super().__init__()
        self.dit_model = dit_model.module if hasattr(dit_model, 'module') else dit_model

    def forward(self, x, cond, t):
        return self.dit_model.forward_sampling(x, cond, t)

@torch.no_grad()
def run_evaluation(val_loader, ema, vae_cond_model, vae_pred_model, args, device, it, logger, log_):
    """
    Quantitative evaluation on the validation set: PSNR, SSIM, LPIPS, FVD.
    """
    try:
        from evals.fvd.fvd import calculate_fvd
        from evals.fvd.download import load_i3d_pretrained
        from losses.lpips import LPIPS
        from skimage.metrics import structural_similarity as ssim_func
    except ImportError as e:
        log_(f"❌ Failed to import evaluation libraries: {e}")
        return

    log_(f"🚀 Running quantitative evaluation (target samples: {args.eval_samples})...")
    
    i3d = load_i3d_pretrained(device)
    lpips_fn = LPIPS().eval().to(device)
    all_reals, all_preds = [], []
    num_samples = 0
    
    for x_val, _ in val_loader:
        if num_samples >= args.eval_samples:
            break
            
        b_vis = x_val.size(0)
        x_val = x_val.to(device)
        x_vae_val = rearrange(x_val / 127.5 - 1, 'b t c h w -> b c t h w') 
        
        x_cond_val = x_vae_val[:, :, :args.cond_frames] 
        x_pred_val = x_vae_val[:, :, args.cond_frames : args.cond_frames + args.frames]
        
        c_feat_val = vae_cond_model.module.extract(x_cond_val).detach()
        unwrapped_ema = ema.module if hasattr(ema, 'module') else ema
        true_seq_len = unwrapped_ema.ae_emb_dim
        
        fm_sampler = FlowMatching(
            FMSamplingWrapper(ema), channels=args.in_channels, image_size=true_seq_len, sampling_timesteps=50
        ).to(device)
        
        z_sampled_val = fm_sampler.sample(batch_size=b_vis, cond=c_feat_val)
        
        if hasattr(vae_pred_model.module, 'decode_from_sample'):
            pred_vis_val = vae_pred_model.module.decode_from_sample(z_sampled_val)
        else:
            out_vis_val = vae_pred_model.module.decode(z_sampled_val)
            pred_vis_val = out_vis_val.sample if hasattr(out_vis_val, 'sample') else out_vis_val
            
        pred_vis_val = pred_vis_val.clamp(-1.0, 1.0)
        
        if pred_vis_val.dim() == 4:
            T_pred = pred_vis_val.shape[0] // b_vis
            pred_vis_val = pred_vis_val.view(b_vis, T_pred, pred_vis_val.shape[1], pred_vis_val.shape[2], pred_vis_val.shape[3])
            pred_vis_val = pred_vis_val.permute(0, 2, 1, 3, 4)
            
        min_T = min(x_pred_val.shape[2], pred_vis_val.shape[2])
        real_vid = x_pred_val[:, :, :min_T, :, :].cpu().permute(0, 2, 1, 3, 4)
        gen_vid = pred_vis_val[:, :, :min_T, :, :].cpu().permute(0, 2, 1, 3, 4)
        
        all_reals.append(real_vid)
        all_preds.append(gen_vid)
        num_samples += b_vis

    if len(all_reals) == 0: return

    reals_btchw = torch.cat(all_reals, dim=0)[:args.eval_samples]
    preds_btchw = torch.cat(all_preds, dim=0)[:args.eval_samples]
    
    reals_01 = (reals_btchw + 1.0) / 2.0
    preds_01 = (preds_btchw + 1.0) / 2.0
    
    try:
        psnr_val = compute_psnr(reals_01, preds_01)
        
        reals_np = reals_01.numpy()
        preds_np = preds_01.numpy()
        B, T, C, H, W = reals_np.shape
        ssim_sum = 0.0
        for b in range(B):
            for t in range(T):
                img1 = np.transpose(reals_np[b, t], (1, 2, 0)) 
                img2 = np.transpose(preds_np[b, t], (1, 2, 0))
                ssim_sum += ssim_func(img1, img2, data_range=1.0, channel_axis=-1)
        ssim_val = ssim_sum / (B * T)
        
        reals_11 = reals_01 * 2 - 1
        preds_11 = preds_01 * 2 - 1
        lpips_sum = 0.0
        for b in range(B):
            r_f = reals_11[b].to(device)
            p_f = preds_11[b].to(device)
            with torch.no_grad():
                lpips_sum += lpips_fn(r_f, p_f).mean().item()
        lpips_val = lpips_sum / B
        del lpips_fn
        
        reals_uint8 = (reals_01 * 255).clamp(0, 255).to(torch.uint8).to(device)
        preds_uint8 = (preds_01 * 255).clamp(0, 255).to(torch.uint8).to(device)
        
        reals_fvd = reals_uint8.permute(0, 1, 3, 4, 2) 
        preds_fvd = preds_uint8.permute(0, 1, 3, 4, 2)

        try:
            fvd_val = calculate_fvd(reals_fvd, preds_fvd, i3d, device)
            if isinstance(fvd_val, torch.Tensor): 
                fvd_val = fvd_val.item()
        except Exception as e:
            log_(f"⚠️ FVD calculation failed: {str(e)}")
            fvd_val = 0.0
                
        log_(f"✅ [Eval Report Step {it}] PSNR: {psnr_val:.4f} | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f} | FVD: {fvd_val:.4f}")
        
        if logger is not None:
            logger.scalar_summary('eval/psnr', psnr_val, it)
            logger.scalar_summary('eval/ssim', ssim_val, it)
            logger.scalar_summary('eval/lpips', lpips_val, it)
            logger.scalar_summary('eval/fvd', fvd_val, it)
    except Exception as e:
        log_(f"❌ Evaluation process exception: {str(e)}")
    finally:
        torch.cuda.empty_cache()

def save_image_grid(img, fname, drange, grid_size=None, normalize=True):
    from PIL import Image
    if normalize:
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

    B, C, T, H, W = img.shape
    img = img.transpose(0, 3, 2, 4, 1) 
    img = img.reshape(B * H, T * W, C)
    if C == 1: img = np.repeat(img, 3, axis=2)

    result_img = Image.fromarray(img, 'RGB')
    result_img.save(fname, quality=95)

def log_videos_e2e(gts, predictions, it, save_dir):
    gts_np = gts.detach().cpu().numpy()
    preds_np = predictions.detach().cpu().numpy()
    combined = np.stack([gts_np, preds_np], axis=1)
    combined = combined.reshape(-1, *gts_np.shape[1:])
    save_path = os.path.join(save_dir, f'vis_e2e_{it:07d}.png')
    save_image_grid(combined, save_path, drange=[-1, 1])

def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def get_teacher_model(model_name, device, ckpt_dir):
    if model_name == "VideoMAEv2":
        from models.ssl.videomaev2 import vit_base_patch16_224
        model = vit_base_patch16_224().to(device)
        model.from_pretrained(os.path.join(ckpt_dir, 'VideoMAEv2/vit_b_k710_dl_from_giant.pth'))
    elif model_name == "VideoMAE":
        from models.ssl.videomae import VideoMAE_vit_base_patch16_224
        model = VideoMAE_vit_base_patch16_224().to(device)
        model.from_pretrained(os.path.join(ckpt_dir, 'VideoMAE/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0_9_e1600.pth'))
    elif model_name == 'OminiMAE':
        from models.ssl.omini_mae import vit_base_mae_pretraining
        model = vit_base_mae_pretraining().to(device)
        model.tubelet_size = 2; model.patch_size = 16; model.embed_dim = 768
    elif model_name == 'VJEPA':
        from models.ssl.JEPA import load_VJEPA
        model = load_VJEPA(device=device, pretrained_path=os.path.join(ckpt_dir, 'vjepa_l/vitl16.pth.tar'))
    elif model_name == "VJEPA2":
        model, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')
        model = model.to(device)
        if hasattr(model, 'norm'): model.norm = torch.nn.Identity()
    elif model_name == 'DINOv3':
        model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16').to(device)
        if hasattr(model, 'head'): model.head = torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported align model: {model_name}")

    model.eval()
    set_requires_grad(model, False) 
    return model

def get_align_targets(x_pred, teacher_model, align_model_name, patch_size=16, tubelet_size=2):
    B, C, F, H, W = x_pred.shape
    frames_01 = (x_pred + 1.0) / 2.0
    frames_flat = frames_01.transpose(1, 2).flatten(0, 1)
    norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    frames_norm = Normalize(norm_mean, norm_std)(frames_flat)
    repa_raw_frames = frames_norm.reshape(B, F, C, H, W).transpose(1, 2)

    with torch.no_grad():
        if align_model_name in ['VideoMAEv2', 'VideoMAE', 'OminiMAE', 'VJEPA', 'VJEPA2']:
            align_target = teacher_model(repa_raw_frames)
        elif align_model_name in ['DINOv2', 'DINOv3']:
            frames_2d = repa_raw_frames.transpose(1, 2).flatten(0, 1)
            group_size = 32
            chunked = frames_2d.chunk(max(1, frames_2d.shape[0] // group_size), dim=0)
            features_list = []
            for chunk in chunked:
                out = teacher_model.forward_features(chunk)
                features_list.append(out['x_norm_patchtokens'])
            features_cat = torch.cat(features_list, dim=0) 
            _, L_2d, D = features_cat.shape
            align_target = features_cat.reshape(B, F * L_2d, D)
        else:
            raise NotImplementedError()
    return align_target

def prepare_input(args, x, vae_cond_model=None, vae_pred_model=None):
    p = np.random.random()
    if p < args.cond_prob:
        c, x = x[:, :, :args.cond_frames], x[:, :, args.cond_frames:]
        mask = (c + 1).contiguous().view(c.size(0), -1) ** 2
        mask = torch.where(mask.sum(dim=-1) > 0, 1, 0).view(-1, 1, 1)
        with torch.no_grad():
            z = vae_pred_model.module.extract(x).detach()
            if vae_cond_model is not None:
                c = vae_cond_model.module.extract(c).detach()
            else:
                c = vae_pred_model.module.extract(c).detach()
            c = c * mask + torch.zeros_like(c).to(c.device) * (1 - mask)
    else:
        c, x_tmp = x[:, :, :args.cond_frames], x[:, :, args.cond_frames:]
        mask = (c + 1).contiguous().view(c.size(0), -1) ** 2
        mask = torch.where(mask.sum(dim=-1) > 0, 1, 0).view(-1, 1, 1, 1, 1)
        clip_length = x_tmp.size(2)
        prefix = random.randint(0, args.cond_frames)
        x = x[:, :, prefix:prefix + clip_length, :, :] * mask + x_tmp * (1 - mask)
        with torch.no_grad():
            z = vae_pred_model.module.extract(x).detach()
            c = torch.zeros_like(z).to(x.device)
    return z, c

def config_setup(args):
    config = OmegaConf.load(args.vae_config)
    args.embed_dim = config.vae.params.embed_dim
    args.lossconfig = config.vae.params.lossconfig
    args.perceptual_weight = config.vae.params.perceptual_weight
    args.pred_frames = config.vae.vaeconfig.pred_frames
    args.cond_frames = config.vae.vaeconfig.cond_frames
    base_cfg = {k: v for k, v in config.vae.vaeconfig.items() if k not in ["pred_frames", "cond_frames"]}
    args.vaeconfig = OmegaConf.create({**base_cfg, "frames": config.vae.vaeconfig.pred_frames})
    args.cond_vaeconfig = OmegaConf.create({**base_cfg, "frames": config.vae.vaeconfig.cond_frames})
    args.frames = args.pred_frames + args.cond_frames
    args.res = config.vae.vaeconfig.resolution
    args.amp = config.vae.amp if 'amp' in config.vae else False
    args.fmconfig = config.model.params
    args.cond_prob = config.model.cond_prob if 'cond_prob' in config.model else 0.0
    args.same_noise = config.model.same_noise if 'same_noise' in config.model else True
    args.input_size = config.model.sit_config.input_size
    args.in_channels = config.model.sit_config.in_channels
    args.encoder_depth = config.model.sit_config.encoder_depth
    args.bn_momentum = config.model.sit_config.bn_momentum
    args.lr = config.model.base_learning_rate if 'base_learning_rate' in config.model else 1e-4
    args.max_iter = config.model.max_iter
    args.log_freq = config.model.log_freq
    args.eval_freq = config.model.eval_freq
    args.max_size = config.model.max_size if 'max_size' in config.model else None
    args.eval_samples = config.model.eval_samples if 'eval_samples' in config.model else 16
    args.resume = config.model.resume if 'resume' in config.model else False
    return args