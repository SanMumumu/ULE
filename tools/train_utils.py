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
        if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
            ema_buffers[name].copy_(buffer)
        elif buffer.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
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
    Separated for Cond frames (reconstruction) and Pred frames (generation).
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
    
    all_reals_cond, all_preds_cond = [], []
    all_reals_pred, all_preds_pred = [], []
    all_reals_pred_recon, all_preds_pred_recon = [], [] # [新增] 用于收集预测帧的 VAE 重建结果
    num_samples = 0
    
    unwrapped_vae_cond = vae_cond_model.module if hasattr(vae_cond_model, 'module') else vae_cond_model
    unwrapped_vae_pred = vae_pred_model.module if hasattr(vae_pred_model, 'module') else vae_pred_model
    
    for x_val, _ in val_loader:
        if num_samples >= args.eval_samples:
            break
            
        b_vis = x_val.size(0)
        x_val = x_val.to(device)
        x_vae_val = rearrange(x_val / 127.5 - 1, 'b t c h w -> b c t h w') 
        
        x_cond_val = x_vae_val[:, :, :args.cond_frames] 
        x_pred_val = x_vae_val[:, :, args.cond_frames : args.frames]
        
        c_feat_val_raw = unwrapped_vae_cond.extract(x_cond_val).detach()
        c_feat_val = c_feat_val_raw
        
        # ==================== A. VAE Reconstruction ====================
        if hasattr(unwrapped_vae_cond, 'decode_from_sample'):
            pred_cond_val = unwrapped_vae_cond.decode_from_sample(c_feat_val_raw)
        else:
            out_cond_val = unwrapped_vae_cond.decode(c_feat_val_raw)
            pred_cond_val = out_cond_val.sample if hasattr(out_cond_val, 'sample') else out_cond_val
            
        pred_cond_val = pred_cond_val.clamp(-1.0, 1.0)
        
        if pred_cond_val.dim() == 4:
            T_cond = pred_cond_val.shape[0] // b_vis
            pred_cond_val = pred_cond_val.view(b_vis, T_cond, pred_cond_val.shape[1], pred_cond_val.shape[2], pred_cond_val.shape[3])
            pred_cond_val = pred_cond_val.permute(0, 2, 1, 3, 4)
            
        min_T_cond = min(x_cond_val.shape[2], pred_cond_val.shape[2])
        real_vid_cond = x_cond_val[:, :, :min_T_cond, :, :].cpu().permute(0, 2, 1, 3, 4)
        gen_vid_cond = pred_cond_val[:, :, :min_T_cond, :, :].cpu().permute(0, 2, 1, 3, 4)
        all_reals_cond.append(real_vid_cond)
        all_preds_cond.append(gen_vid_cond)

        # ==================== [新增] VAE Future Frames Reconstruction ====================
        # 直接使用 vae_pred_model 提取真实未来帧的特征，并立即解码，以测试预测域 VAE 的性能
        z_pred_gt_val = unwrapped_vae_pred.extract(x_pred_val).detach()
        
        if hasattr(unwrapped_vae_pred, 'decode_from_sample'):
            pred_recon_val = unwrapped_vae_pred.decode_from_sample(z_pred_gt_val)
        else:
            out_recon_val = unwrapped_vae_pred.decode(z_pred_gt_val)
            pred_recon_val = out_recon_val.sample if hasattr(out_recon_val, 'sample') else out_recon_val
            
        pred_recon_val = pred_recon_val.clamp(-1.0, 1.0)
        
        if pred_recon_val.dim() == 4:
            T_pred_recon = pred_recon_val.shape[0] // b_vis
            pred_recon_val = pred_recon_val.view(b_vis, T_pred_recon, pred_recon_val.shape[1], pred_recon_val.shape[2], pred_recon_val.shape[3])
            pred_recon_val = pred_recon_val.permute(0, 2, 1, 3, 4)
            
        min_T_pred_recon = min(x_pred_val.shape[2], pred_recon_val.shape[2])
        real_vid_pred_recon = x_pred_val[:, :, :min_T_pred_recon, :, :].cpu().permute(0, 2, 1, 3, 4)
        gen_vid_pred_recon = pred_recon_val[:, :, :min_T_pred_recon, :, :].cpu().permute(0, 2, 1, 3, 4)
        
        all_reals_pred_recon.append(real_vid_pred_recon)
        all_preds_pred_recon.append(gen_vid_pred_recon)


        # ==================== B. DiT/FM Generation ====================
        unwrapped_ema = ema.module if hasattr(ema, 'module') else ema
        true_seq_len = unwrapped_ema.ae_emb_dim
        
        fm_sampler = FlowMatching(
            FMSamplingWrapper(ema), channels=args.in_channels, image_size=true_seq_len, sampling_timesteps=50
        ).to(device)
        
        running_mean = unwrapped_ema.bn.running_mean.view(1, -1, 1).to(device)
        running_var = unwrapped_ema.bn.running_var.view(1, -1, 1).to(device)
        
        running_mean_cond = unwrapped_ema.cond_bn.running_mean.view(1, -1, 1).to(device)
        running_var_cond = unwrapped_ema.cond_bn.running_var.view(1, -1, 1).to(device)
        
        if num_samples == 0:
            log_(f"--- Latent Stats Debug ---")
            log_(f"Cond Feat (GT)    | Mean: {c_feat_val.mean().item():.4f}, Std: {c_feat_val.std().item():.4f}")
            log_(f"Pred Feat (GT)    | Mean: {z_pred_gt_val.mean().item():.4f}, Std: {z_pred_gt_val.std().item():.4f}")
            log_(f"Model BN Mean     | {running_mean.mean().item():.4f}")
            log_(f"Model BN Var (sq) | {torch.sqrt(running_var).mean().item():.4f}")
            log_(f"Model cond_BN Mean| {running_mean_cond.mean().item():.4f}")
            log_(f"Model cond_BN Var | {torch.sqrt(running_var_cond).mean().item():.4f}")

        z_sampled_val = fm_sampler.sample(batch_size=b_vis, cond=c_feat_val, guidance_scale=args.cfg_scale)
        
        if num_samples == 0:
             log_(f"Sampled (Raw)     | Mean: {z_sampled_val.mean().item():.4f}, Std: {z_sampled_val.std().item():.4f}")

        z_sampled_val = z_sampled_val * torch.sqrt(running_var + 1e-4) + running_mean
        
        if num_samples == 0:
             log_(f"Sampled (DeNorm)  | Mean: {z_sampled_val.mean().item():.4f}, Std: {z_sampled_val.std().item():.4f}")
             log_(f"--------------------------")
                
        if hasattr(unwrapped_vae_pred, 'decode_from_sample'):
            pred_vis_val = unwrapped_vae_pred.decode_from_sample(z_sampled_val)
        else:
            out_vis_val = unwrapped_vae_pred.decode(z_sampled_val)
            pred_vis_val = out_vis_val.sample if hasattr(out_vis_val, 'sample') else out_vis_val
            
        pred_vis_val = pred_vis_val.clamp(-1.0, 1.0)
        
        if pred_vis_val.dim() == 4:
            T_pred = pred_vis_val.shape[0] // b_vis
            pred_vis_val = pred_vis_val.view(b_vis, T_pred, pred_vis_val.shape[1], pred_vis_val.shape[2], pred_vis_val.shape[3])
            pred_vis_val = pred_vis_val.permute(0, 2, 1, 3, 4)
            
        min_T_pred = min(x_pred_val.shape[2], pred_vis_val.shape[2])
        real_vid_pred = x_pred_val[:, :, :min_T_pred, :, :].cpu().permute(0, 2, 1, 3, 4)
        gen_vid_pred = pred_vis_val[:, :, :min_T_pred, :, :].cpu().permute(0, 2, 1, 3, 4)
        all_reals_pred.append(real_vid_pred)
        all_preds_pred.append(gen_vid_pred)
        
        num_samples += b_vis


    def _calculate_metrics_for_split(reals_list, preds_list):
        reals_btchw = torch.cat(reals_list, dim=0)[:args.eval_samples]
        preds_btchw = torch.cat(preds_list, dim=0)[:args.eval_samples]
        
        reals_01 = (reals_btchw + 1.0) / 2.0
        preds_01 = (preds_btchw + 1.0) / 2.0
        
        psnr_v = compute_psnr(reals_01, preds_01)
        
        reals_np = reals_01.numpy()
        preds_np = preds_01.numpy()
        B, T, C, H, W = reals_np.shape
        ssim_sum = 0.0
        for b in range(B):
            for t in range(T):
                img1 = np.transpose(reals_np[b, t], (1, 2, 0)) 
                img2 = np.transpose(preds_np[b, t], (1, 2, 0))
                ssim_sum += ssim_func(img1, img2, data_range=1.0, channel_axis=-1)
        ssim_v = ssim_sum / (B * T)
        
        reals_11 = reals_01 * 2 - 1
        preds_11 = preds_01 * 2 - 1
        lpips_sum = 0.0
        for b in range(B):
            r_f = reals_11[b].to(device)
            p_f = preds_11[b].to(device)
            with torch.no_grad():
                lpips_sum += lpips_fn(r_f, p_f).mean().item()
        lpips_v = lpips_sum / B
        
        reals_uint8 = (reals_01 * 255).clamp(0, 255).to(torch.uint8).to(device)
        preds_uint8 = (preds_01 * 255).clamp(0, 255).to(torch.uint8).to(device)
        
        reals_fvd = reals_uint8.permute(0, 1, 3, 4, 2) 
        preds_fvd = preds_uint8.permute(0, 1, 3, 4, 2)

        try:
            fvd_v = calculate_fvd(reals_fvd, preds_fvd, i3d, device)
            if isinstance(fvd_v, torch.Tensor): 
                fvd_v = fvd_v.item()
        except Exception as e:
            log_(f"⚠️ FVD calculation failed: {str(e)}")
            fvd_v = 0.0
            
        return psnr_v, ssim_v, lpips_v, fvd_v

    if len(all_reals_pred) == 0: return

    try:
        c_psnr, c_ssim, c_lpips, c_fvd = _calculate_metrics_for_split(all_reals_cond, all_preds_cond)
        log_(f"✅ [Eval Cond Reconstruct Step {it}] PSNR: {c_psnr:.4f} | SSIM: {c_ssim:.4f} | LPIPS: {c_lpips:.4f} | FVD: {c_fvd:.4f}")

        pr_psnr, pr_ssim, pr_lpips, pr_fvd = _calculate_metrics_for_split(all_reals_pred_recon, all_preds_pred_recon)
        log_(f"✅ [Eval Pred Reconstruct Step {it}] PSNR: {pr_psnr:.4f} | SSIM: {pr_ssim:.4f} | LPIPS: {pr_lpips:.4f} | FVD: {pr_fvd:.4f}")
        
        p_psnr, p_ssim, p_lpips, p_fvd = _calculate_metrics_for_split(all_reals_pred, all_preds_pred)
        log_(f"✅ [Eval Pred Generation Step {it}] PSNR: {p_psnr:.4f} | SSIM: {p_ssim:.4f} | LPIPS: {p_lpips:.4f} | FVD: {p_fvd:.4f}")
        
        if logger is not None:
            logger.scalar_summary('eval_pred/psnr', p_psnr, it)
            logger.scalar_summary('eval_pred/ssim', p_ssim, it)
            logger.scalar_summary('eval_pred/lpips', p_lpips, it)
            logger.scalar_summary('eval_pred/fvd', p_fvd, it)
            
            logger.scalar_summary('eval_pred_recon/psnr', pr_psnr, it)
            logger.scalar_summary('eval_pred_recon/ssim', pr_ssim, it)
            logger.scalar_summary('eval_pred_recon/lpips', pr_lpips, it)
            logger.scalar_summary('eval_pred_recon/fvd', pr_fvd, it)
            
    except Exception as e:
        log_(f"❌ Evaluation process exception: {str(e)}")
        
    finally:
        del lpips_fn
        torch.cuda.empty_cache()


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
    args.vae_max_iter = config.vae.max_iter
    args.log_freq = config.model.log_freq
    args.eval_freq = config.model.eval_freq
    args.max_size = config.model.max_size if 'max_size' in config.model else None
    args.eval_samples = config.model.eval_samples if 'eval_samples' in config.model else 16
    args.resume = config.model.resume if 'resume' in config.model else False
    # NOTE:
    # `model.params.w` comes from legacy diffusion configs and is often set to 0.
    # Re-using it as CFG scale silently disables conditional guidance during FM sampling.
    # Prefer explicit `model.cfg_scale` (or `model.params.cfg_scale`) for this code path.
    if 'cfg_scale' in config.model:
        args.cfg_scale = config.model.cfg_scale
    elif 'cfg_scale' in config.model.params:
        args.cfg_scale = config.model.params.cfg_scale
    return args
