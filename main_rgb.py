import os
import time
import copy
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
import torchvision
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import OrderedDict

import numpy as np
import random

from tools.dataloader import get_loaders
from tools.utils import AverageMeter, setup_distibuted_training, setup_logger, set_random_seed

from models.vae.vae_vit_rope import ViTAutoencoder
from models.fm.DiT import DiT, DiT_models
from losses.perceptual import LPIPSWithDiscriminator

from losses.fm import FlowMatching

_num_moments = 3  
_reduce_dtype = torch.float32  
_counter_dtype = torch.float64  
_rank = 0  
_sync_device = None  
_sync_called = False  

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

@torch.no_grad()
def run_evaluation(val_loader, ema, vae_cond_model, vae_pred_model, args, device, it, logger, log_):
    """
    Quantitative evaluation on the validation set: PSNR, SSIM, LPIPS, FVD.
    """
    try:
        from evals.fvd.fvd import calculate_fvd
        from losses.lpips import LPIPS
        from skimage.metrics import structural_similarity as ssim_func
        import numpy as np
    except ImportError as e:
        log_(f"❌ Failed to import evaluation libraries (check skimage, scipy, etc.): {e}")
        return

    log_(f"🚀 Running quantitative evaluation (target samples: {args.eval_samples})...")
    
    lpips_fn = LPIPS().eval().to(device)
    
    all_reals = []
    all_preds = []
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

    if len(all_reals) == 0:
        return

    reals_btchw = torch.cat(all_reals, dim=0)[:args.eval_samples]
    preds_btchw = torch.cat(all_preds, dim=0)[:args.eval_samples]
    
    reals_01 = (reals_btchw + 1.0) / 2.0
    preds_01 = (preds_btchw + 1.0) / 2.0
    
    try:
        # =======================================================
        # 1. Compute PSNR
        # =======================================================
        psnr_val = compute_psnr(reals_01, preds_01)
        
        # =======================================================
        # 2. Compute SSIM
        # =======================================================
        reals_np = reals_01.numpy()
        preds_np = preds_01.numpy()
        B, T, C, H, W = reals_np.shape
        ssim_sum = 0.0
        for b in range(B):
            for t in range(T):
                img1 = np.transpose(reals_np[b, t], (1, 2, 0)) 
                img2 = np.transpose(preds_np[b, t], (1, 2, 0))
                s = ssim_func(img1, img2, data_range=1.0, channel_axis=-1)
                ssim_sum += s
        ssim_val = ssim_sum / (B * T)
        
        # =======================================================
        # 3. Compute LPIPS
        # =======================================================
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
        
        # =======================================================
        # 4. Compute FVD 
        # =======================================================
        reals_uint8 = (reals_01 * 255).clamp(0, 255).to(torch.uint8).to(device)
        preds_uint8 = (preds_01 * 255).clamp(0, 255).to(torch.uint8).to(device)
        
        try:
            fvd_val = calculate_fvd(reals_uint8, preds_uint8, device)
            if isinstance(fvd_val, torch.Tensor): fvd_val = fvd_val.item()
        except Exception as e:
            log_(f"⚠️ FVD calculation failed, skipping this metric: {str(e)}")
            fvd_val = 0.0
            
        log_(f"✅ [Eval Report Step {it}] PSNR: {psnr_val:.4f} | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f} | FVD: {fvd_val:.4f}")
        
        if logger is not None:
            logger.scalar_summary('eval/psnr', psnr_val, it)
            logger.scalar_summary('eval/ssim', ssim_val, it)
            logger.scalar_summary('eval/lpips', lpips_val, it)
            logger.scalar_summary('eval/fvd', fvd_val, it)

    except Exception as e:
        log_(f"❌ Evaluation process exception: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        torch.cuda.empty_cache()

class FMSamplingWrapper(nn.Module):
    def __init__(self, dit_model):
        super().__init__()
        self.dit_model = dit_model.module if hasattr(dit_model, 'module') else dit_model

    def forward(self, x, cond, t):
        return self.dit_model.forward_sampling(x, cond, t)
    
def save_image_grid(img, fname, drange, grid_size=None, normalize=True):
    import numpy as np
    from PIL import Image
    if normalize:
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

    B, C, T, H, W = img.shape
    img = img.transpose(0, 3, 2, 4, 1) 
    img = img.reshape(B * H, T * W, C)
    if C == 1:
        img = np.repeat(img, 3, axis=2)

    result_img = Image.fromarray(img, 'RGB')
    result_img.save(fname, quality=95)

def log_videos_e2e(gts, predictions, it, save_dir):
    """Concatenate and save ground truth and predicted videos"""
    import numpy as np
    import os

    gts_np = gts.detach().cpu().numpy()
    preds_np = predictions.detach().cpu().numpy()

    combined = np.stack([gts_np, preds_np], axis=1)
    combined = combined.reshape(-1, *gts_np.shape[1:])

    save_path = os.path.join(save_dir, f'vis_e2e_{it:07d}.png')
    save_image_grid(combined, save_path, drange=[-1, 1])

def init_multiprocessing(rank, sync_device):
    global _rank, _sync_device
    assert not _sync_called
    _rank = rank
    _sync_device = sync_device

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

# ============================================================================
# 4. End-to-end core training loop 
# ============================================================================
def main(rank, args):
    device = torch.device('cuda', rank)
    setup_distibuted_training(args, rank)
    sync_device = torch.device('cuda', rank) if args.n_gpus > 1 else None
    init_multiprocessing(rank=rank, sync_device=sync_device)
    torch.cuda.set_device(rank)
    log_, logger = setup_logger(args, rank)

    train_loader, val_loader, _ = get_loaders(rank, copy.deepcopy(args))
    if rank == 0: 
        args.output = logger.logdir 
        log_(f"Loading dataset {args.data} with resolution {args.res}")
        
    # ======================== Initialize models ========================
    vae_pred_model = ViTAutoencoder(args.embed_dim, args.vaeconfig).to(device)
    vae_pred_criterion = LPIPSWithDiscriminator(
        disc_start=args.lossconfig.params.disc_start,
        timesteps=args.pred_frames,
        perceptual_weight=args.perceptual_weight
    ).to(device)

    vae_cond_model = ViTAutoencoder(args.embed_dim, args.cond_vaeconfig).to(device)
    vae_cond_criterion = LPIPSWithDiscriminator(
        disc_start=args.lossconfig.params.disc_start,
        timesteps=args.cond_frames,
        perceptual_weight=args.perceptual_weight
    ).to(device)

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = DiT_models[args.model](
        input_size=args.input_size,
        in_channels=args.in_channels,
        z_dims=[384],
        encoder_depth=args.encoder_depth,
        bn_momentum=args.bn_momentum,
        **block_kwargs
    ).to(device)
    
    ema = copy.deepcopy(model).to(device)  
    update_ema(ema, model, decay=0)

    teacher_model = get_teacher_model(args.align_model, device, args.align_ckpt_dir)
        
    # ======================== DDP Wrapping ========================
    vae_pred_model = nn.parallel.DistributedDataParallel(vae_pred_model, device_ids=[device], find_unused_parameters=False)
    vae_pred_criterion = nn.parallel.DistributedDataParallel(vae_pred_criterion, device_ids=[device], find_unused_parameters=False)
    vae_cond_model = nn.parallel.DistributedDataParallel(vae_cond_model, device_ids=[device], find_unused_parameters=False)
    vae_cond_criterion = nn.parallel.DistributedDataParallel(vae_cond_criterion, device_ids=[device], find_unused_parameters=False)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True) 

    # ======================== Optimizers ========================
    if args.scale_lr: args.lr *= args.batch_size
    
    opt_vae_g = torch.optim.AdamW(
        list(vae_pred_model.parameters()) + list(vae_cond_model.parameters()),
        lr=args.lr, betas=(0.5, 0.9)
    )
    opt_vae_d = torch.optim.AdamW(
        list(vae_pred_criterion.module.discriminator_2d.parameters()) + 
        list(vae_pred_criterion.module.discriminator_3d.parameters()) +
        list(vae_cond_criterion.module.discriminator_2d.parameters()) + 
        list(vae_cond_criterion.module.discriminator_3d.parameters()), 
        lr=args.lr, betas=(0.5, 0.9)
    )
    opt_dit = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler_vae = CosineAnnealingLR(opt_vae_g, T_max=args.max_iter, eta_min=args.lr / 100)
    scheduler_dit = CosineAnnealingLR(opt_dit, T_max=args.max_iter, eta_min=0)

    scaler_g = torch.amp.GradScaler('cuda') if args.amp else None
    scaler_d = torch.amp.GradScaler('cuda') if args.amp else None
    scaler_fm = torch.amp.GradScaler('cuda') if args.amp else None

    # ======================== TensorBoard Loss Dictionary ========================
    losses = dict()
    losses['vae_cond_recon'] = AverageMeter()
    losses['vae_pred_recon'] = AverageMeter()
    losses['vae_repa'] = AverageMeter()
    losses['dit_denoise'] = AverageMeter()
    losses['dit_repa'] = AverageMeter()
    losses['d_loss'] = AverageMeter()

    # ======================== Training Loop ========================
    accum_iter = 3
    disc_opt = False
    disc_start = vae_pred_criterion.module.discriminator_iter_start 
    last_it = 0
    
    if rank == 0:
        os.makedirs(args.output, exist_ok=True)
    
    pbar = tqdm(total=args.max_iter, initial=last_it, dynamic_ncols=True, disable=(rank != 0))

    for it, (x, _) in enumerate(train_loader):
        it += last_it
        if it > args.max_iter: break
        pbar.update(1)
        
        batch_size = x.size(0)
        x = x.to(device)
        x_vae = rearrange(x / 127.5 - 1, 'b t c h w -> b c t h w') 
        
        x_cond = x_vae[:, :, :args.cond_frames] 
        x_pred = x_vae[:, :, args.cond_frames : args.cond_frames + args.frames]
        
        if not disc_opt:
            # -------------------------------------------------------------
            # Stage 1: Dual VAE Generator + REPA
            # -------------------------------------------------------------
            set_requires_grad(vae_pred_model, True)
            set_requires_grad(vae_cond_model, True)
            set_requires_grad(model, False) 
            model.eval() 
            
            with torch.autocast(device_type='cuda', enabled=args.amp):
                align_target = get_align_targets(x_pred, teacher_model, args.align_model)

                x_cond_hat, kl_loss_cond = vae_cond_model(x_cond)
                z_cond = vae_cond_model.module.extract(x_cond)
                vae_cond_loss, _, _ = vae_cond_criterion(
                    kl_loss_cond, x_cond, rearrange(x_cond_hat, '(b t) c h w -> b c t h w', b=batch_size),
                    optimizer_idx=0, global_step=it
                )
                
                x_traget_hat, kl_loss_pred = vae_pred_model(x_pred)
                z_pred = vae_pred_model.module.extract(x_pred)
                vae_pred_loss, _, _ = vae_pred_criterion(
                    kl_loss_pred, x_pred, rearrange(x_traget_hat, '(b t) c h w -> b c t h w', b=batch_size),
                    optimizer_idx=0, global_step=it
                )
                
                _, cond = prepare_input(args, x_vae, vae_cond_model, vae_pred_model)
                vae_loss = (vae_cond_loss + vae_pred_loss) / accum_iter
                
                # Fix: Use .module to bypass DDP wrapper for frozen DiT to avoid warnings
                vae_align_outputs = model.module(
                    x=z_pred, cond=cond, align_target=align_target,
                    time_input=None, noises=None, align_only=True
                )

                vae_repa_val = vae_align_outputs["proj_loss"].mean()
                vae_loss = vae_loss + (args.vae_align_proj_coeff * vae_repa_val / accum_iter)
                
                time_input = vae_align_outputs["time_input"]
                noises = vae_align_outputs["noises"]

            if args.amp: scaler_g.scale(vae_loss).backward()
            else: vae_loss.backward()

            losses['vae_cond_recon'].update(vae_cond_loss.item(), 1)
            losses['vae_pred_recon'].update(vae_pred_loss.item(), 1)
            losses['vae_repa'].update(vae_repa_val.item(), 1)

            # -------------------------------------------------------------
            # Stage 2: DiT Flow Matching
            # -------------------------------------------------------------
            set_requires_grad(vae_pred_model, False) 
            set_requires_grad(vae_cond_model, False)
            set_requires_grad(model, True)
            model.train()
            
            z_pred_detached = z_pred.detach()
            cond_detached = cond.detach()
            
            with torch.autocast(device_type='cuda', enabled=args.amp):
                sit_outputs = model(
                    x=z_pred_detached, cond=cond_detached, align_target=align_target,
                    time_input=time_input, noises=noises, align_only=False
                )
                
                dit_denoise_val = sit_outputs["denoising_loss"].mean()
                dit_repa_val = sit_outputs["proj_loss"].mean()
                
                dit_loss = (dit_denoise_val + args.dit_align_proj_coeff * dit_repa_val) / accum_iter
                
            if args.amp: scaler_fm.scale(dit_loss).backward()
            else: dit_loss.backward()

            losses['dit_denoise'].update(dit_denoise_val.item(), 1)
            losses['dit_repa'].update(dit_repa_val.item(), 1)

            if it % accum_iter == accum_iter - 1:
                if args.amp:
                    scale_before_g = scaler_g.get_scale()
                    scale_before_fm = scaler_fm.get_scale()
                    
                    scaler_g.step(opt_vae_g)
                    scaler_g.update()
                    scaler_fm.step(opt_dit)
                    scaler_fm.update()
                    
                    if scaler_g.get_scale() >= scale_before_g:
                        scheduler_vae.step()
                    if scaler_fm.get_scale() >= scale_before_fm:
                        scheduler_dit.step()
                else:
                    opt_vae_g.step()
                    opt_dit.step()
                    scheduler_vae.step()
                    scheduler_dit.step()
                    
                opt_vae_g.zero_grad(set_to_none=True)
                opt_dit.zero_grad(set_to_none=True)
                
                unwrapped_model = model.module if hasattr(model, 'module') else model
                update_ema(ema, unwrapped_model)

        else:
            # -------------------------------------------------------------
            # Stage 3: Discriminator Training
            # -------------------------------------------------------------
            if it % accum_iter == 0: 
                vae_pred_criterion.zero_grad(set_to_none=True)
                vae_cond_criterion.zero_grad(set_to_none=True)
                
            with torch.autocast(device_type='cuda', enabled=args.amp):
                with torch.no_grad(): 
                    # Fix: Use .module for eval layers to avoid DDP warnings
                    x_pred_tilde, vq_loss_pred = vae_pred_model.module(x_pred)
                    x_cond_tilde, vq_loss_cond = vae_cond_model.module(x_cond)
                
                d_loss_pred = vae_pred_criterion(vq_loss_pred, x_pred, rearrange(x_pred_tilde, '(b t) c h w -> b c t h w', b=batch_size), optimizer_idx=1, global_step=it)
                d_loss_cond = vae_cond_criterion(vq_loss_cond, x_cond, rearrange(x_cond_tilde, '(b t) c h w -> b c t h w', b=batch_size), optimizer_idx=1, global_step=it)
                
                d_loss_total = (d_loss_pred + d_loss_cond) / accum_iter

            if args.amp:
                scaler_d.scale(d_loss_total).backward()
                if it % accum_iter == accum_iter - 1:
                    scaler_d.unscale_(opt_vae_d)
                    torch.nn.utils.clip_grad_norm_(vae_pred_criterion.module.discriminator_2d.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(vae_pred_criterion.module.discriminator_3d.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(vae_cond_criterion.module.discriminator_2d.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(vae_cond_criterion.module.discriminator_3d.parameters(), 1.0)
                    scaler_d.step(opt_vae_d); scaler_d.update()
            else:
                d_loss_total.backward()
                if it % accum_iter == accum_iter - 1: opt_vae_d.step()

            losses['d_loss'].update(d_loss_total.item() * accum_iter, 1)

        if it % accum_iter == accum_iter - 1 and it // accum_iter >= disc_start:
            disc_opt = not disc_opt

        # =========================================================================
        # TensorBoard & Tqdm Updates
        # =========================================================================
        if rank == 0 and it % 10 == 0:
            if not disc_opt:
                pbar.set_description(f"VAE Tgt: {losses['vae_pred_recon'].average:.4f} | DiT Denoise: {losses['dit_denoise'].average:.4f}")
            else:
                pbar.set_description(f"Disc Loss: {losses['d_loss'].average:.4f}")
                
        if it % args.log_freq == 0:
            if rank == 0 and logger is not None:
                logger.scalar_summary('train/vae_cond_recon', losses['vae_cond_recon'].average, it)
                logger.scalar_summary('train/vae_pred_recon', losses['vae_pred_recon'].average, it)
                logger.scalar_summary('train/vae_repa_loss', losses['vae_repa'].average, it)
                logger.scalar_summary('train/dit_denoise_loss', losses['dit_denoise'].average, it)
                logger.scalar_summary('train/dit_repa_loss', losses['dit_repa'].average, it)
                logger.scalar_summary('train/d_loss', losses['d_loss'].average, it)
                
                logger.scalar_summary('train/lr_vae', scheduler_vae.get_lr()[0], it)
                logger.scalar_summary('train/lr_dit', scheduler_dit.get_lr()[0], it)

            # Clear cached Meters across all processes to prevent memory leaks or desync
            for k in losses.keys():
                losses[k] = AverageMeter()

        # =========================================================================
        # Periodic Video Generation and Visualization
        # =========================================================================
        if rank == 0 and it % args.eval_freq == 0 and it > 0:
            ema.eval()
            with torch.no_grad():
                try:
                    b_vis = 1
                    c_init_vis = x_cond[:b_vis].clone()
                    gt_pred_vis = x_pred[:b_vis].clone().cpu()
                    
                    c_feat_vis = vae_cond_model.module.extract(c_init_vis).detach()
                    
                    unwrapped_ema = ema.module if hasattr(ema, 'module') else ema
                    true_seq_len = unwrapped_ema.ae_emb_dim
                    
                    fm_sampler = FlowMatching(
                        FMSamplingWrapper(ema), 
                        channels=args.in_channels, 
                        image_size=true_seq_len, 
                        sampling_timesteps=50
                    ).to(device)
                    
                    z_sampled = fm_sampler.sample(batch_size=b_vis, cond=c_feat_vis)
                    
                    if hasattr(vae_pred_model.module, 'decode_from_sample'):
                        pred_vis = vae_pred_model.module.decode_from_sample(z_sampled)
                    else:
                        out_vis = vae_pred_model.module.decode(z_sampled)
                        pred_vis = out_vis.sample if hasattr(out_vis, 'sample') else out_vis
                        
                    pred_vis = pred_vis.clamp(-1.0, 1.0).cpu()
                    
                    if pred_vis.dim() == 4:
                        T_pred = pred_vis.shape[0] // b_vis
                        pred_vis = pred_vis.view(b_vis, T_pred, pred_vis.shape[1], pred_vis.shape[2], pred_vis.shape[3])
                        pred_vis = pred_vis.permute(0, 2, 1, 3, 4)
                        
                    min_T = min(gt_pred_vis.shape[2], pred_vis.shape[2])
                    gt_pred_vis = gt_pred_vis[:, :, :min_T, :, :]
                    pred_vis = pred_vis[:, :, :min_T, :, :]
                    
                    combined = torch.stack([gt_pred_vis, pred_vis], dim=1) 
                    
                    flat_combined = combined.permute(0, 1, 3, 2, 4, 5).contiguous()
                    flat_combined = flat_combined.view(-1, flat_combined.shape[3], flat_combined.shape[4], flat_combined.shape[5])
                    
                    flat_combined = (flat_combined + 1.0) / 2.0
                    flat_combined = flat_combined.clamp(0.0, 1.0)
                    
                    save_path = os.path.join(args.output, f'vis_e2e_{it:07d}.png')
                    torchvision.utils.save_image(flat_combined, save_path, nrow=min_T, normalize=False)
                    
                except Exception as e:
                    log_(f"❌ Visualization error (skipped): {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                finally:
                    torch.cuda.empty_cache()

        # =========================================================================
        # Save Model Checkpoints
        # =========================================================================
        if rank == 0 and it % 10000 == 0 and it > 0:
            ckpt_path = os.path.join(args.output, f'ckpt_{it:07d}.pt')
            torch.save({
                'vae_pred_model': vae_pred_model.module.state_dict(),
                'vae_cond_model': vae_cond_model.module.state_dict(),
                'dit_model': model.module.state_dict(),
                'ema_model': ema.state_dict(),
                'opt_vae_g': opt_vae_g.state_dict(),
                'opt_vae_d': opt_vae_d.state_dict(),
                'opt_dit': opt_dit.state_dict(),
                'it': it,
            }, ckpt_path)
            log_(f"Saved full checkpoint to: {ckpt_path}")
            
        # =========================================================================
        # Quantitative Evaluation
        # =========================================================================
        if rank == 0 and it % args.eval_freq == 0 and it > 0:
            ema.eval()
            vae_cond_model.eval()
            vae_pred_model.eval()
            
            run_evaluation(
                val_loader=val_loader,
                ema=ema,
                vae_cond_model=vae_cond_model,
                vae_pred_model=vae_pred_model,
                args=args,
                device=device,
                it=it,
                logger=logger,
                log_=log_
            )
            
            ema.train()
            vae_cond_model.train()
            vae_pred_model.train()

    pbar.close()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file for all next arguments")
    parser.add_argument('--exp', type=str, default='e2e', help='Type of training to run [vae, fm, mmfm, e2e]')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--n_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for dataloader')
    parser.add_argument('--output', type=str, default='./results_e2e', help='Output directory where to store exp results')

    # Data Args
    parser.add_argument('--data', type=str, default='CITYSCAPES_RGB')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_folder', type=str, default='/mnt/data/wangsen/SyncVP/data')

    # VAE Args
    parser.add_argument('--vae_config', type=str, default='configs/test.yaml')

    # DiT Args
    parser.add_argument("--model", type=str, default="DiT-S", choices=DiT_models.keys(), help="The model to train.")
    parser.add_argument('--rgb_model', type=str, default='')
    parser.add_argument('--depth_model', type=str, default='')
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bn-momentum", type=float, default=0.1)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)

    # Lr scheduler & Eval Settings
    parser.add_argument('--no_sched', action='store_true')
    parser.add_argument('--scale_lr', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--samples', type=int, default=0)
    parser.add_argument('--traj', type=int, default=1)
    parser.add_argument('--NFE', type=int, default=25)
    parser.add_argument('--no_depth_cond', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--future_frames', type=int, default=28)
    
    # Vision Foundation Model
    parser.add_argument("--align_model", type=str, default="VideoMAEv2", choices=["VideoMAEv2", "VideoMAE", "OminiMAE", "DINOv3", "VJEPA", "VJEPA2"])
    parser.add_argument("--align_ckpt_dir", type=str, default="./ckpts")
    
    parser.add_argument('--vae_align_proj_coeff', type=float, default=1.0)
    parser.add_argument('--dit_align_proj_coeff', type=float, default=0.5)

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        for key, value in yaml_config.items():
            setattr(args, key, value)

    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if os.path.exists('.torch_distributed_init'):
        os.remove('.torch_distributed_init')
        
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.exp == 'e2e':
        args = config_setup(args)
        runner = main
    else:
        pass 

    if args.n_gpus == 1 or args.eval:
        runner(rank=0, args=args)
    else:
        torch.multiprocessing.spawn(fn=runner, args=(args,), nprocs=args.n_gpus)