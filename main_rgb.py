import os
import time
import copy
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from tqdm import tqdm
from glob import glob
from einops import rearrange
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from tools.dataloader import get_loaders
from tools.utils import AverageMeter, setup_distibuted_training, setup_logger, set_random_seed

from models.vae.vae_vit_rope import ViTAutoencoder
from models.fm.DiT import DiT, DiT_models
from losses.perceptual import LPIPSWithDiscriminator
from losses.fm import FlowMatching

from models.ssl.videomaev2 import vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch16_224, vit_giant_patch14_224
from models.ssl.videomae import vit_base_patch16_224 as VideoMAE_vit_base_patch16_224


_num_moments = 3  
_reduce_dtype = torch.float32  
_counter_dtype = torch.float64  
_rank = 0  
_sync_device = None  
_sync_called = False  
_counters = dict()  
_cumulative = dict()  

def init_multiprocessing(rank, sync_device):
    global _rank, _sync_device
    assert not _sync_called
    _rank = rank
    _sync_device = sync_device

def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def get_teacher_model(model_name, device, ckpt_dir):
    """
    Initialize and return the frozen video foundation model (Teacher Model).
    """
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
        model.tubelet_size = 2
        model.patch_size = 16
        model.embed_dim = 768
        
    elif model_name == 'VJEPA':
        from models.ssl.JEPA import load_VJEPA
        model = load_VJEPA(device=device, pretrained_path=os.path.join(ckpt_dir, 'vjepa_l/vitl16.pth.tar'))
        
    elif model_name == "VJEPA2":
        model, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')
        model = model.to(device)
        if hasattr(model, 'norm'):
            model.norm = torch.nn.Identity()
                    
    elif model_name == 'DINOv3':
        model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16').to(device)
        if hasattr(model, 'head'):
            model.head = torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported align model: {model_name}")

    model.eval()
    set_requires_grad(model, False) 
    return model

def get_align_targets(x_pred, teacher_model, align_model_name, patch_size=16, tubelet_size=2):
    """
    Unified processing of feature extraction for video or image foundation models, used for REPA alignment.
    
    Parameters:
        x_pred: Tensor of shape [B, C, F, H, W], expected value range is [-1, 1]
        teacher_model: Frozen vision foundation model (Teacher)
        align_model_name: Name of the foundation model used (e.g., 'VideoMAEv2', 'DINOv3', etc.)
        patch_size: Spatial patch size (default 16)
        tubelet_size: Temporal tubelet size (default 2)
        
    Returns:
        align_target: Tensor of shape [B, L, D], where L is the sequence length and D is the feature dimension
    """
    B, C, F, H, W = x_pred.shape

    # ==========================================
    # 1. Normalization preprocessing ([-1, 1] -> [0, 1] -> Normalize)
    # ==========================================
    frames_01 = (x_pred + 1.0) / 2.0
    
    # Flatten the temporal dimension into the Batch dimension to use Normalize: [B*F, C, H, W]
    frames_flat = frames_01.transpose(1, 2).flatten(0, 1)

    # DINO and VideoMAE series typically use the ImageNet default mean and variance
    norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    frames_norm = Normalize(norm_mean, norm_std)(frames_flat)
    
    # Restore 3D video dimensions: [B, C, F, H, W]
    repa_raw_frames = frames_norm.reshape(B, F, C, H, W).transpose(1, 2)

    # Dimension safety check (ensure 8x128x128 can be perfectly divided into patches)
    assert F % tubelet_size == 0, f"Frame {F} must be divisible by tubelet_size {tubelet_size}"
    assert H % patch_size == 0 and W % patch_size == 0, f"H, W must be divisible by {patch_size}"

    # ==========================================
    # 2. Foundation model feature extraction
    # ==========================================
    with torch.no_grad():
        if align_model_name in ['VideoMAEv2', 'VideoMAE', 'OminiMAE', 'VJEPA', 'VJEPA2']:
            # 3D video model input requirement: [B, C, F, H, W]
            features = teacher_model(repa_raw_frames)
            align_target = features
                
        elif align_model_name in ['DINOv2', 'DINOv3']:
            frames_2d = repa_raw_frames.transpose(1, 2).flatten(0, 1)
            group_size = 32
            chunked = frames_2d.chunk(max(1, frames_2d.shape[0] // group_size), dim=0)
            features_list = []
            for chunk in chunked:
                out = teacher_model.forward_features(chunk)
                patch_tokens = out['x_norm_patchtokens'] 
                features_list.append(patch_tokens)
            features_cat = torch.cat(features_list, dim=0) # [B*F, L_2d, D]
            _, L_2d, D = features_cat.shape
            align_target = features_cat.reshape(B, F * L_2d, D)
            
        else:
            raise NotImplementedError(f"Alignment model {align_model_name} is not supported.")

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
    
    # --- C. FM (DiT) specific training configurations ---
    args.fmconfig = config.model.params
    args.cond_prob = config.model.cond_prob if 'cond_prob' in config.model else 0.0
    args.same_noise = config.model.same_noise if 'same_noise' in config.model else True
    args.input_size = config.model.sit_config.input_size
    args.in_channels = config.model.sit_config.in_channels
    args.encoder_depth = config.model.sit_config.encoder_depth
    args.bn_momentum = config.model.sit_config.bn_momentum

    # --- D. Global/Shared configurations ---
    args.lr = config.model.base_learning_rate if 'base_learning_rate' in config.model else 1e-4
    args.max_iter = config.model.max_iter
    # args.res = ae_pred_config.model.params.ddconfig.resolution
    # args.frames = ae_pred_config.model.params.ddconfig.frames
    args.log_freq = config.model.log_freq
    args.eval_freq = config.model.eval_freq
    args.max_size = config.model.max_size if 'max_size' in config.model else None
    args.eval_samples = config.model.eval_samples if 'eval_samples' in config.model else 16
    args.resume = config.model.resume if 'resume' in config.model else False

    return args

# ============================================================================
# 4. End-to-end core training loop (supports dual VAE synchronous co-training)
# ============================================================================
def main(rank, args):
    device = torch.device('cuda', rank)

    setup_distibuted_training(args, rank)
    sync_device = torch.device('cuda', rank) if args.n_gpus > 1 else None
    init_multiprocessing(rank=rank, sync_device=sync_device)
    torch.cuda.set_device(rank)
    log_, logger = setup_logger(args, rank)

    train_loader, val_loader, _ = get_loaders(rank, copy.deepcopy(args))
    if rank == 0: log_(f"Loading dataset {args.data} with resolution {args.res}")
    if rank == 0: log_("Initializing pred VAE (8-frames) and Condition VAE (2-frames)...")

    # ======================== Initialize dual VAE models ========================
    # 1. pred VAE (8-frames)
    vae_pred_model = ViTAutoencoder(args.embed_dim, args.vaeconfig).to(device)
    vae_pred_criterion = LPIPSWithDiscriminator(
        disc_start=args.lossconfig.params.disc_start,
        timesteps=args.pred_frames,
        perceptual_weight=args.perceptual_weight
    ).to(device)

    # 2. Cond VAE (2-frames)
    vae_cond_model = ViTAutoencoder(args.embed_dim, args.cond_vaeconfig).to(device)
    vae_cond_criterion = LPIPSWithDiscriminator(
        disc_start=args.lossconfig.params.disc_start,
        timesteps=args.cond_frames,
        perceptual_weight=args.perceptual_weight
    ).to(device)

    # 3. DiT Flow Matching Model
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = DiT_models[args.model](
        input_size=args.input_size,
        in_channels=args.in_channels,
        z_dims=[384],
        encoder_depth=args.encoder_depth,
        bn_momentum=args.bn_momentum,
        **block_kwargs
    ).to(device)
    # make a copy of the model for EMA
    model = model.to(device)
    ema = copy.deepcopy(model).to(device)  # Create an EMA of the model for use after training

    fm_criterion = FlowMatching(
        model, 
        channels=args.in_channels, 
        image_size=args.res, loss_type='l2'
    ).to(device)

    teacher_model = get_teacher_model(args.align_model, device, args.align_ckpt_dir)
    if rank == 0:
        log_(
            f"Params (M) -> VAE_Pred: {sum(p.numel() for p in vae_pred_model.parameters())/1e6:.2f}M, "
            f"VAE_Cond: {sum(p.numel() for p in vae_cond_model.parameters())/1e6:.2f}M, "
            f"DiT: {sum(p.numel() for p in model.parameters())/1e6:.2f}M, "
            f"{args.align_model}: {sum(p.numel() for p in teacher_model.parameters())/1e6:.2f}M"
        ) 
        
    # ======================== DDP Wrapping ========================
    # Enable find_unused_parameters=True because the model skips the final prediction head when align_only=True
    vae_pred_model = nn.parallel.DistributedDataParallel(vae_pred_model, device_ids=[device], find_unused_parameters=False)
    vae_pred_criterion = nn.parallel.DistributedDataParallel(vae_pred_criterion, device_ids=[device], find_unused_parameters=False)
    
    vae_cond_model = nn.parallel.DistributedDataParallel(vae_cond_model, device_ids=[device], find_unused_parameters=False)
    vae_cond_criterion = nn.parallel.DistributedDataParallel(vae_cond_criterion, device_ids=[device], find_unused_parameters=False)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True) 

    # ======================== Optimizers ========================
    if args.scale_lr: args.lr *= args.batch_size
    
    # Generator optimizer: includes generators of both VAEs + projection head
    opt_vae_g = torch.optim.AdamW(
        list(vae_pred_model.parameters()) + 
        list(vae_cond_model.parameters()),
        lr=args.lr, betas=(0.5, 0.9)
    )
    
    # Discriminator optimizer: includes discriminators of both VAEs
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

    scaler_g = GradScaler() if args.amp else None
    scaler_d = GradScaler() if args.amp else None
    scaler_fm = GradScaler() if args.amp else None

    # ======================== Training Loop ========================
    accum_iter = 3
    disc_opt = False
    disc_start = vae_pred_criterion.module.discriminator_iter_start # Based on pred VAE
    last_it = 0
    
    pbar = tqdm(total=args.max_iter, initial=last_it, dynamic_ncols=True, disable=(rank != 0))

    for it, (x, _) in enumerate(train_loader):
        it += last_it
        if it > args.max_iter: break
        pbar.update(1)
        
        batch_size = x.size(0)
        x = x.to(device)
        x_vae = rearrange(x / 127.5 - 1, 'b t c h w -> b c t h w') # [-1, 1]
        
        # --- Data split: 2-frame condition (cond) and 8-frame target (pred) ---
        x_cond = x_vae[:, :, :args.cond_frames] 
        x_pred = x_vae[:, :, args.cond_frames : args.cond_frames + args.frames]
        
        align_target = get_align_targets(x_pred, teacher_model, args.align_model)

        if not disc_opt:
            # -------------------------------------------------------------
            # Stage 1: Dual VAE Generator Training + REPA (Target alignment only)
            # -------------------------------------------------------------
            set_requires_grad(vae_pred_model, True)
            set_requires_grad(vae_cond_model, True)
            set_requires_grad(model, True) 
            
            with autocast(enabled=args.amp):
                # 1). Forward pass: VAE
                z_cond, x_cond_hat, kl_loss_cond = vae_cond_model(x_cond)
                vae_cond_loss, _, _ = vae_cond_criterion(
                    kl_loss_cond, x_cond, rearrange(x_cond_hat, '(b t) c h w -> b c t h w', b=batch_size),
                    optimizer_idx=0, global_step=it
                )
                
                z_pred, x_traget_hat, kl_loss_pred = vae_pred_model(x_pred)
                vae_pred_loss, _, _ = vae_pred_criterion(
                    kl_loss_pred, x_pred, rearrange(x_traget_hat, '(b t) c h w -> b c t h w', b=batch_size),
                    optimizer_idx=0, global_step=it
                )
                
                vae_loss = (vae_cond_loss + vae_pred_loss) / accum_iter
                
                # 2). Backward pass: VAE, compute the VAE loss, backpropagate, and update the VAE; Then, compute the discriminator loss and update the discriminator
                # Record the time_input and noises for the VAE alignment, so that we avoid sampling again
                time_input = None
                noises = None

                # Turn off grads for the DiT model (avoid REPA gradient on the DiT model)
                requires_grad(model, False)
                # Avoid BN stats to be updated by the VAE
                model.eval()
                
                # Compute the REPA alignment loss for VAE updates
                vae_align_outputs = model(
                    x=x_traget_hat,
                    align_target=align_target,
                    time_input=time_input,
                    noises=noises,
                    align_only=True
                )

                vae_loss = vae_loss + args.vae_align_proj_coeff * vae_align_outputs["proj_loss"].mean()
                # Save the `time_input` and `noises` and reuse them for the DiT model forward pass
                time_input = vae_align_outputs["time_input"]
                noises = vae_align_outputs["noises"]

                
            # if args.amp:
            #     scaler_g.scale(loss_g_phase1).backward(retain_graph=True)
            # else:
            #     loss_g_phase1.backward(retain_graph=True)

            # -------------------------------------------------------------
            # Stage 2: DiT Flow Matching (Completely cut off gradients for both VAEs!)
            # -------------------------------------------------------------
            z_pred_detached = z_pred.detach()
            z_cond_detached = z_cond.detach()
            
            set_requires_grad(vae_pred_model, False) 
            set_requires_grad(vae_cond_model, False)
            
            with autocast(enabled=args.amp):
                # Compute vector field matching Loss (Target: z_pred_detached, Condition: z_cond_detached)
                loss_fm, _ = fm_criterion(z_pred_detached.float(), z_cond_detached.float())
                
            if args.amp:
                scaler_fm.scale(loss_fm).backward()
            else:
                loss_fm.backward()

            # --- Step and Update ---
            if it % accum_iter == accum_iter - 1:
                if args.amp:
                    scaler_g.step(opt_vae_g); scaler_g.update()
                    scaler_fm.step(opt_dit); scaler_fm.update()
                else:
                    opt_vae_g.step()
                    opt_dit.step()
                opt_vae_g.zero_grad()
                opt_dit.zero_grad()
                scheduler_vae.step()
                scheduler_dit.step()
        else:
            # -------------------------------------------------------------
            # Stage 3: Dual VAE Discriminator Synchronous Training
            # -------------------------------------------------------------
            if it % accum_iter == 0: 
                vae_pred_criterion.zero_grad()
                vae_cond_criterion.zero_grad()
                
            with autocast(enabled=args.amp):
                with torch.no_grad(): 
                    x_pred_tilde, vq_loss_pred = vae_pred_model(x_pred)
                    x_cond_tilde, vq_loss_cond = vae_cond_model(x_cond)
                
                # pred Disc Loss
                d_loss_pred = vae_pred_criterion(vq_loss_pred, x_pred, rearrange(x_pred_tilde, '(b t) c h w -> b c t h w', b=batch_size), optimizer_idx=1, global_step=it)
                # Cond Disc Loss
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

        # Alternation logic and logging
        if it % accum_iter == accum_iter - 1 and it // accum_iter >= disc_start:
            disc_opt = not disc_opt

        if rank == 0 and it % 10 == 0:
            pbar.set_description(f"VAE Recon(Tgt): {vae_cond_loss.item():.4f} | REPA: {vae_loss.item():.4f} | DiT FM: {loss_fm.item():.4f}")

    pbar.close()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, help="Path to config file for all next arguments")
    parser.add_argument('--exp', type=str, default='e2e', help='Type of training to run [vae, fm, mmfm, e2e]')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--n_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--output', type=str, default='./debug', help='Output directory where to store exp results')

    # Data Args
    parser.add_argument('--data', type=str, default='CITYSCAPES_RGB')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_folder', type=str, default='/mnt/data/wangsen/SyncVP/data')

    # VAE Args
    parser.add_argument('--vae_config', type=str, default='configs/test.yaml')

    # DiT Args
    parser.add_argument("--model", type=str, default="DiT-B/2", choices=DiT_models.keys(),
                        help="The model to train.")
    parser.add_argument('--rgb_model', type=str, default='')
    parser.add_argument('--depth_model', type=str, default='')
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bn-momentum", type=float, default=0.1)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to compile the model for faster training")

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
    parser.add_argument("--align_model", type=str, default="VideoMAEv2", 
                        choices=["VideoMAEv2", "VideoMAE", "OminiMAE", "DINOv3", "VJEPA", "VJEPA2"],
                        help="Video foundation model used for REPA alignment.")
    parser.add_argument("--align_ckpt_dir", type=str, default="./ckpts", 
                        help="Base directory for the align model checkpoints.")

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