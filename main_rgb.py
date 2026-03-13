import os
import copy
import yaml
import argparse
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR

from tools.dataloader import get_loaders
from tools.utils import AverageMeter, setup_distibuted_training, setup_logger, set_random_seed

from tools.train_utils import (
    init_multiprocessing, update_ema, run_evaluation, save_image_grid, 
    log_videos_e2e, set_requires_grad, get_teacher_model, get_align_targets, 
    prepare_input, config_setup, FMSamplingWrapper
)

from models.vae.vae_vit_rope import ViTAutoencoder
from models.fm.DiT import DiT, DiT_models
from losses.perceptual import LPIPSWithDiscriminator
from losses.fm import FlowMatching

# ============================================================================
# Core training loop 
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
    losses['cond_vae_loss'] = AverageMeter()
    losses['pred_vae_loss'] = AverageMeter()
    losses['repa_vae_loss'] = AverageMeter()
    losses['dit_denoise_loss'] = AverageMeter()
    losses['repa_dit_loss'] = AverageMeter()
    losses['disc_loss'] = AverageMeter()
    losses['vae_cond_recon'] = AverageMeter()
    losses['vae_pred_recon'] = AverageMeter()

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
                
                with torch.no_grad():
                    cond_recon_loss = torch.nn.functional.mse_loss(x_cond, rearrange(x_cond_hat, '(b t) c h w -> b c t h w', b=batch_size))
                    pred_recon_loss = torch.nn.functional.mse_loss(x_pred, rearrange(x_traget_hat, '(b t) c h w -> b c t h w', b=batch_size))

                _, cond = prepare_input(args, x_vae, vae_cond_model, vae_pred_model)
                vae_loss = (vae_cond_loss + vae_pred_loss) / accum_iter
                
                vae_align_outputs = model.module(
                    x=z_pred, cond=cond, align_target=align_target,
                    time_input=None, noises=None, align_only=True
                )

                vae_repa_val = vae_align_outputs["align_vae_loss"].mean()
                vae_loss = vae_loss + (args.vae_align_proj_coeff * vae_repa_val / accum_iter)
                
                time_input = vae_align_outputs["time_input"]
                noises = vae_align_outputs["noises"]

            if args.amp: scaler_g.scale(vae_loss).backward()
            else: vae_loss.backward()

            losses['cond_vae_loss'].update(vae_cond_loss.item(), 1)
            losses['pred_vae_loss'].update(vae_pred_loss.item(), 1)
            losses['repa_vae_loss'].update(vae_repa_val.item(), 1)
            losses['vae_cond_recon'].update(cond_recon_loss.item(), 1)
            losses['vae_pred_recon'].update(pred_recon_loss.item(), 1)

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
                dit_repa_val = sit_outputs["align_dit_loss"].mean()
                
                dit_loss = (dit_denoise_val + args.dit_align_proj_coeff * dit_repa_val) / accum_iter
                
            if args.amp: scaler_fm.scale(dit_loss).backward()
            else: dit_loss.backward()

            losses['dit_denoise_loss'].update(dit_denoise_val.item(), 1)
            losses['repa_dit_loss'].update(dit_repa_val.item(), 1)

            if it % accum_iter == accum_iter - 1:
                if args.amp:
                    scale_before_g = scaler_g.get_scale()
                    scale_before_fm = scaler_fm.get_scale()
                    
                    scaler_g.step(opt_vae_g)
                    scaler_g.update()
                    scaler_fm.step(opt_dit)
                    scaler_fm.update()
                    
                    if scaler_g.get_scale() >= scale_before_g: scheduler_vae.step()
                    if scaler_fm.get_scale() >= scale_before_fm: scheduler_dit.step()
                else:
                    opt_vae_g.step()
                    opt_dit.step()
                    scheduler_vae.step()
                    scheduler_dit.step()
                    
                opt_vae_g.zero_grad(set_to_none=True)
                opt_dit.zero_grad(set_to_none=True)
                
                current_decay = min(0.9999, (it + 1) / (it + 10))
                unwrapped_model = model.module if hasattr(model, 'module') else model
                update_ema(ema, unwrapped_model, decay=current_decay)

        else:
            # -------------------------------------------------------------
            # Stage 3: Discriminator Training
            # -------------------------------------------------------------
            if it % accum_iter == 0: 
                vae_pred_criterion.zero_grad(set_to_none=True)
                vae_cond_criterion.zero_grad(set_to_none=True)
                
            with torch.autocast(device_type='cuda', enabled=args.amp):
                with torch.no_grad(): 
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

            losses['disc_loss'].update(d_loss_total.item() * accum_iter, 1)
            
        if it % accum_iter == accum_iter - 1 and it // accum_iter >= disc_start:
            disc_opt = not disc_opt

        # =========================================================================
        # TensorBoard & Tqdm Updates
        # =========================================================================
        if rank == 0 and it % 10 == 0:
            if not disc_opt:
                pbar.set_description(f"VAE: {losses['pred_vae_loss'].average:.3f} | Denoise: {losses['dit_denoise_loss'].average:.3f} | REPA:  {losses['repa_vae_loss'].average:.3f}")
            else:
                pbar.set_description(f"Disc Loss: {losses['disc_loss'].average:.3f}")
                
        if it % args.log_freq == 0:
            if rank == 0 and logger is not None:
                logger.scalar_summary('loss/cond_vae_loss', losses['cond_vae_loss'].average, it)
                logger.scalar_summary('loss/pred_vae_loss', losses['pred_vae_loss'].average, it)
                logger.scalar_summary('loss/repa_vae_loss', losses['repa_vae_loss'].average, it)
                logger.scalar_summary('loss/dit_denoise_loss', losses['dit_denoise_loss'].average, it)
                logger.scalar_summary('loss/repa_dit_loss', losses['repa_dit_loss'].average, it)
                logger.scalar_summary('loss/disc_loss', losses['disc_loss'].average, it)
                
                logger.scalar_summary('lr/lr_vae', scheduler_vae.get_lr()[0], it)
                logger.scalar_summary('lr/lr_dit', scheduler_dit.get_lr()[0], it)
                
                logger.scalar_summary('recon/vae_cond_recon', losses['vae_cond_recon'].average, it)
                logger.scalar_summary('recon/vae_pred_recon', losses['vae_pred_recon'].average, it)

            for k in losses.keys():
                losses[k] = AverageMeter()

        # =========================================================================
        # Periodic Video Generation and Visualization
        # =========================================================================
        if rank == 0 and it % args.eval_freq == 0 and it > 0:
            model.eval() 
            with torch.no_grad():
                try:
                    b_vis = 1
                    c_init_vis = x_cond[:b_vis].clone()
                    gt_pred_vis = x_pred[:b_vis].clone().cpu()
                    
                    c_feat_vis = vae_cond_model.module.extract(c_init_vis).detach()
                    
                    unwrapped_model = model.module if hasattr(model, 'module') else model
                    true_seq_len = unwrapped_model.ae_emb_dim
                    
                    fm_sampler = FlowMatching(
                        FMSamplingWrapper(unwrapped_model),
                        channels=args.in_channels, 
                        image_size=true_seq_len, 
                        sampling_timesteps=50
                    ).to(device)
                    
                    z_sampled = fm_sampler.sample(batch_size=b_vis, cond=c_feat_vis)
                    running_mean_vis = unwrapped_model.bn.running_mean.view(1, -1, 1).to(device)
                    running_var_vis = unwrapped_model.bn.running_var.view(1, -1, 1).to(device)
                    z_sampled = z_sampled * torch.sqrt(running_var_vis + 1e-5) + running_mean_vis
                    
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
                    flat_combined = ((flat_combined + 1.0) / 2.0).clamp(0.0, 1.0)
                    
                    save_path = os.path.join(args.output, f'vis_e2e_{it:07d}.png')
                    torchvision.utils.save_image(flat_combined, save_path, nrow=min_T, normalize=False)
                except Exception as e:
                    log_(f"❌ Visualization error: {str(e)}")
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
        if rank == 0 and it % args.eval_freq  == 0 and it > 0:
            ema.eval(); vae_cond_model.eval(); vae_pred_model.eval()
            
            run_evaluation(
                val_loader=val_loader, ema=ema, vae_cond_model=vae_cond_model,
                vae_pred_model=vae_pred_model, args=args, device=device,
                it=it, logger=logger, log_=log_
            )
            
            ema.train(); vae_cond_model.train(); vae_pred_model.train()

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
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--traj', type=int, default=1)
    parser.add_argument('--NFE', type=int, default=25)
    parser.add_argument('--no_depth_cond', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--future_frames', type=int, default=28)
    
    # Vision Foundation Model
    parser.add_argument("--align_model", type=str, default="VideoMAEv2", choices=["VideoMAEv2", "VideoMAE", "OminiMAE", "DINOv3", "VJEPA", "VJEPA2"])
    parser.add_argument("--align_ckpt_dir", type=str, default="./ckpts")
    
    parser.add_argument('--vae_align_proj_coeff', type=float, default=1.0)
    parser.add_argument('--dit_align_proj_coeff', type=float, default=1.0)

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