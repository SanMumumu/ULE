import os
import copy
import yaml
import argparse
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from einops import rearrange

from tools.dataloader import get_loaders
from tools.utils import Logger, AverageMeter, set_random_seed
from tools.train_utils import config_setup, FMSamplingWrapper

from models.vae.vae_vit_rope import ViTAutoencoder
from models.fm.DiT import DiT_models
from losses.fm import FlowMatching

from evals.fvd.fvd import get_fvd_logits, frechet_distance
from evals.fvd.download import load_i3d_pretrained
from evals.ssim.ssim import calculate_ssim, SSIM
import lpips

class DummyWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

def save_image_comparison(gts, preds, fname, start_sample, end_sample, cond_frames, is_bgr=True):
    if len(gts) == 0 or len(preds) == 0:
        return
        
    start_sample = max(0, start_sample)
    end_sample = min(len(gts), end_sample)
    
    if start_sample >= end_sample:
        print("⚠️ 切片范围无效，请检查 --save_start_sample 和 --save_end_sample")
        return

    gts_slice = gts[start_sample:end_sample]
    preds_slice = preds[start_sample:end_sample]
    
    gts_slice = gts_slice[:, :, cond_frames:, :, :]
    preds_slice = preds_slice[:, :, cond_frames:, :, :]
    
    if is_bgr:
        gts_slice = gts_slice[:, [2, 1, 0], :, :, :]
        preds_slice = preds_slice[:, [2, 1, 0], :, :, :]

    gts_slice = gts_slice.to(torch.float32) / 255.0
    preds_slice = preds_slice.to(torch.float32) / 255.0
    
    B, C, T, H, W = gts_slice.shape 
    
    combined_frames = []
    for b in range(B):
        gt_seq = gts_slice[b].permute(1, 0, 2, 3)   
        pred_seq = preds_slice[b].permute(1, 0, 2, 3)
        combined_frames.append(gt_seq)
        combined_frames.append(pred_seq)
        
    combined_tensor = torch.cat(combined_frames, dim=0) 
    
    grid = torchvision.utils.make_grid(combined_tensor, nrow=T, normalize=False, padding=2)
    torchvision.utils.save_image(grid, fname)
    print(f"📸 成功保存对比长图 (共 {B} 个样本, 每行严格 {T} 帧): {fname}")


def lpips_video(pred_videos, real_videos, lpips_model, is_bgr=True):
    lpips_loss = []
    device = next(lpips_model.parameters()).device 
    
    pred_videos = rearrange(pred_videos, 'b t h w c -> b t c h w')
    
    if is_bgr:
        pred_videos = pred_videos[:, :, [2, 1, 0], :, :]
        real_videos = real_videos[:, :, [2, 1, 0], :, :]
    
    pred_videos = (pred_videos.to(device).to(torch.float32) / 127.5) - 1.0
    real_videos = (real_videos.to(device).to(torch.float32) / 127.5) - 1.0
    
    for pred, real in zip(pred_videos, real_videos):
        lpips_loss.append(lpips_model(pred, real).mean().item())
        
    return lpips_loss


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to test.yaml config')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to ckpt_xxx.pt')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--n_gpus', type=int, default=1, help='Number of GPUs to use')
    
    parser.add_argument('--future_frames', type=int, default=28, help='How many future frames to predict')
    # ========== 关键修改：默认设为 0，代表遍历全集 ==========
    parser.add_argument('--samples', type=int, default=0, help='Evaluation sample size (0 means full test set)')
    parser.add_argument('--traj', type=int, default=1, help='Number of generation trajectories')
    parser.add_argument('--NFE', type=int, default=50, help='Flow matching steps')
    parser.add_argument('--output', type=str, default='./eval_benchmark_results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for dataloader')

    parser.add_argument('--vae_config', type=str, default='configs/model.yaml')
    parser.add_argument('--data', type=str, default='CITYSCAPES_RGB')
    parser.add_argument('--data_folder', type=str, default='/mnt/nodestor/ws/UniWM/DATA/data')
    parser.add_argument('--model', type=str, default='DiT-S')
    parser.add_argument('--amp', action='store_true', default=True) 
    parser.add_argument('--fused_attn', action='store_true', default=True)
    parser.add_argument('--qk_norm', action='store_true', default=False)
    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--embed_dim', type=int, default=4)
    
    parser.add_argument('--save_start_sample', type=int, default=0)
    parser.add_argument('--save_end_sample', type=int, default=4)
    parser.add_argument('--is_bgr', action='store_true', default=True, help='If True, corrects Cityscapes BGR to RGB')
    
    cmd_args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class Args: pass
    args = Args()
    args.__dict__.update(cmd_args.__dict__)
    
    with open(cmd_args.config, "r") as f:
        config = yaml.safe_load(f)
        
    if config is not None:
        args.__dict__.update(config)
        
    args.batch_size = cmd_args.batch_size
    args.future_frames = cmd_args.future_frames
    args.model = cmd_args.model 
    args.train = False

    args = config_setup(args)

    os.makedirs(args.output, exist_ok=True)
    logger = Logger("eval_fair", path=args.output, resume=False)
    log_ = logger.log

    args.frames = args.cond_frames + args.future_frames
    log_(f"🚀 Starting SyncVP-Aligned Benchmark Evaluation")
    
    _, _, test_loader = get_loaders(0, copy.deepcopy(args))

    # ========== 核心对齐 SyncVP: 动态计算实际测试样本数量 ==========
    # 如果命令行没有指定 --samples（即默认 0），则使用整个 test_loader 的大小
    args.samples = args.samples if args.samples > 0 else len(test_loader) * args.batch_size
    
    log_(f"Dataset: {args.data} | Cond: {args.cond_frames} | Future: {args.future_frames}")
    log_(f"Target Samples: {args.samples} (Batch Size: {args.batch_size})")

    log_(f"📦 Loading Full Checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')

    vae_pred_model = ViTAutoencoder(args.embed_dim, args.vaeconfig)
    vae_pred_model.load_state_dict(ckpt['vae_pred_model'])
    vae_pred_model = DummyWrapper(vae_pred_model).to(device).eval()

    vae_cond_model = ViTAutoencoder(args.embed_dim, args.cond_vaeconfig)
    vae_cond_model.load_state_dict(ckpt['vae_cond_model'])
    vae_cond_model = DummyWrapper(vae_cond_model).to(device).eval()

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    ema_model = DiT_models[args.model](
        input_size=args.input_size, in_channels=args.in_channels,
        z_dims=[384], encoder_depth=args.encoder_depth, bn_momentum=args.bn_momentum,
        **block_kwargs
    )
    ema_model.load_state_dict(ckpt['ema_model'])
    ema_model = DummyWrapper(ema_model).to(device).eval()

    true_seq_len = ema_model.module.ae_emb_dim
    fm_sampler = FlowMatching(
        FMSamplingWrapper(ema_model.module),
        channels=args.in_channels, image_size=true_seq_len, sampling_timesteps=args.NFE
    ).to(device)

    running_mean = ema_model.module.bn.running_mean.view(1, -1, 1).to(device)
    running_var = ema_model.module.bn.running_var.view(1, -1, 1).to(device)

    ssim_metric = SSIM()
    lpips_model = lpips.LPIPS(net='alex').to(device)
    i3d = load_i3d_pretrained(device)

    losses = {'ssim': AverageMeter(), 'lpips': AverageMeter()}
    gt_embeddings, pred_embeddings = [], []
    gts, predictions = [], []

    # 精准计算进度条的步数
    eval_batches = min(len(test_loader), max(1, args.samples // args.batch_size))

    for n, (x, _) in enumerate(tqdm(test_loader, desc="Evaluating", total=eval_batches)):
        if n >= eval_batches: break
        k = x.size(0)

        c_init = x[:, :args.cond_frames]
        c_init_feat = vae_cond_model.module.extract(
            rearrange(c_init / 127.5 - 1, 'b t c h w -> b c t h w').to(device).to(torch.float32).detach()
        )

        _ssim, _lpips = [], []

        for traj_idx in range(args.traj):
            
            with torch.autocast(device_type='cuda', enabled=args.amp):
                z = fm_sampler.sample(batch_size=k, cond=c_init_feat)
            
            z = z.to(torch.float32)
            z = z * torch.sqrt(running_var.to(torch.float32) + 1e-4) + running_mean.to(torch.float32)
            
            if hasattr(vae_pred_model.module, 'decode_from_sample'):
                pred = vae_pred_model.module.decode_from_sample(z).clamp(-1, 1).cpu()
            else:
                out = vae_pred_model.module.decode(z)
                pred = (out.sample if hasattr(out, 'sample') else out).clamp(-1, 1).cpu()
            
            pred = (1 + rearrange(pred, '(b t) c h w -> b t h w c', b=k)) * 127.5
            pred = pred.type(torch.uint8)

            while pred.size(1) < args.future_frames:
                c = pred[:, -args.cond_frames:]
                c_feat = vae_cond_model.module.extract(
                    rearrange(c / 127.5 - 1, 'b t h w c-> b c t h w').to(device).to(torch.float32).detach()
                )
                
                with torch.autocast(device_type='cuda', enabled=args.amp):
                    z_new = fm_sampler.sample(batch_size=k, cond=c_feat)
                
                z_new = z_new.to(torch.float32)
                z_new = z_new * torch.sqrt(running_var.to(torch.float32) + 1e-4) + running_mean.to(torch.float32)
                
                if hasattr(vae_pred_model.module, 'decode_from_sample'):
                    new_pred = vae_pred_model.module.decode_from_sample(z_new).clamp(-1, 1).cpu()
                else:
                    new_out = vae_pred_model.module.decode(z_new)
                    new_pred = (new_out.sample if hasattr(new_out, 'sample') else new_out).clamp(-1, 1).cpu()
                
                new_pred = (1 + rearrange(new_pred, '(b t) c h w -> b t h w c', b=k)) * 127.5
                new_pred = new_pred.type(torch.uint8)
                
                pred = torch.cat([pred, new_pred], dim=1)
            
            if pred.size(1) > args.future_frames:
                pred = pred[:, :args.future_frames]

            real = rearrange(x, 'b t c h w -> b t h w c').cpu().type(torch.uint8)
            cond = x[:, :args.cond_frames].cpu()
            real_future = x[:, args.cond_frames:].cpu()

            pred_ssim_input = rearrange(pred, 'b t h w c -> b t c h w').cpu()
            
            _ssim.append(calculate_ssim(pred_ssim_input, real_future, ssim_metric))
            _lpips.append(lpips_video(pred, real_future, lpips_model, is_bgr=cmd_args.is_bgr))

            pred_fvd_vid = torch.cat([rearrange(cond.type(torch.uint8), 'b t c h w -> b t h w c'), pred], dim=1)
            
            if traj_idx == 0:
                gt_embeddings.append(get_fvd_logits(real.numpy(), i3d=i3d, device=device))
                pred_embeddings.append(get_fvd_logits(pred_fvd_vid.cpu().numpy(), i3d=i3d, device=device))

                # 无论遍历多少验证集，收集画图的样本仅到 save_end_sample 为止，杜绝内存爆满
                current_collected = sum(p.size(0) for p in predictions)
                if current_collected < cmd_args.save_end_sample:
                    gts.append(rearrange(x.type(torch.uint8), 'b t c h w -> b c t h w').cpu())
                    predictions.append(rearrange(pred_fvd_vid, 'b t h w c -> b c t h w').cpu())

        ssim_best = np.max(_ssim, axis=0) if len(_ssim) > 1 else np.array(_ssim[0]) if len(_ssim) > 0 else np.array([-1])
        lpips_best = np.min(_lpips, axis=0) if len(_lpips) > 1 else np.array(_lpips[0])
        losses['ssim'].update(ssim_best.mean())
        losses['lpips'].update(lpips_best.mean())

    if len(gts) > 0:
        gts_tensor = torch.cat(gts, dim=0)
        preds_tensor = torch.cat(predictions, dim=0)
        
        save_path = os.path.join(logger.logdir, f'vis_samples_{cmd_args.save_start_sample}_to_{cmd_args.save_end_sample}.png')
        log_(f"Saving Grid (Row=Sample, Col=Future Frames) to {save_path}...")
        save_image_comparison(
            gts=gts_tensor, 
            preds=preds_tensor, 
            fname=save_path, 
            start_sample=cmd_args.save_start_sample, 
            end_sample=cmd_args.save_end_sample,
            cond_frames=args.cond_frames, 
            is_bgr=cmd_args.is_bgr
        )

    gt_embeddings = torch.cat(gt_embeddings)
    pred_embeddings = torch.cat(pred_embeddings)
    fvd = frechet_distance(pred_embeddings.clone().detach(), gt_embeddings.clone().detach())

    log_(f"==================================================")
    log_(f"✅ FINAL SYNCVP BENCHMARK METRICS (Samples: {len(gt_embeddings)}, Future Frames: {args.future_frames})")
    log_(f"FVD:   {fvd.item():.4f}")
    log_(f"SSIM:  {losses['ssim'].average:.4f}")
    log_(f"LPIPS: {losses['lpips'].average:.4f}")
    log_(f"==================================================")

if __name__ == "__main__":
    main()