import copy

from models.vae.vae_vit_rope import ViTAutoencoder
from models.fm.DiT import FMWrapper, DiT
from models.fm.uniDiT import FlowMatchingWrapper, UnifiedDiT
from tools.dataloader import get_loaders
from tools.utils import Logger, file_name
import torch
from evals.eval import eval_autoencoder, eval_flow_matching, eval_multimodal_fm


def setup(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fn = "eval_" + file_name(args)
    logger = Logger(fn, path=args.output, resume=args.resume)
    logger.log(args)
    logger.log(f'Log path: {logger.logdir}')

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    log_(f"Loading dataset {args.data} with resolution {args.res}")

    args.frames = args.future_frames
    train_loader, _, test_loader = get_loaders(0, copy.deepcopy(args))

    args.frames += args.cond_frames

    if args.train:
        return log_, logger, train_loader, device
    return log_, logger, test_loader, device


def autoencoder_eval(args):
    log_, logger, test_loader, device = setup(args)
    log_(f"Loading vae model")
    model = ViTAutoencoder(args.embed_dim, args.ddconfig)
    model.load_state_dict(torch.load(args.ae_model))
    model = model.to(device)

    if args.train:
        samples = args.samples if args.samples > 0 else 256  # Train might be too big
    else:
        samples = args.samples if args.samples > 0 else len(test_loader) * args.batch_size

    log_(f"Evaluation on {samples} samples")
    fvd, ssim, lpips, psnr = eval_autoencoder(0, model, test_loader, 0, samples, logger)
    log_(f"FVD: {fvd}, SSIM: {ssim}, LPIPS: {lpips * 1000}, PSNR: {psnr}")


class DummyWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


def flowmatching_eval(args):
    log_, logger, test_loader, device = setup(args)

    log_(f"Loading vae model")
    autoencoder_model = ViTAutoencoder(args.embed_dim, args.ddconfig)
    autoencoder_model.load_state_dict(torch.load(args.ae_model))
    autoencoder_model = DummyWrapper(autoencoder_model).to(device)
    autoencoder_model.eval()

    autoencoder_cond_model = ViTAutoencoder(args.embed_dim, args.ae_cond_ddconfig)
    autoencoder_cond_model.load_state_dict(torch.load(args.ae_cond_model))
    autoencoder_cond_model = DummyWrapper(autoencoder_cond_model).to(device)
    autoencoder_cond_model.eval()

    log_(f"Loading flow matching model")
    backbone = DiT(**args.dit_config, frames=8)
    fm_model = FMWrapper(backbone)
    fm_model.load_state_dict(torch.load(args.fm_model))
    fm_model = DummyWrapper(fm_model).to(device)
    fm_model.eval()

    # Count flow matching model parameters
    log_(f"Flow matching model has {sum(p.numel() for p in fm_model.parameters()) / 1e6:.2f}M parameters")

    if args.train:
        samples = args.samples if args.samples > 0 else 256
    else:
        samples = args.samples if args.samples > 0 else len(test_loader) * args.batch_size

    trajectories = args.traj if args.traj > 0 else 1
    log_(f"Evaluation on {samples} samples with {trajectories} generated samples")

    fvd, ssim, lpips = eval_flow_matching(0, fm_model, autoencoder_model, autoencoder_cond_model, test_loader, 0,
                                      samples, logger, args.frames, args.cond_frames, trajectories, sampling_timesteps=args.NFE)
    log_(f"FVD: {fvd}, SSIM: {ssim}, LPIPS: {lpips * 1000}")


def multimodal_flowmatching_eval(args):
    log_, logger, test_loader, device = setup(args)

    log_(f"Loading vae models")
    autoencoder_model_rgb = ViTAutoencoder(args.embed_dim, args.ddconfig).to(device)
    autoencoder_model_depth = ViTAutoencoder(args.embed_dim, args.ddconfig).to(device)

    autoencoder_model_rgb.load_state_dict(torch.load(args.ae_model))
    autoencoder_model_depth.load_state_dict(torch.load(args.ae_model_depth))
    autoencoder_model_rgb = DummyWrapper(autoencoder_model_rgb).to(device)
    autoencoder_model_depth = DummyWrapper(autoencoder_model_depth).to(device)
    autoencoder_model_rgb.eval()
    autoencoder_model_depth.eval()

    if args.ae_cond_model != '':
        autoencoder_cond_model_rgb = ViTAutoencoder(args.embed_dim, args.ae_cond_ddconfig).to(device)
        autoencoder_cond_model_depth = ViTAutoencoder(args.embed_dim, args.ae_cond_ddconfig).to(device)
        autoencoder_cond_model_rgb.load_state_dict(torch.load(args.ae_cond_model))
        autoencoder_cond_model_depth.load_state_dict(torch.load(args.ae_cond_model_depth))
        autoencoder_cond_model_rgb = DummyWrapper(autoencoder_cond_model_rgb).to(device)
        autoencoder_cond_model_depth = DummyWrapper(autoencoder_cond_model_depth).to(device)
        autoencoder_cond_model_rgb.eval()
        autoencoder_cond_model_depth.eval()
    else:
        autoencoder_cond_model_rgb = autoencoder_model_rgb
        autoencoder_cond_model_depth = autoencoder_model_depth

    log_(f"Loading multi modal flow mating model")
    unified_model = UnifiedDiT(
        input_size=args.unetconfig.image_size // 4, # 假设 VAE 下采样 4 倍
        in_channels=args.unetconfig.in_channels,    # 通常为 4
        hidden_size=getattr(args, 'hidden_size', 768),
        depth=getattr(args, 'depth', 12),
        num_heads=getattr(args, 'num_heads', 12),
        frames=args.frames,
        num_modalities=2  # RGB + Depth
    ).to(device)
    fm_model = FlowMatchingWrapper(unified_model).to(device)
    fm_model.load_state_dict(torch.load(args.fm_model))
    fm_model = DummyWrapper(fm_model).to(device)
    fm_model.eval()

    # Count flow matching model parameters
    log_(f"Flow matching model has {sum(p.numel() for p in fm_model.parameters())} parameters")

    if args.train:
        samples = args.samples if args.samples > 0 else 256
    else:
        samples = args.samples if args.samples > 0 else len(test_loader) * args.batch_size

    trajectories = args.traj if args.traj > 0 else 1

    log_(f"Evaluation on {samples} samples with {trajectories} generated samples")

    if args.no_depth_cond:
        log_(f"!!! Depth conditioning disabled !!!")

    fvd_rgb, fvd_depth, ssim_rgb, ssim_depth, lpips_rgb, lpips_depth, l2 = eval_multimodal_fm(
        0,
        ema_model=fm_model,
        ae_rgb=autoencoder_model_rgb,
        ae_depth=autoencoder_model_depth,
        ae_cond_rgb=autoencoder_cond_model_rgb,
        ae_cond_depth=autoencoder_cond_model_depth,
        loader=test_loader,
        it=0,
        samples=samples,
        logger=logger,
        frames=args.frames,
        cond_frames=args.cond_frames,
        trajectories=trajectories,
        w=args.fmconfig.w,
        no_depth_cond=args.no_depth_cond,
        same_noise=args.same_noise
    )

    log_(f"RGB - FVD Avg: {fvd_rgb}, SSIM: {ssim_rgb}, LPIPS: {lpips_rgb * 1000}")
    log_(f"Depth - FVD Avg: {fvd_depth}, SSIM: {ssim_depth}, LPIPS: {lpips_depth * 1000}, L2: {l2 * 100}")
