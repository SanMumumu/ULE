import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.fm.utils import timestep_embedding, pad_triplane_cond


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class TriplaneRoPE(nn.Module):
    def __init__(self, head_dim, frames, input_size, base=10000):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be divisible by 2"
        
        d_t = (head_dim // 3) // 2 * 2
        d_y = (head_dim // 3) // 2 * 2
        d_x = head_dim - d_t - d_y
        
        inv_freq_t = 1.0 / (base ** (torch.arange(0, d_t, 2).float() / d_t))
        inv_freq_y = 1.0 / (base ** (torch.arange(0, d_y, 2).float() / d_y))
        inv_freq_x = 1.0 / (base ** (torch.arange(0, d_x, 2).float() / d_x))
        
        H = W = input_size
        T = frames
        
        xy_t = torch.zeros(H * W)
        xy_y = torch.arange(H).view(H, 1).expand(H, W).flatten().float()
        xy_x = torch.arange(W).view(1, W).expand(H, W).flatten().float()
        
        yt_t = torch.arange(T).view(T, 1).expand(T, H).flatten().float()
        yt_y = torch.arange(H).view(1, H).expand(T, H).flatten().float()
        yt_x = torch.zeros(T * H)
        
        xt_t = torch.arange(T).view(T, 1).expand(T, W).flatten().float()
        xt_y = torch.zeros(T * W)
        xt_x = torch.arange(W).view(1, W).expand(T, W).flatten().float()
        
        t_coords = torch.cat([xy_t, yt_t, xt_t], dim=0)
        y_coords = torch.cat([xy_y, yt_y, xt_y], dim=0)
        x_coords = torch.cat([xy_x, yt_x, xt_x], dim=0)
        
        freqs_half = torch.cat([
            torch.outer(t_coords, inv_freq_t), # [seq_len, d_t/2]
            torch.outer(y_coords, inv_freq_y), # [seq_len, d_y/2]
            torch.outer(x_coords, inv_freq_x)  # [seq_len, d_x/2]
        ], dim=-1) # -> [seq_len, head_dim/2]
        
        freqs = torch.cat([freqs_half, freqs_half], dim=-1) # -> [seq_len, head_dim]
        
        self.register_buffer("cos_cached", freqs.cos().unsqueeze(0).unsqueeze(0)) # [1, 1, seq_len, head_dim]
        self.register_buffer("sin_cached", freqs.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # x shape: [B, num_heads, seq_len, head_dim]
        cos = self.cos_cached[:, :, :x.size(2), :]
        sin = self.sin_cached[:, :, :x.size(2), :]
        return (x * cos) + (rotate_half(x) * sin)

# =========================================================================
# Inline Helper Modules from LightningDiT (RMSNorm, SwiGLU, modulate)
# =========================================================================

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

def modulate(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# =========================================================================
# FMWrapper
# =========================================================================

class FMWrapper(nn.Module):
    def __init__(self, model, conditioning_key=None):
        super().__init__()
        self.fm_model = model
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, cond, t):
        if self.conditioning_key is None:
            out = self.fm_model(x, cond, t)
        else:
            raise NotImplementedError()
        return out

# =========================================================================
# LightningDiT Core Blocks
# =========================================================================

class Attention(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        qkv_bias=False, 
        qk_norm=False, 
        use_rmsnorm=False, 
        fused_attn=True
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        q, k = self.q_norm(q), self.k_norm(k)
        
        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class DiTBlock(nn.Module):
    """
    Upgraded to LightningDiT Block architecture.
    """
    def __init__(
        self, 
        hidden_size, 
        num_heads, 
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False, 
        use_rmsnorm=False,
        wo_shift=False,
        fused_attn=True
    ):
        super().__init__()
        
        # 1. Normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            
        # 2. Attention
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            fused_attn=fused_attn
        )
        
        # 3. MLP Layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, hidden_size),
            )
            
        # 4. AdaLN Modulation
        self.wo_shift = wo_shift
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )

    def forward(self, x, c, feat_rope=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
            
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

# =========================================================================
# Main DiT Model (with Repa-E integration)
# =========================================================================

class DiT(nn.Module):
    def __init__(
        self,
        input_size=32,       
        in_channels=4,       
        hidden_size=384,
        depth=12,
        num_heads=6,
        frames=8,            
        learn_sigma=False,
        max_seq_len=4096,
        teacher_dim=768,     
        aligned_depth=4,
        time_scale_factor=1000,
        bn_momentum=0.1,
        use_qknorm=True, 
        use_swiglu=True, 
        use_rmsnorm=True,
        wo_shift=True,   
        use_rope=True,   
        fused_attn=True, 
        **kwargs             
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size = input_size
        self.image_size = input_size 
        self.frames = frames
        self.aligned_depth = aligned_depth
        self.time_scale_factor = time_scale_factor
        
        self.len_xy = input_size * input_size
        self.len_yt = frames * input_size
        self.len_xt = frames * input_size
        self.seq_len = self.len_xy + self.len_yt + self.len_xt
        
        self.ae_emb_dim = (input_size * input_size) + (frames * input_size) + (frames * input_size)

        self.x_embedder = nn.Linear(in_channels * 2, hidden_size)
        
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size), requires_grad=True)
        self.segment_embed = nn.Parameter(torch.zeros(1, 3, hidden_size), requires_grad=True)
        
        # Placeholder for RoPE if explicitly passed/injected later for Tri-plane structure
        if use_rope:
            head_dim = hidden_size // num_heads
            self.feat_rope = TriplaneRoPE(head_dim, frames, input_size)
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size, 
                num_heads,
                use_qknorm=use_qknorm,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
                wo_shift=wo_shift,
                fused_attn=fused_attn
            ) for _ in range(depth)
        ])

        self.repa_proj = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(in_channels=hidden_size, out_channels=teacher_dim, kernel_size=1)
        )
        
        self.final_layer = FinalLayer(hidden_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        
        # 👑 [CRITICAL] 严格保留 Repa-E 核心 BatchNorm 设计
        self.bn = nn.BatchNorm1d(in_channels, eps=1e-4, momentum=bn_momentum, affine=False, track_running_stats=True)
        self.cond_bn = nn.BatchNorm1d(in_channels, eps=1e-4, momentum=bn_momentum, affine=False, track_running_stats=True)
        self.bn.reset_running_stats()
        self.cond_bn.reset_running_stats()
        
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        for module in self.repa_proj.modules():
            if isinstance(module, nn.Conv1d): 
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Zero-out adaLN modulation layers in blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.segment_embed, std=0.02)

    def get_loss(self, pred, target):
        return torch.nn.functional.mse_loss(target, pred, reduction='none')

    def _normalize_cond(self, cond):
        is_uncond = cond.abs().amax(dim=(1, 2), keepdim=True) < 1e-12
        valid_mask = ~is_uncond.view(-1)

        if valid_mask.any():
            if valid_mask.all():
                cond = self.cond_bn(cond)
            else:
                cond_out = cond.clone()
                cond_out[valid_mask] = self.cond_bn(cond[valid_mask])
                cond = cond_out

        return cond.masked_fill(is_uncond, 0.0)

    def _infer_teacher_grid(self, num_teacher_tokens):
        preferred_t = max(1, self.frames // 2)
        candidate_ts = []
        for value in (preferred_t, self.frames, 1):
            if value not in candidate_ts:
                candidate_ts.append(value)

        for t_vfm in candidate_ts:
            if num_teacher_tokens % t_vfm != 0:
                continue
            spatial_tokens = num_teacher_tokens // t_vfm
            spatial_size = int(round(spatial_tokens ** 0.5))
            if spatial_size * spatial_size == spatial_tokens:
                return t_vfm, spatial_size, spatial_size

        best_triplet = None
        best_score = None
        max_t = min(num_teacher_tokens, max(self.frames, 1))
        for t_vfm in range(1, max_t + 1):
            if num_teacher_tokens % t_vfm != 0:
                continue
            spatial_tokens = num_teacher_tokens // t_vfm
            h_vfm = int(spatial_tokens ** 0.5)
            while h_vfm > 0:
                if spatial_tokens % h_vfm == 0:
                    w_vfm = spatial_tokens // h_vfm
                    score = abs(h_vfm - w_vfm) + abs(t_vfm - preferred_t)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_triplet = (t_vfm, h_vfm, w_vfm)
                    break
                h_vfm -= 1

        if best_triplet is None:
            raise ValueError(f"Unable to infer a sampling grid for {num_teacher_tokens} teacher tokens.")
        return best_triplet

    def compute_triplane_repa_loss(self, student_feats, align_target, align_only, device,
                                   margin_vae=0.5, margin_dit=0.1,
                                   distmat_weight=1.0, cos_weight=1.0):
        """
        Projects triplane features to 3D voxel space using grid sampling 
        and computes the cosine similarity against the teacher's target features.
        """
        b, _, teacher_dim = student_feats.shape
        
        # 1. Reshape triplane 1D sequences back to 2D spatial feature maps
        xy_plane = student_feats[:, :self.len_xy, :].transpose(1, 2).view(b, teacher_dim, self.input_size, self.input_size)
        yt_plane = student_feats[:, self.len_xy:self.len_xy+self.len_yt, :].transpose(1, 2).view(b, teacher_dim, self.frames, self.input_size)
        xt_plane = student_feats[:, self.len_xy+self.len_yt:, :].transpose(1, 2).view(b, teacher_dim, self.frames, self.input_size)
        
        # 2. Define standard 3D voxel grid coordinates corresponding to VFM output (T=4, H=8, W=8)
        _, num_teacher_tokens, _ = align_target.shape
        t_vfm, h_vfm, w_vfm = self._infer_teacher_grid(num_teacher_tokens)
        if t_vfm * h_vfm * w_vfm != num_teacher_tokens:
            raise ValueError(
                f"Sampling grid volume {t_vfm * h_vfm * w_vfm} does not match teacher tokens {num_teacher_tokens}."
            )
        t = torch.linspace(-1, 1, steps=t_vfm, device=device)
        y = torch.linspace(-1, 1, steps=h_vfm, device=device)
        x = torch.linspace(-1, 1, steps=w_vfm, device=device)
        grid_t, grid_y, grid_x = torch.meshgrid(t, y, x, indexing='ij') 
        
        # 3. Construct 2D sampling grids matching grid_sample format (X, Y)
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2).expand(b, -1, -1, -1) 
        grid_yt = torch.stack([grid_y, grid_t], dim=-1).view(1, 1, -1, 2).expand(b, -1, -1, -1) 
        grid_xt = torch.stack([grid_x, grid_t], dim=-1).view(1, 1, -1, 2).expand(b, -1, -1, -1) 
        
        # 4. Bilinear interpolation across all planes
        feat_xy = F.grid_sample(xy_plane, grid_xy, mode='bilinear', align_corners=True).squeeze(2)
        feat_yt = F.grid_sample(yt_plane, grid_yt, mode='bilinear', align_corners=True).squeeze(2)
        feat_xt = F.grid_sample(xt_plane, grid_xt, mode='bilinear', align_corners=True).squeeze(2)
        
        # 5. Fuse sampled points and compute cosine similarity loss
        student_feats_aligned = (feat_xy + feat_yt + feat_xt).transpose(1, 2)
        
        # ==========================================
        # [CRITICAL] Detach teacher features to prevent gradient leakage
        # ==========================================
        align_target = align_target.detach()
        
        teacher_feats_norm = F.normalize(align_target, dim=-1)
        student_feats_norm = F.normalize(student_feats_aligned, dim=-1)
        
        if align_only: 
            cos_sim = (teacher_feats_norm * student_feats_norm).sum(dim=-1)
            loss_cos = F.relu(1.0 - margin_vae - cos_sim).mean()
            
            student_sim = torch.bmm(student_feats_norm, student_feats_norm.transpose(1, 2))
            teacher_sim = torch.bmm(teacher_feats_norm, teacher_feats_norm.transpose(1, 2))
            
            loss_distmat = F.relu((student_sim - teacher_sim).abs() - margin_vae).mean()
            
            proj_loss = (loss_distmat * distmat_weight) + (loss_cos * cos_weight)
            return proj_loss
        else: 
            student_sim = torch.bmm(student_feats_norm, student_feats_norm.transpose(1, 2))
            teacher_sim = torch.bmm(teacher_feats_norm, teacher_feats_norm.transpose(1, 2))
            trd_loss = F.relu((student_sim - teacher_sim).abs() - margin_dit).mean()
            
            return trd_loss
        

    def forward(self, x, align_target, cond=None, time_input=None, noises=None, align_only=False):
        x = self.bn(x)
        b, device, dtype = x.shape[0], x.device, x.dtype
        
        if time_input is None:
            time_input = torch.rand((b,), device=device, dtype=dtype)
        else:
            time_input = time_input.to(device=device, dtype=dtype)
        
        if noises is None:
            noises = torch.randn_like(x)
        else:
            noises = noises.to(device=device, dtype=dtype)
         
        t_expanded = time_input.view(x.shape[0], *([1] * (len(x.shape) - 1)))
        pred_v = t_expanded * x + (1 - t_expanded) * noises
        
        target_v = x - noises

        if cond is not None:
            cond = self._normalize_cond(cond)
            if cond.shape[2] != pred_v.shape[2]:
                cond = pad_triplane_cond(self.input_size, cond, pred_v.shape[2])
        else:
            cond = torch.zeros_like(pred_v)
        
        pred_v = torch.cat([pred_v, cond], dim=1).transpose(1, 2)
        pred_v = self.x_embedder(pred_v)
        
        seq_len = pred_v.shape[1]
        if seq_len > self.pos_embed.shape[1]:
             pred_v = pred_v + self.pos_embed[:, :self.pos_embed.shape[1], :] 
        else:
             pred_v = pred_v + self.pos_embed[:, :seq_len, :]
             
        seg_indices = torch.cat([
            torch.zeros(self.len_xy, device=device).long(),
            torch.ones(self.len_yt, device=device).long(),
            torch.full((self.len_xt,), 2, device=device).long()
        ])
        pred_v = pred_v + self.segment_embed[:, seg_indices, :]
        
        t_emb = timestep_embedding(time_input * self.time_scale_factor, self.x_embedder.out_features)
        t_emb = self.t_embedder(t_emb)
        
        proj_loss = torch.tensor(0.0, device=device)
        
        for i, block in enumerate(self.blocks):
            pred_v = block(pred_v, t_emb, feat_rope=self.feat_rope)
            
            if (i + 1) == self.aligned_depth:
                student_feats_conv = pred_v.transpose(1, 2)
                student_feats_conv = self.repa_proj(student_feats_conv)
                student_feats = student_feats_conv.transpose(1, 2)
                
                if align_target is not None:
                    proj_loss = self.compute_triplane_repa_loss(student_feats, align_target, align_only, device)
                
                if align_only:
                    return {
                        "align_vae_loss": proj_loss,
                        "time_input": time_input,
                        "noises": noises,
                    }
                        
        pred_v = self.final_layer(pred_v, t_emb)
        pred_v = pred_v.transpose(1, 2)
        denoising_loss = self.get_loss(pred_v, target_v)
        
        return {
            "denoising_loss": denoising_loss,
            "align_dit_loss": proj_loss,
            "time_input": time_input,
            "noises": noises,
        }
        
        
    @torch.no_grad()
    def forward_sampling(self, x, cond, time_input):
        device = x.device
        pred_v = x

        if cond is not None:
            cond = self._normalize_cond(cond)
            if cond.shape[2] != pred_v.shape[2]:
                cond = pad_triplane_cond(self.input_size, cond, pred_v.shape[2])
        else:
            cond = torch.zeros_like(pred_v)

        pred_v = torch.cat([pred_v, cond], dim=1)
    
        pred_v = pred_v.transpose(1, 2)
        pred_v = self.x_embedder(pred_v)
        
        seq_len = pred_v.shape[1]
        if seq_len > self.pos_embed.shape[1]:
             pred_v = pred_v + self.pos_embed[:, :self.pos_embed.shape[1], :] 
        else:
             pred_v = pred_v + self.pos_embed[:, :seq_len, :]
             
        seg_indices = torch.cat([
            torch.zeros(self.len_xy, device=device).long(),
            torch.ones(self.len_yt, device=device).long(),
            torch.full((self.len_xt,), 2, device=device).long()
        ])
        pred_v = pred_v + self.segment_embed[:, seg_indices, :]
        
        t_emb = timestep_embedding(time_input, self.x_embedder.out_features)
        t_emb = self.t_embedder(t_emb)
        
        for block in self.blocks:
            pred_v = block(pred_v, t_emb, feat_rope=self.feat_rope)
                        
        pred_v = self.final_layer(pred_v, t_emb)
        pred_v = pred_v.transpose(1, 2)
        return pred_v


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 

    emb = np.concatenate([emb_h, emb_w], axis=1) 
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega) 

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb

#################################################################################
#                                   DiT Configs                                 #
#################################################################################

def DiT_XL(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, aligned_depth=8, **kwargs)

def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, aligned_depth=8, **kwargs)

def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, aligned_depth=4, **kwargs)

def DiT_S(**kwargs):
    return DiT(depth=12, hidden_size=384, num_heads=6, aligned_depth=4, **kwargs)

DiT_models = {
    'DiT-XL': DiT_XL,
    'DiT-L':  DiT_L, 
    'DiT-B':  DiT_B, 
    'DiT-S':  DiT_S, 
}
