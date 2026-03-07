import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.fm.utils import timestep_embedding


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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).flatten(2) 
        x = self.proj(x)
        return x
    
    
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm)
        
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


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

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        self.repa_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, teacher_dim)
        )
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        # Note that we disable affine parameters in the batch norm layer, to avoid affine hacking diffusion loss
        self.bn = nn.BatchNorm1d(in_channels, eps=1e-4, momentum=0.1, affine=False, track_running_stats=True)
        self.bn.reset_running_stats()
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
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_loss(self, pred, target):
        return torch.nn.functional.mse_loss(target, pred)

    def compute_triplane_repa_loss(self, student_feats, align_target, device):
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
        t_vfm, h_vfm, w_vfm = 4, 8, 8
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
        
        teacher_feats_norm = F.normalize(align_target, dim=-1)
        student_feats_norm = F.normalize(student_feats_aligned, dim=-1)
        
        # loss ∈ [0.0, 2.0] lower means better
        proj_loss = 1.0 - (teacher_feats_norm * student_feats_norm).sum(dim=-1).mean()
        
        return proj_loss

    def forward(self, x, align_target, cond=None, time_input=None, noises=None, align_only=False):
        x = self.bn(x)
        b, device, dtype = x.shape[0], x.device, x.dtype
        
        if time_input is None:
            time_input = torch.sigmoid(torch.randn((b,), device=device, dtype=dtype))
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
            if cond.shape[2] != pred_v.shape[2]:
                 cond = F.pad(cond, (0, pred_v.shape[2] - cond.shape[2]), "constant", 0)
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
        
        t_emb = timestep_embedding(time_input * self.time_scale_factor, self.x_embedder.out_features)
        t_emb = self.t_embedder(t_emb)
        
        proj_loss = torch.tensor(0.0, device=device)
        
        for i, block in enumerate(self.blocks):
            pred_v = block(pred_v, t_emb)
            
            if align_only and (i + 1) == self.aligned_depth:
                student_feats = self.repa_proj(pred_v)
                
                if align_target is not None:
                    # Invoke encapsulated alignment logic
                    proj_loss = self.compute_triplane_repa_loss(student_feats, align_target, device)
                
                return {
                    "proj_loss": proj_loss,
                    "time_input": time_input,
                    "noises": noises,
                }
                        
        pred_v = self.final_layer(pred_v, t_emb)
        pred_v = pred_v.transpose(1, 2)
        denoising_loss = self.get_loss(pred_v, target_v)
        
        return {
            "denoising_loss": denoising_loss,
            "proj_loss": proj_loss,
            "time_input": time_input,
            "noises": noises,
        }
        
        
    @torch.no_grad()
    def forward_sampling(self, x, cond, time_input):
        """
        专门用于推理和采样的前向传播（跳过加噪和Loss计算，直接输出速度预测 v）
        x: 当前的 noisy latent [B, C, L]
        cond: 条件特征 [B, C, L]
        time_input: 时间步 [B,]
        """
        device = x.device
        pred_v = x

        if cond is not None:
            if cond.shape[2] != pred_v.shape[2]:
                 cond = F.pad(cond, (0, pred_v.shape[2] - cond.shape[2]), "constant", 0)
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
        
        t_emb = timestep_embedding(time_input * self.time_scale_factor, self.x_embedder.out_features)
        t_emb = self.t_embedder(t_emb)
        
        for block in self.blocks:
            pred_v = block(pred_v, t_emb)
                        
        pred_v = self.final_layer(pred_v, t_emb)
        pred_v = pred_v.transpose(1, 2)
        return pred_v


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
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