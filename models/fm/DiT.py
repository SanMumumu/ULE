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
        input_size=32,       # Latent spatial size (e.g. 32 for 128x128 img)
        in_channels=4,       # Latent dim
        hidden_size=384,
        depth=12,
        num_heads=6,
        frames=8,            # Number of frames
        learn_sigma=False,
        max_seq_len=4096,
        **kwargs             # Handle extra args like patch_size if present
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size = input_size
        self.image_size = input_size # Compatibility alias for eval.py
        self.frames = frames
        
        self.len_xy = input_size * input_size
        self.len_yt = frames * input_size
        self.len_xt = frames * input_size
        self.seq_len = self.len_xy + self.len_yt + self.len_xt
        
        # Calculate total latent sequence length (L)
        # L = spatial (H*W) + temporal_yt (T*H) + temporal_xt (T*W)
        self.ae_emb_dim = (input_size * input_size) + (frames * input_size) + (frames * input_size)

        # 1. Input Projection
        # Input is concatenated with condition: (N, C+C, L)
        self.x_embedder = nn.Linear(in_channels * 2, hidden_size)
        
        # 2. Timestep Embedding
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # 3. Positional Embedding (1D learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size), requires_grad=True)
        # 3.1 Segment Embedding
        self.segment_embed = nn.Parameter(torch.zeros(1, 3, hidden_size), requires_grad=True)

        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # 5. Output Layer
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()


    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.segment_embed, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def get_loss(self, pred, target):
        if self.loss_type == 'l1':
            return (target - pred).abs().mean()
        elif self.loss_type == 'l2':
            return torch.nn.functional.mse_loss(target, pred)
        else:
            raise NotImplementedError(f"unknown loss type '{self.loss_type}'")

    def forward(self, x, align_target, cond=None, time_input=None, noise=None, align_only=False):
        """
        x: (N, C, L)
        cond: (N, C, L)
        time_input: (N,)
        """
        b, device = x.shape[0], x.device
        if time_input is None:
            time_input = torch.sigmoid(torch.randn((b,), device=device)) 
            
        time_input = time_input.to(device=x.device, dtype=x.dtype)
        
        if noise is None:
            noise = torch.randn_like(x)
        else:
            noise = noise.to(device=x.device, dtype=x.dtype)
            
        x_1 = x 
        t_expanded = time_input.view(b, *([1] * (len(x.shape) - 1)))
        mu_t = t_expanded * x_1 + (1 - t_expanded) * noise

        target_v = x_1 - noise

        # Handle conditioning by concatenation
        if cond is not None:
            if cond.shape[2] != mu_t.shape[2]:
                 # Safety pad if needed
                 import torch.nn.functional as F
                 cond = F.pad(cond, (0, mu_t.shape[2] - cond.shape[2]), "constant", 0)
            mu_t = torch.cat([mu_t, cond], dim=1) # (N, 2*C, L)
        
        # (N, C, L) -> (N, L, C)
        mu_t = mu_t.transpose(1, 2)
        
        # Embed
        mu_t = self.x_embedder(mu_t)
        
        # Add Positional Embedding
        seq_len = mu_t.shape[1]
        if seq_len > self.pos_embed.shape[1]:
             # Handle case where seq_len exceeds max_seq_len (unlikely if configured right)
             mu_t = mu_t + self.pos_embed[:, :self.pos_embed.shape[1], :] 
        else:
             mu_t = mu_t + self.pos_embed[:, :seq_len, :]
             
        # Add Segment Embedding
        device = mu_t.device
        seg_indices = torch.cat([
            torch.zeros(self.len_xy, device=device).long(),
            torch.ones(self.len_yt, device=device).long(),
            torch.full((self.len_xt,), 2, device=device).long()
        ])
        mu_t = mu_t + self.segment_embed[:, seg_indices, :]
        
        # Timestep
        t_emb = timestep_embedding(time_input, self.x_embedder.out_features)
        t_emb = self.t_embedder(t_emb)
        
        # Blocks
        for block in self.blocks:
            mu_t = block(mu_t, t_emb)
            
        # Output
        mu_t = self.final_layer(mu_t, t_emb)
        
        # (N, L, C) -> (N, C, L)
        mu_t = mu_t.transpose(1, 2)
        
        denoising_loss = self.get_loss(mu_t, target_v)
        
        return {
            "denoising_loss": denoising_loss,
            "proj_loss": proj_loss,
            "time_input": time_input,
            "noises": noise,
        }
    

#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL(**kwargs):
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)

def DiT_S(**kwargs):
    return DiT(depth=12, hidden_size=384, num_heads=6, **kwargs)

DiT_models = {
    'DiT-XL': DiT_XL,
    'DiT-L':  DiT_L, 
    'DiT-B':  DiT_B, 
    'DiT-S':  DiT_S, 
}