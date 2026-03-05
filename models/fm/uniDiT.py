import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fm.utils import timestep_embedding
import numpy as np
from models.fm.DiT import DiTBlock, FinalLayer 

class FlowMatchingWrapper(nn.Module):
    def __init__(self, model, conditioning_key=None):
        super().__init__()
        self.fm_model = model
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, cond, t_rgb, t_depth):
        if self.conditioning_key is None:
            out = self.fm_model(x, cond, t_rgb, t_depth)
        else:
            raise NotImplementedError()

        return out

class UnifiedDiT(nn.Module):
    def __init__(
        self,
        input_size=32,       
        in_channels=4,       
        hidden_size=768,     
        depth=12,
        num_heads=12,
        frames=8,            
        num_modalities=2,    
        max_seq_len=4096,    
        **kwargs             
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels 
        self.frames = frames
        self.input_size = input_size
        
        self.len_xy = input_size * input_size
        self.len_yt = frames * input_size
        self.len_xt = frames * input_size
        self.single_modality_len = self.len_xy + self.len_yt + self.len_xt 
        
        self.ae_emb_dim = (input_size * input_size) + (frames * input_size) + (frames * input_size)

        self.x_embedder = nn.Linear(in_channels * 2, hidden_size)
        
        self.t_embedder = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.single_modality_len, hidden_size), requires_grad=True)
        self.triplane_embed = nn.Parameter(torch.zeros(1, 3, hidden_size), requires_grad=True)
        self.modality_embed = nn.Embedding(num_modalities, hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.triplane_embed, std=0.02)
        torch.nn.init.normal_(self.modality_embed.weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, cond=None, t_rgb=None, t_depth=None):
        # 1. Concat Condition
        if cond is not None:
             x = torch.cat([x, cond], dim=1)
        
        x = x.transpose(1, 2)
        x = self.x_embedder(x)
        
        # 2. Positional Embedding
        num_mods = x.shape[1] // self.single_modality_len
        repeated_pos_embed = self.pos_embed.repeat(1, num_mods, 1)
        x = x + repeated_pos_embed[:, :x.shape[1], :]

        # 3. Triplane Embedding
        device = x.device
        base_indices = torch.cat([
            torch.zeros(self.len_xy, device=device).long(),
            torch.ones(self.len_yt, device=device).long(),
            torch.full((self.len_xt,), 2, device=device).long()
        ])
        triplane_indices = base_indices.repeat(num_mods)
        x = x + self.triplane_embed[:, triplane_indices, :]

        # 4. Modality Embedding
        modality_indices = torch.cat([
            torch.zeros(self.single_modality_len, device=device).long(),
            torch.ones(self.single_modality_len, device=device).long()
        ])
        modality_indices = modality_indices.unsqueeze(0).repeat(x.shape[0], 1)
        x = x + self.modality_embed(modality_indices)

        # 5. Dual Timestep Embedding (处理 t_rgb 和 t_depth)
        # 兼容性处理：防止 None
        if t_rgb is None: t_rgb = torch.zeros((x.shape[0],), device=device)
        if t_depth is None: t_depth = t_rgb

        t_rgb_emb = timestep_embedding(t_rgb, 256).to(x.dtype)
        t_rgb_emb = self.t_embedder(t_rgb_emb)
        
        t_depth_emb = timestep_embedding(t_depth, 256).to(x.dtype)
        t_depth_emb = self.t_embedder(t_depth_emb)
        
        half_len = self.single_modality_len
        t_emb = torch.cat([
            t_rgb_emb.unsqueeze(1).repeat(1, half_len, 1),
            t_depth_emb.unsqueeze(1).repeat(1, half_len, 1)
        ], dim=1)
        
        # Add Time Embedding to Tokens
        x = x + t_emb 
        
        # Dummy global time for AdaLN (Using RGB time as placeholder)
        dummy_t = t_rgb_emb 
        
        for block in self.blocks:
            x = block(x, dummy_t)
            
        x = self.final_layer(x, dummy_t)
        x = x.transpose(1, 2)
        return x