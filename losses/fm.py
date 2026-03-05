import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from tqdm import tqdm
from functools import partial
from inspect import isfunction

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class FlowMatching(nn.Module):
    def __init__(self,
                 model,
                 sampling_timesteps=10, # FM requires fewer steps, e.g., 10-50
                 loss_type="l2",
                 image_size=32,
                 channels=4,
                 sigma_min=1e-5,        # Avoid numerical instability at t=0 or t=1
                 ):
        super().__init__()
        self.model = model
        self.channels = channels
        self.loss_type = loss_type
        
        # Handle latent dimension size retrieval
        try:
            self.image_size = model.module.fm_model.ae_emb_dim
        except AttributeError:
            # Handle property access path for multi-modal models
            try:
                self.image_size = model.module.fm_model.diff_rgb.fm_model.ae_emb_dim
            except:
                self.image_size = image_size # Fallback

        self.sigma_min = sigma_min
        self.sampling_timesteps = sampling_timesteps
        
        # Scale factor to map t in [0,1] to the model's preferred [0, 1000] range
        self.time_scale_factor = 1000.0 

    def get_loss(self, pred, target):
        if self.loss_type == 'l1':
            return (target - pred).abs().mean()
        elif self.loss_type == 'l2':
            return torch.nn.functional.mse_loss(target, pred)
        else:
            raise NotImplementedError(f"unknown loss type '{self.loss_type}'")

    def forward(self, x, cond=None, *args, **kwargs):
        """
        x: ground truth data (x_1)
        """
        b, device = x.shape[0], x.device
        
        # 1. Logit-Normal sampling for time t ~ U[0, 1]
        t = torch.sigmoid(torch.randn((b,), device=device)) # torch.rand((b,), device=device)
        
        # 2. Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(x)
        x_1 = x # Target data
        
        # 3. Linear interpolation (Optimal Transport Probability Path)
        # x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1 
        # Simplified: x_t = (1-t)x_0 + t*x_1 (assuming sigma_min -> 0)
        # We use the version with sigma_min for numerical stability
        t_expanded = t.view(b, *([1] * (len(x.shape) - 1)))
        mu_t = t_expanded * x_1 + (1 - t_expanded) * x_0
        
        # 4. Calculate target velocity vector (Flow Target)
        # For OT-CFM (Optimal Transport Conditional Flow Matching), the target is: x_1 - x_0
        target_v = x_1 - x_0

        # 5. Model predicts velocity v_t
        # Note: UNet Time Embeddings are usually designed for 0-1000 integers or large floats.
        # Passing 0-1 directly can cause Positional Embedding collapse, so we scale by 1000.
        model_out = self.model(mu_t, cond, t * self.time_scale_factor)
        
        # 6. Compute Loss
        loss = self.get_loss(model_out, target_v)
        
        loss_dict = {'train/loss': loss.item()}
        return loss, loss_dict

    @torch.no_grad()
    def sample(self, batch_size=16, cond=None, return_intermediates=False):
        # Euler ODE Solver
        device = next(self.model.parameters()).device
        
        # Initial noise x_0
        shape = (batch_size, self.channels, self.image_size)
        x = torch.randn(shape, device=device)
        
        dt = 1.0 / self.sampling_timesteps
        intermediates = [x]
        
        for i in range(self.sampling_timesteps):
            t_value = i / self.sampling_timesteps
            t = torch.full((batch_size,), t_value, device=device)
            
            # Predict velocity field v
            v_pred = self.model(x, cond, t * self.time_scale_factor)
            
            # Euler step: x_{t+dt} = x_t + v * dt
            x = x + v_pred * dt
            
            if return_intermediates:
                intermediates.append(x)
                
        if return_intermediates:
            return x, intermediates
        return x

class MMFlowMatching(FlowMatching):
    def __init__(self, model, *args, **kwargs):
        # 1. Extract subclass-specific arguments first
        self.same_noise = kwargs.pop('same_noise', False)
        # 2. Initialize the parent class
        super().__init__(model, *args, **kwargs)
        self.image_size = model.module.fm_model.ae_emb_dim

    def forward(self, x_rgb, x_depth, cond_rgb=None, cond_depth=None, t_rgb=None, t_depth=None, same_noise=None, *args, **kwargs):
        """
        x_rgb: [B, C, L] 
        x_depth: [B, C, L]
        cond_rgb: [B, C, L]
        cond_depth: [B, C, L]
        t_rgb: [B] (Dual timesteps sampled externally)
        t_depth: [B] (Dual timesteps sampled externally)
        same_noise: bool (External hybrid strategy; uses self.same_noise if None)
        """
        b, c, half_len = x_rgb.shape
        device = x_rgb.device
        
        # 0. Determine noise strategy (Hybrid Strategy Support)
        use_same_noise = same_noise if same_noise is not None else self.same_noise

        # 1. Concatenate data and conditions (Concatenate to Unified Sequence)
        x_1_joint = torch.cat([x_rgb, x_depth], dim=2)
        cond_joint = torch.cat([cond_rgb, cond_depth], dim=2) if (cond_rgb is not None and cond_depth is not None) else None

        # 2. Sample noise x_0 (Source Distribution)
        noise_rgb = torch.randn((b, c, half_len), device=device)
        
        if use_same_noise:
            # Sync Branch: Force shared noise to leverage geometric priors
            noise_depth = noise_rgb 
        else:
            # Decoupled Branch: Use independent noise
            noise_depth = torch.randn((b, c, half_len), device=device)
            
        x_0 = torch.cat([noise_rgb, noise_depth], dim=2)

        # 3. Handle Dual Timesteps
        if t_rgb is None:
            t_rgb = torch.rand((b,), device=device)
        if t_depth is None:
            t_depth = t_rgb if use_same_noise else torch.rand((b,), device=device)

        # Construct expanded timestep vectors for interpolation
        t_rgb_exp = t_rgb.view(b, 1, 1).repeat(1, 1, half_len)
        t_depth_exp = t_depth.view(b, 1, 1).repeat(1, 1, half_len)
        
        # t_joint: [B, 1, Total_Len]
        t_joint_exp = torch.cat([t_rgb_exp, t_depth_exp], dim=2)

        # 4. Interpolate x_t (Optimal Transport Path)
        x_t = (1 - t_joint_exp) * x_0 + t_joint_exp * x_1_joint
        
        # 5. Calculate target velocity v
        target_v = x_1_joint - x_0
        
        # 6. (Unified DiT Prediction)
        v_pred = self.model(
            x_t, 
            cond_joint, 
            t_rgb * self.time_scale_factor, 
            t_depth * self.time_scale_factor
        )

        # 7. Loss Calculation
        v_pred_rgb = v_pred[:, :, :half_len]
        v_pred_depth = v_pred[:, :, half_len:]
        
        target_v_rgb = target_v[:, :, :half_len]
        target_v_depth = target_v[:, :, half_len:]

        loss_rgb = self.get_loss(v_pred_rgb, target_v_rgb)
        loss_depth = self.get_loss(v_pred_depth, target_v_depth)
        
        loss_dict = {
            'train/loss_rgb': loss_rgb.item(),
            'train/loss_depth': loss_depth.item(),
            'train/loss_total': (loss_rgb + loss_depth).item()
        }
        
        return loss_rgb, loss_depth, loss_dict

    @torch.no_grad()
    def sample(self, batch_size=16, cond_rgb=None, cond_depth=None, return_intermediates=False):
        """
        Sampling function: Defaults to Joint Generation, i.e., t_rgb = t_depth
        """
        device = next(self.model.parameters()).device
        shape = (batch_size, self.channels, self.image_size) 
        
        x_rgb = torch.randn(shape, device=device)
        if self.same_noise:
            x_depth = x_rgb.clone()
        else:
            x_depth = torch.randn(shape, device=device)
            
        cond_joint = torch.cat([cond_rgb, cond_depth], dim=2) if (cond_rgb is not None and cond_depth is not None) else None
            
        dt = 1.0 / self.sampling_timesteps
        
        for i in range(self.sampling_timesteps):
            t_value = i / self.sampling_timesteps
            
            t = torch.full((batch_size,), t_value, device=device)
            
            x_t = torch.cat([x_rgb, x_depth], dim=2)
            
            v_pred = self.model(
                x_t, 
                cond_joint, 
                t * self.time_scale_factor, 
                t * self.time_scale_factor
            )
            
            half_len = x_rgb.shape[2]
            v_pred_rgb = v_pred[:, :, :half_len]
            v_pred_depth = v_pred[:, :, half_len:]
            
            x_rgb = x_rgb + v_pred_rgb * dt
            x_depth = x_depth + v_pred_depth * dt
            
        return x_rgb, x_depth