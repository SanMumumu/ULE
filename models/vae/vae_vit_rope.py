import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vae.vit_modules import TimeSformerEncoder, TimeSformerDecoder
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ---------------------------------------------------------------------
# VAE Utilities 
# ---------------------------------------------------------------------
class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def mode(self):
        return self.mean
    
# ---------------------------------------------------------------------
# 3D RoPE Utilities
# ---------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        inv_freq = 1.0 / (max_period ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len=None):
        # x: can be used to determine device/dtype, or passed as existing indices
        if seq_len is None:
            seq_len = x.shape[1] 
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(t, freqs):
    # t: [batch, seq_len, head_dim] or similar
    # freqs: [batch, seq_len, head_dim] (broadcastable)
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())

def apply_3d_rope(q, k, freqs_3d):
    """
    Applies 3D RoPE by splitting the head dimension into 3 parts.
    freqs_3d: list/tuple of [freqs_dim1, freqs_dim2, freqs_dim3]
              Each freq tensor should correspond to the split dimension size.
              e.g. for h_xy (Time, Height, Width), these correspond to T, H, W rotations.
    """
    if freqs_3d is None:
        return q, k

    # Split Q and K into 3 chunks along the last dimension (head_dim)
    # Note: dim_head might not be perfectly divisible by 3, so we handle remainders.
    dim = q.shape[-1]
    d1 = dim // 3
    d2 = dim // 3
    d3 = dim - d1 - d2 # Remainder goes to last chunk
    
    q_splits = torch.split(q, [d1, d2, d3], dim=-1)
    k_splits = torch.split(k, [d1, d2, d3], dim=-1)
    
    f1, f2, f3 = freqs_3d

    # Apply rotations to each split
    # We assume freqs are broadcastable to the split shape
    # q_splits[0] shape: [B, H, N, d1]
    # f1 shape:          [B, 1, N, d1] (example)
    
    q_out = []
    k_out = []
    
    # Chunk 1
    q_out.append(apply_rotary_pos_emb(q_splits[0], f1))
    k_out.append(apply_rotary_pos_emb(k_splits[0], f1))
    
    # Chunk 2
    q_out.append(apply_rotary_pos_emb(q_splits[1], f2))
    k_out.append(apply_rotary_pos_emb(k_splits[1], f2))
    
    # Chunk 3
    q_out.append(apply_rotary_pos_emb(q_splits[2], f3))
    k_out.append(apply_rotary_pos_emb(k_splits[2], f3))
    
    q_new = torch.cat(q_out, dim=-1)
    k_new = torch.cat(k_out, dim=-1)
    
    return q_new, k_new

# ---------------------------------------------------------------------
# Modified Layers
# ---------------------------------------------------------------------

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, **kwargs): # Accept kwargs to ignore freqs
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout_p = dropout
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, freqs_3d=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Apply 3D RoPE to Q and K
        if freqs_3d is not None:
             q, k = apply_3d_rope(q, k, freqs_3d)

        out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, freqs_3d=None):
        for attn, ff in self.layers:
            x = attn(x, freqs_3d=freqs_3d) + x
            x = ff(x) + x
        return x


# ---------------------------------------------------------------------
# ViTAutoencoder with 3D RoPE
# ---------------------------------------------------------------------

class ViTAutoencoder(nn.Module):
    def __init__(self,
                 embed_dim,
                 vaeconfig,
                 image_key="image",
                 ):
        super().__init__()
        self.splits = vaeconfig["splits"]
        self.s = vaeconfig["frames"] // self.splits

        if isinstance(vaeconfig["resolution"], int):
            self.res_h = vaeconfig["resolution"]
            self.res_w = vaeconfig["resolution"]
        else:
            self.res_h = vaeconfig["resolution"][0]
            self.res_w = vaeconfig["resolution"][1]

        self.embed_dim = embed_dim
        self.image_key = image_key

        patch_size = 8
        self.down = 3

        max_res = self.res_h if self.res_h > self.res_w else self.res_w
        if max_res == 128 or max_res == 64:
            patch_size = 4
            self.down = 2

        self.encoder = TimeSformerEncoder(dim=vaeconfig["channels"],
                                          image_size=max_res,
                                          num_frames=vaeconfig["frames"],
                                          channels=vaeconfig["in_channels"],
                                          depth=vaeconfig["layers"] if "layers" in vaeconfig else 8,
                                          patch_size=patch_size)

        self.decoder = TimeSformerDecoder(dim=vaeconfig["channels"],
                                          image_size=max_res,
                                          num_frames=vaeconfig["frames"],
                                          channels=vaeconfig["in_channels"],
                                          depth=vaeconfig["layers"] if "layers" in vaeconfig else 8,
                                          patch_size=patch_size)

        self.to_pixel = nn.Sequential(
            Rearrange('b (t h w) c -> (b t) c h w', h=self.res_h // patch_size, w=self.res_w // patch_size),
            nn.ConvTranspose2d(vaeconfig["channels"], vaeconfig["out_channels"], kernel_size=(patch_size, patch_size),
                               stride=patch_size),
        )

        self.act = nn.Sigmoid()
        ts = torch.linspace(-1, 1, steps=self.s).unsqueeze(-1)
        self.register_buffer('coords', ts)

        # Learnable tokens (CLS style)
        self.xy_token = nn.Parameter(torch.randn(1, 1, vaeconfig["channels"]))
        self.xt_token = nn.Parameter(torch.randn(1, 1, vaeconfig["channels"]))
        self.yt_token = nn.Parameter(torch.randn(1, 1, vaeconfig["channels"]))

        # --- ADDED: Rotary Embeddings ---
        # Calculate dim_head used in Transformers
        dim = vaeconfig["channels"]
        heads = 4
        self.dim_head = dim // 8 
        
        d1 = self.dim_head // 3
        d2 = self.dim_head // 3
        d3 = self.dim_head - d1 - d2
        
        self.rope_d1 = RotaryEmbedding(d1)
        self.rope_d2 = RotaryEmbedding(d2)
        self.rope_d3 = RotaryEmbedding(d3)

        # Transformers
        self.xy_quant_attn = Transformer(dim, 4, heads, self.dim_head, 512)
        self.yt_quant_attn = Transformer(dim, 4, heads, self.dim_head, 512)
        self.xt_quant_attn = Transformer(dim, 4, heads, self.dim_head, 512)
        
        self.kl_weight = vaeconfig.get("kl_weight", 1e-6)

        self.pre_xy = torch.nn.Conv2d(dim, 2 * self.embed_dim, 1)
        self.pre_xt = torch.nn.Conv2d(dim, 2 * self.embed_dim, 1)
        self.pre_yt = torch.nn.Conv2d(dim, 2 * self.embed_dim, 1)

        self.post_xy = torch.nn.Conv2d(self.embed_dim, dim, 1)
        self.post_xt = torch.nn.Conv2d(self.embed_dim, dim, 1)
        self.post_yt = torch.nn.Conv2d(self.embed_dim, dim, 1)

    def get_rope_freqs(self, device, h_idx, w_idx, t_idx):
        f_h1 = self.rope_d1(torch.zeros(1, h_idx, device=device)).squeeze(0) 
        f_h2 = self.rope_d2(torch.zeros(1, h_idx, device=device)).squeeze(0) 
        f_h3 = self.rope_d3(torch.zeros(1, h_idx, device=device)).squeeze(0) 
        
        f_w1 = self.rope_d1(torch.zeros(1, w_idx, device=device)).squeeze(0) 
        f_w2 = self.rope_d2(torch.zeros(1, w_idx, device=device)).squeeze(0)
        f_w3 = self.rope_d3(torch.zeros(1, w_idx, device=device)).squeeze(0)

        f_t1 = self.rope_d1(torch.zeros(1, t_idx, device=device)).squeeze(0)
        f_t2 = self.rope_d2(torch.zeros(1, t_idx, device=device)).squeeze(0)
        f_t3 = self.rope_d3(torch.zeros(1, t_idx, device=device)).squeeze(0)
        
        return (f_h1, f_h2, f_h3), (f_w1, f_w2, f_w3), (f_t1, f_t2, f_t3)

    def construct_3d_freqs(self, shape_info, rope_freqs, mode):
        B, H, W, T = shape_info
        (fh1, fh2, fh3), (fw1, fw2, fw3), (ft1, ft2, ft3) = rope_freqs
        
        def pad_freq(f, num=1):
            return F.pad(f, (0, 0, 0, num), value=0.0)

        if mode == 'xy':
            ft1_seq = pad_freq(ft1).unsqueeze(0).expand(B*H*W, -1, -1)
            fh2_seq = fh2.view(1, H, 1, 1, -1).expand(B, -1, W, 1, -1).reshape(B*H*W, 1, -1).expand(-1, T+1, -1)
            fw3_seq = fw3.view(1, 1, W, 1, -1).expand(B, H, -1, 1, -1).reshape(B*H*W, 1, -1).expand(-1, T+1, -1)
            return [ft1_seq.unsqueeze(1), fh2_seq.unsqueeze(1), fw3_seq.unsqueeze(1)] 

        elif mode == 'yt':
            fh1_seq = pad_freq(fh1).unsqueeze(0).expand(B*T*W, -1, -1)
            ft2_seq = ft2.view(1, T, 1, 1, -1).expand(B, -1, W, 1, -1).reshape(B*T*W, 1, -1).expand(-1, H+1, -1)
            fw3_seq = fw3.view(1, 1, W, 1, -1).expand(B, T, -1, 1, -1).reshape(B*T*W, 1, -1).expand(-1, H+1, -1)
            return [fh1_seq.unsqueeze(1), ft2_seq.unsqueeze(1), fw3_seq.unsqueeze(1)]

        elif mode == 'xt':
            fw1_seq = pad_freq(fw1).unsqueeze(0).expand(B*T*H, -1, -1)
            ft2_seq = ft2.view(1, T, 1, 1, -1).expand(B, -1, H, 1, -1).reshape(B*T*H, 1, -1).expand(-1, W+1, -1)
            fh3_seq = fh3.view(1, 1, H, 1, -1).expand(B, T, -1, 1, -1).reshape(B*T*H, 1, -1).expand(-1, W+1, -1)
            return [fw1_seq.unsqueeze(1), ft2_seq.unsqueeze(1), fh3_seq.unsqueeze(1)]

        return None

    def encode(self, x, return_1d=False):
        b = x.size(0)
        x = rearrange(x, 'b c t h w -> b t c h w')
        
        # --- ENCODER PASS ---
        h = self.encoder(x)
        
        eff_h = self.res_h // (2 ** self.down)
        eff_w = self.res_w // (2 ** self.down)
        eff_t = self.s
        
        h = rearrange(h, 'b (t h w) c -> b c t h w', t=eff_t, h=eff_h)
        rope_freqs = self.get_rope_freqs(x.device, eff_h, eff_w, eff_t)

        # --- XY PLANE ---
        h_xy = rearrange(h, 'b c t h w -> (b h w) t c')
        xy_token = repeat(self.xy_token, '1 1 d -> bhw 1 d', bhw=h_xy.size(0))
        h_xy = torch.cat([h_xy, xy_token], dim=1)
        freqs_xy = self.construct_3d_freqs((b, eff_h, eff_w, eff_t), rope_freqs, 'xy')
        h_xy = self.xy_quant_attn(h_xy, freqs_3d=freqs_xy)[:, -1] 
        h_xy = rearrange(h_xy, '(b h w) c -> b c h w', b=b, h=eff_h)

        # --- YT PLANE ---
        h_yt = rearrange(h, 'b c t h w -> (b t w) h c')
        yt_token = repeat(self.yt_token, '1 1 d -> btw 1 d', btw=h_yt.size(0))
        h_yt = torch.cat([h_yt, yt_token], dim=1)
        freqs_yt = self.construct_3d_freqs((b, eff_h, eff_w, eff_t), rope_freqs, 'yt')
        h_yt = self.yt_quant_attn(h_yt, freqs_3d=freqs_yt)[:, -1]
        h_yt = rearrange(h_yt, '(b t w) c -> b c t w', b=b, w=eff_w)

        # --- XT PLANE ---
        h_xt = rearrange(h, 'b c t h w -> (b t h) w c')
        xt_token = repeat(self.xt_token, '1 1 d -> bth 1 d', bth=h_xt.size(0))
        h_xt = torch.cat([h_xt, xt_token], dim=1)
        freqs_xt = self.construct_3d_freqs((b, eff_h, eff_w, eff_t), rope_freqs, 'xt')
        h_xt = self.xt_quant_attn(h_xt, freqs_3d=freqs_xt)[:, -1]
        h_xt = rearrange(h_xt, '(b t h) c -> b c t h', b=b, h=eff_h)

        # --- Post Processing ---
        moments_xy = self.pre_xy(h_xy)
        posterior_xy = DiagonalGaussianDistribution(moments_xy)
        z_xy = posterior_xy.sample()

        moments_yt = self.pre_yt(h_yt)
        posterior_yt = DiagonalGaussianDistribution(moments_yt)
        z_yt = posterior_yt.sample()

        moments_xt = self.pre_xt(h_xt)
        posterior_xt = DiagonalGaussianDistribution(moments_xt)
        z_xt = posterior_xt.sample()

        # 4. KL Loss
        kl_loss = (posterior_xy.kl() + posterior_yt.kl() + posterior_xt.kl())
        kl_loss = torch.mean(kl_loss) * self.kl_weight

        h_xy_dec = self.post_xy(z_xy)
        h_yt_dec = self.post_yt(z_yt)
        h_xt_dec = self.post_xt(z_xt)

        h_xy_dec = h_xy_dec.unsqueeze(-3).expand(-1, -1, self.s, -1, -1)
        h_yt_dec = h_yt_dec.unsqueeze(-2).expand(-1, -1, -1, eff_h, -1)
        h_xt_dec = h_xt_dec.unsqueeze(-1).expand(-1, -1, -1, -1, eff_w)

        h_combined = h_xy_dec + h_yt_dec + h_xt_dec

        if return_1d:
            flat_xy = z_xy.view(z_xy.size(0), z_xy.size(1), -1)
            flat_yt = z_yt.view(z_yt.size(0), z_yt.size(1), -1)
            flat_xt = z_xt.view(z_xt.size(0), z_xt.size(1), -1)

            z_extracted = torch.cat([flat_xy, flat_yt, flat_xt], dim=-1)
            return h_combined, kl_loss, z_extracted

        return h_combined, kl_loss

    def decode(self, z):
        dec = self.decoder(z) 
        return 2 * self.act(self.to_pixel(dec)).contiguous() - 1

    def forward(self, input, return_extract=False):
        input = rearrange(input, 'b c (n t) h w -> (b n) c t h w', n=self.splits)
        if return_extract:
            z_dec, kl_loss, z_extracted = self.encode(input, return_1d=True)
            dec = self.decode(z_dec)
            return dec, kl_loss, z_extracted
        else:
            z_dec, kl_loss = self.encode(input, return_1d=False)
            dec = self.decode(z_dec)
            return dec, kl_loss


    def extract(self, x):
        b = x.size(0)
        x = rearrange(x, 'b c t h w -> b t c h w')
        h = self.encoder(x)
        eff_h = self.res_h // (2 ** self.down)
        eff_w = self.res_w // (2 ** self.down)
        eff_t = self.s
        h = rearrange(h, 'b (t h w) c -> b c t h w', t=eff_t, h=eff_h)
        rope_freqs = self.get_rope_freqs(x.device, eff_h, eff_w, eff_t)

        # XY
        h_xy = rearrange(h, 'b c t h w -> (b h w) t c')
        xy_token = repeat(self.xy_token, '1 1 d -> bhw 1 d', bhw=h_xy.size(0))
        h_xy = torch.cat([h_xy, xy_token], dim=1)
        freqs_xy = self.construct_3d_freqs((b, eff_h, eff_w, eff_t), rope_freqs, 'xy')
        h_xy = self.xy_quant_attn(h_xy, freqs_3d=freqs_xy)[:, -1] 
        h_xy = rearrange(h_xy, '(b h w) c -> b c h w', b=b, h=eff_h)

        # YT
        h_yt = rearrange(h, 'b c t h w -> (b t w) h c')
        yt_token = repeat(self.yt_token, '1 1 d -> btw 1 d', btw=h_yt.size(0))
        h_yt = torch.cat([h_yt, yt_token], dim=1)
        freqs_yt = self.construct_3d_freqs((b, eff_h, eff_w, eff_t), rope_freqs, 'yt')
        h_yt = self.yt_quant_attn(h_yt, freqs_3d=freqs_yt)[:, -1]
        h_yt = rearrange(h_yt, '(b t w) c -> b c t w', b=b, w=eff_w)

        # XT
        h_xt = rearrange(h, 'b c t h w -> (b t h) w c')
        xt_token = repeat(self.xt_token, '1 1 d -> bth 1 d', bth=h_xt.size(0))
        h_xt = torch.cat([h_xt, xt_token], dim=1)
        freqs_xt = self.construct_3d_freqs((b, eff_h, eff_w, eff_t), rope_freqs, 'xt')
        h_xt = self.xt_quant_attn(h_xt, freqs_3d=freqs_xt)[:, -1]
        h_xt = rearrange(h_xt, '(b t h) c -> b c t h', b=b, h=eff_h)

        # MLP & Mode Extraction
        moments_xy = self.pre_xy(h_xy)
        h_xy = DiagonalGaussianDistribution(moments_xy).mode()
    
        moments_yt = self.pre_yt(h_yt)
        h_yt = DiagonalGaussianDistribution(moments_yt).mode()
        
        moments_xt = self.pre_xt(h_xt)
        h_xt = DiagonalGaussianDistribution(moments_xt).mode()

        h_xy = h_xy.view(h_xy.size(0), h_xy.size(1), -1)
        h_yt = h_yt.view(h_yt.size(0), h_yt.size(1), -1)
        h_xt = h_xt.view(h_xt.size(0), h_xt.size(1), -1)

        ret = torch.cat([h_xy, h_yt, h_xt], dim=-1)
        return ret
    
    # decode_from_sample remains unchanged logic-wise
    def decode_from_sample(self, h):

        res1 = self.res_h // (2**self.down)
        res2 = self.res_w // (2**self.down)

        h_xy = h[:, :, 0:res1*res2].view(h.size(0), h.size(1), res1, res2)
        h_yt = h[:, :, res1*res2:res2*(res1+self.s)].view(h.size(0), h.size(1), self.s, res2)
        h_xt = h[:, :, res2*(res1+self.s):res2*(res1+self.s)+res1*self.s].view(h.size(0), h.size(1), self.s, res1)

        h_xy = self.post_xy(h_xy)
        h_yt = self.post_yt(h_yt)
        h_xt = self.post_xt(h_xt)

        h_xy = h_xy.unsqueeze(-3).expand(-1,-1,self.s,-1, -1)
        h_yt = h_yt.unsqueeze(-2).expand(-1,-1,-1,res1, -1)
        h_xt = h_xt.unsqueeze(-1).expand(-1,-1,-1,-1,res2)

        z = h_xy + h_yt + h_xt

        b = z.size(0)
        dec = self.decoder(z)
        return 2*self.act(self.to_pixel(dec)).contiguous()-1
