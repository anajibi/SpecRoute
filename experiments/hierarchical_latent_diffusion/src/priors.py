"""Vector diffusion priors conditioned on preceding hierarchy levels."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


def cosine_betas(n: int, s=.008):
    x = torch.linspace(0, n, n + 1); a = torch.cos(((x / n + s) / (1 + s)) * math.pi / 2) ** 2; a = a / a[0]
    return (1 - a[1:] / a[:-1]).clamp(1e-5, .999)

def timestep_embedding(t: torch.Tensor, dim: int):
    half = dim // 2; f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1)); e = t.float()[:, None] * f[None]
    return F.pad(torch.cat([e.cos(), e.sin()], -1), (0, dim % 2))


class DiTMLPBlock(nn.Module):
    """AdaLN-modulated residual MLP block."""
    def __init__(self, dim: int):
        super().__init__(); self.norm = nn.LayerNorm(dim, elementwise_affine=False); self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 3)); self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
    def forward(self, x, context):
        shift, scale, gate = self.mod(context).chunk(3, -1); return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class LevelPrior(nn.Module):
    def __init__(self, z_dim: int, cond_dim: int, hidden_dim=512, num_layers=6, num_timesteps=1000, schedule="cosine"):
        super().__init__(); self.z_dim, self.cond_dim, self.num_timesteps = z_dim, cond_dim, num_timesteps
        betas = cosine_betas(num_timesteps) if schedule == "cosine" else torch.linspace(1e-4, .02, num_timesteps)
        alphas = 1 - betas; abar = alphas.cumprod(0)
        for n, v in {"betas":betas,"alphas":alphas,"alpha_bars":abar}.items(): self.register_buffer(n, v)
        self.in_proj = nn.Linear(z_dim, hidden_dim); self.time_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)); self.cond_proj = nn.Linear(cond_dim, hidden_dim) if cond_dim else None
        self.blocks = nn.ModuleList([DiTMLPBlock(hidden_dim) for _ in range(num_layers)]); self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, z_dim))

    def _check_cond(self, cond, batch):
        if self.cond_dim == 0:
            if cond is not None: raise ValueError("Unconditional prior does not accept cond")
        elif cond is None or cond.shape != (batch, self.cond_dim): raise ValueError(f"cond must have shape ({batch}, {self.cond_dim})")
    def predict_noise(self, z, t, cond=None):
        self._check_cond(cond, z.shape[0]); c = self.time_mlp(timestep_embedding(t, self.in_proj.out_features)); c = c + (self.cond_proj(cond) if self.cond_proj is not None else 0); h = self.in_proj(z)
        for block in self.blocks: h = block(h, c)
        return self.out(h)
    def forward(self, z_clean, cond=None):
        self._check_cond(cond, z_clean.shape[0]); t = torch.randint(self.num_timesteps, (z_clean.shape[0],), device=z_clean.device); noise = torch.randn_like(z_clean); a = self.alpha_bars[t, None]; noisy = a.sqrt()*z_clean + (1-a).sqrt()*noise
        return F.mse_loss(self.predict_noise(noisy, t, cond), noise)
    @torch.no_grad()
    def sample(self, batch_size, cond=None, device="cuda", num_inference_steps=50):
        device = torch.device(device); self._check_cond(cond, batch_size); z = torch.randn(batch_size, self.z_dim, device=device); times = torch.linspace(self.num_timesteps-1, 0, min(num_inference_steps, self.num_timesteps), device=device).long().unique_consecutive()
        for i, t0 in enumerate(times):
            t = torch.full((batch_size,), t0, device=device, dtype=torch.long); abar = self.alpha_bars[t0]; eps = self.predict_noise(z, t, cond); clean = (z-(1-abar).sqrt()*eps)/abar.sqrt(); next_a = self.alpha_bars[times[i+1]] if i+1 < len(times) else torch.tensor(1.,device=device); z = next_a.sqrt()*clean + (1-next_a).sqrt()*eps
        return z
    @torch.no_grad()
    def invert(self, z_clean, cond=None, num_steps=50):
        self._check_cond(cond, z_clean.shape[0]); z=z_clean; times=torch.linspace(0,self.num_timesteps-1,min(num_steps,self.num_timesteps),device=z.device).long().unique_consecutive()
        for t0 in times[1:]:
            t=torch.full((z.shape[0],),t0,device=z.device,dtype=torch.long); eps=self.predict_noise(z,t,cond); a=self.alpha_bars[t0]; z=a.sqrt()*z_clean+(1-a).sqrt()*eps
        return z


class HierarchicalPriorStack(nn.Module):
    def __init__(self, level_dims, prior_kwargs=None):
        super().__init__(); self.level_dims=list(level_dims); kw=prior_kwargs or {}; self.levels=nn.ModuleList([LevelPrior(d,sum(level_dims[:i]),**kw) for i,d in enumerate(level_dims)])
    def compute_loss(self,zs):
        if len(zs)!=len(self.levels): raise ValueError("Incorrect number of levels")
        losses=[p(zs[i], None if i==0 else torch.cat(zs[:i],-1)) for i,p in enumerate(self.levels)]; return torch.stack(losses).sum(), losses
    @torch.no_grad()
    def sample_full(self,batch_size,device="cuda",num_inference_steps=50):
        zs=[]
        for i,p in enumerate(self.levels): zs.append(p.sample(batch_size,None if i==0 else torch.cat(zs,-1),device,num_inference_steps))
        return zs
