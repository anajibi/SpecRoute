"""Vector and spatial conditional diffusion priors."""
import torch
from torch import nn
from torch.nn import functional as F
from .diffusion import DiffusionSchedule,timestep_embedding
class VectorDiffusionPrior(nn.Module):
    def __init__(self,dim,cond_dim=0,steps=1000):
        super().__init__(); self.dim=dim; self.cond_dim=cond_dim; self.schedule=DiffusionSchedule(steps); self.net=nn.Sequential(nn.Linear(dim+cond_dim+128,1024),nn.SiLU(),nn.Linear(1024,1024),nn.SiLU(),nn.Linear(1024,dim))
    def forward(self,x,t,*conditions): return self.net(torch.cat([x,timestep_embedding(t,128),*conditions],1))
    def loss(self,x,*c): t=torch.randint(0,len(self.schedule.alpha_bar),(x.shape[0],),device=x.device); n,e=self.schedule.noisy(x,t); return F.mse_loss(self(n,t,*c),e)
    def sample(self,*c,batch_size=None,steps=20): b=c[0].shape[0] if c else batch_size; device=c[0].device if c else next(self.parameters()).device; return self.schedule.sample(self,(b,self.dim),device,c,steps)
class SpatialDiffusionPrior(nn.Module):
    def __init__(self,channels,size,spatial_cond_channels=0,global_cond_dim=0,steps=1000):
        super().__init__(); self.channels=channels; self.size=size; self.schedule=DiffusionSchedule(steps); self.g=nn.Linear(global_cond_dim+128,128); self.net=nn.Sequential(nn.Conv2d(channels+spatial_cond_channels,256,3,1,1),nn.SiLU(),nn.Conv2d(256,256,3,1,1),nn.SiLU(),nn.Conv2d(256,channels,3,1,1))
    def forward(self,x,t,spatial_cond=None,global_cond=None):
        parts=[x];
        if spatial_cond is not None: parts.append(F.interpolate(spatial_cond,(self.size,self.size),mode='nearest'))
        g=torch.zeros(x.shape[0],0,device=x.device) if global_cond is None else global_cond; bias=self.g(torch.cat([g,timestep_embedding(t,128)],1))[:,:self.channels,None,None]; return self.net(torch.cat(parts,1))+bias
    def loss(self,x,spatial_cond=None,global_cond=None): t=torch.randint(0,len(self.schedule.alpha_bar),(x.shape[0],),device=x.device); n,e=self.schedule.noisy(x,t); return F.mse_loss(self(n,t,spatial_cond,global_cond),e)
    def sample(self,spatial_cond=None,global_cond=None,batch_size=None,steps=20): ref=spatial_cond if spatial_cond is not None else global_cond; b=ref.shape[0] if ref is not None else batch_size; device=ref.device if ref is not None else next(self.parameters()).device; return self.schedule.sample(self,(b,self.channels,self.size,self.size),device,(spatial_cond,global_cond),steps)
class SpatialDiffusionPrior8x8(SpatialDiffusionPrior):
    def __init__(self,channels=128,**kw): super().__init__(channels,8,**kw)
class SpatialDiffusionPrior16x16(SpatialDiffusionPrior):
    def __init__(self,channels=64,**kw): super().__init__(channels,16,**kw)
