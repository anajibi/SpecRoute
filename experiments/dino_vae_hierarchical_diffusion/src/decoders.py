"""Deterministic debugging decoder and final latent diffusion decoder."""
import random, torch
from torch import nn
from torch.nn import functional as F
from .diffusion import DiffusionSchedule,timestep_embedding

def apply_level_dropout(zs,k=None,mode=None):
    out={n:z for n,z in zs.items() if n.startswith('Z')}; k=k or len(out)
    if mode is None:
        r=random.random()
        if k == 3:
            mode='all' if r < .7 else ('drop_fine' if r < .8 else 'drop_two' if r < .9 else 'random')
        else:
            mode='all' if r < .6 else ('drop_fine' if r < .7 else 'drop_two' if r < .8 else 'random_mid' if r < .9 else 'coarse_only')
    names=sorted(out,key=lambda x:int(x[1:])); drop=[]
    if mode=='drop_fine': drop=names[:1]
    elif mode=='drop_two': drop=names[:2]
    elif mode=='random': drop=[random.choice(names)]
    elif mode=='random_mid': drop=[random.choice(names[1:-1])]
    elif mode=='coarse_only': drop=names[:-1]
    for n in drop: out[n]=torch.zeros_like(out[n])
    return out

def global_condition(zs): return torch.cat([z for n,z in sorted(zs.items(),reverse=True) if z.ndim==2],1)
def spatial(zs,size): return torch.cat([F.interpolate(z,(size,size),mode='nearest') for z in zs.values() if z.ndim==4],1)
class DeterministicLatentDecoder(nn.Module):
    def __init__(self,k=3):
        super().__init__(); cin=192 if k==3 else 256; gd=512 if k==3 else 768; self.g=nn.Linear(gd,128); self.net=nn.Sequential(nn.Conv2d(cin+128,256,3,1,1),nn.SiLU(),nn.Conv2d(256,128,3,1,1),nn.SiLU(),nn.Conv2d(128,4,3,1,1))
    def forward(self,*levels,dropout_mode=None):
        zs={f'Z{len(levels)-i}':z for i,z in enumerate(levels)}; zs=apply_level_dropout(zs,mode=dropout_mode) if (dropout_mode or self.training) else zs; s=spatial(zs,32); g=self.g(global_condition(zs))[:,:,None,None].expand(-1,-1,32,32); return self.net(torch.cat([s,g],1))
class LatentDecoderDiffusion32x32(nn.Module):
    def __init__(self,k=3,steps=1000):
        super().__init__(); sc=192 if k==3 else 256; gd=512 if k==3 else 768; self.k=k; self.schedule=DiffusionSchedule(steps); self.cond=nn.Linear(gd+128,128); self.net=nn.Sequential(nn.Conv2d(4+sc,256,3,1,1),nn.SiLU(),nn.Conv2d(256,128,3,1,1),nn.SiLU(),nn.Conv2d(128,4,3,1,1))
    def forward(self,x,t,*levels):
        zs={f'Z{len(levels)-i}':z for i,z in enumerate(levels)}; c=self.cond(torch.cat([global_condition(zs),timestep_embedding(t,128)],1))[:,:,None,None]; return self.net(torch.cat([x,spatial(zs,32)],1))+c[:,:4]
    def loss(self,z0,*levels):
        if self.training:
            zs=apply_level_dropout({f'Z{len(levels)-i}':z for i,z in enumerate(levels)})
            levels=tuple(zs[f'Z{i}'] for i in range(len(levels),0,-1))
        t=torch.randint(0,len(self.schedule.alpha_bar),(z0.shape[0],),device=z0.device); noisy,eps=self.schedule.noisy(z0,t); return F.mse_loss(self(noisy,t,*levels),eps)
    def sample(self,*levels,steps=20): return self.schedule.sample(self,(levels[0].shape[0],4,32,32),levels[0].device,levels,steps)
