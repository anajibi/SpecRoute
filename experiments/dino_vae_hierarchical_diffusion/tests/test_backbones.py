import torch
from torch import nn
from types import SimpleNamespace
from dino_vae_hierarchical_diffusion.src.backbones import FrozenSDVAE,FrozenDINOv2
class V(nn.Module):
 def __init__(self):super().__init__();self.p=nn.Parameter(torch.ones(1));self.config=SimpleNamespace(scaling_factor=.2)
 def encode(self,x):return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:torch.zeros(x.shape[0],4,32,32)))
 def decode(self,z):return SimpleNamespace(sample=torch.zeros(z.shape[0],3,256,256))
class D(nn.Module):
 def __init__(self):super().__init__();self.p=nn.Parameter(torch.ones(1))
 def forward_features(self,x):return {'x_norm_clstoken':torch.zeros(x.shape[0],768),'x_norm_patchtokens':torch.zeros(x.shape[0],256,768)}
def test_frozen_backbones_and_shapes():
 v=FrozenSDVAE(model=V());d=FrozenDINOv2(model=D());x=torch.randn(2,3,256,256);z=v.encode(x);c,m=d(x);assert not any(p.requires_grad for p in v.parameters());assert not any(p.requires_grad for p in d.parameters());assert z.shape==(2,4,32,32);assert v.decode(z).shape==(2,3,256,256);assert c.shape==(2,768);assert m.shape==(2,768,16,16)
