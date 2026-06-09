import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from types import SimpleNamespace
import torch
from torch import nn
from src.backbones import DINOv2Backbone,SDVAEBackbone
class Dino(nn.Module):
 def __init__(self):super().__init__();self.p=nn.Parameter(torch.ones(1))
 def forward_features(self,x):return {'x_norm_clstoken':x.mean((2,3)),'x_norm_patchtokens':x.flatten(2).transpose(1,2)}
class Dist:
 def __init__(self,x):self.x=x
 def mode(self):return self.x
class VAE(nn.Module):
 def __init__(self):super().__init__();self.p=nn.Parameter(torch.ones(1));self.config=SimpleNamespace(scaling_factor=.5)
 def encode(self,x):return SimpleNamespace(latent_dist=Dist(x[:,:1]))
 def decode(self,z):return SimpleNamespace(sample=z.repeat(1,3,1,1))
def test_backbones_hard_frozen_and_detached():
 d=DINOv2Backbone(model=Dino());v=SDVAEBackbone(model=VAE());d.train();v.train();assert not d.training and not v.training;assert all(not p.requires_grad for p in d.parameters());assert all(not p.requires_grad for p in v.parameters());x=torch.randn(2,3,16,16,requires_grad=True);assert not d(x).requires_grad and not v.encode(x).requires_grad and not v.decode(torch.randn(2,1,4,4,requires_grad=True)).requires_grad
