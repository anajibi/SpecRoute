"""Shared VAE/DINO evidence pyramid; raw RGB never enters this module."""
import torch
from torch import nn

class DownBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, 2, 1), nn.GroupNorm(min(32, cout), cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, 1, 1), nn.SiLU())

class EvidencePyramid(nn.Module):
    def __init__(self, dino_dim=768):
        super().__init__()
        self.f32=nn.Sequential(nn.Conv2d(4,128,3,1,1),nn.SiLU()); self.f16=DownBlock(128,256); self.f8=DownBlock(256,384); self.f4=DownBlock(384,512)
        self.d16=nn.Sequential(nn.Conv2d(dino_dim,256,1),nn.SiLU()); self.d8=DownBlock(256,256); self.d4=DownBlock(256,256)
        self.dg=nn.Sequential(nn.Linear(dino_dim,512),nn.SiLU(),nn.Linear(512,512))
    def forward(self,z0,dino_cls,dino_map):
        f32=self.f32(z0); f16=self.f16(f32); f8=self.f8(f16); f4=self.f4(f8)
        d16=self.d16(dino_map); d8=self.d8(d16); d4=self.d4(d8)
        return {"F32":f32,"F16":f16,"F8":f8,"F4":f4,"Fg":f4.mean((2,3)),"D16":d16,"D8":d8,"D4":d4,"Dg":self.dg(dino_cls)}
