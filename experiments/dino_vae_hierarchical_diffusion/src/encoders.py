"""Top-down stochastic hierarchical encoders."""
import torch
from torch import nn
from torch.nn import functional as F

def sample(mu, logvar, training): return mu + torch.exp(.5*logvar)*torch.randn_like(mu) if training else mu
class FilmBlock(nn.Module):
    def __init__(self, cin, cout, cond):
        super().__init__(); self.conv=nn.Sequential(nn.Conv2d(cin,cout,3,1,1),nn.GroupNorm(min(32,cout),cout),nn.SiLU(),nn.Conv2d(cout,cout,3,1,1)); self.film=nn.Linear(cond,cout*2); self.skip=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x,c):
        scale,bias=self.film(c).chunk(2,1); return F.silu(self.conv(x)*(1+scale[:,:,None,None])+bias[:,:,None,None]+self.skip(x))
class Heads(nn.Module):
    def __init__(self,cin,cout): super().__init__(); self.mu=nn.Conv2d(cin,cout,1); self.lv=nn.Conv2d(cin,cout,1)
    def forward(self,x): return self.mu(x),self.lv(x).clamp(-12,8)
def put(out,name,mu,lv,training): out[name]=sample(mu,lv,training); out['mu'+name[1:]]=mu; out['logvar'+name[1:]]=lv

class HierarchicalEncoderK3(nn.Module):
    def __init__(self):
        super().__init__(); self.e3=nn.Sequential(nn.Linear(1024,1024),nn.SiLU(),nn.Linear(1024,1024)); self.e2=FilmBlock(640,256,512); self.h2=Heads(256,128); self.e1=FilmBlock(640,192,512); self.h1=Heads(192,64)
    def forward(self,e):
        out={}; mu3,lv3=self.e3(torch.cat([e['Fg'],e['Dg']],1)).chunk(2,1); lv3=lv3.clamp(-12,8); put(out,'Z3',mu3,lv3,self.training)
        h2=self.e2(torch.cat([e['F8'],e['D8']],1),out['Z3']); put(out,'Z2',*self.h2(h2),self.training)
        h1=self.e1(torch.cat([e['F16'],e['D16'],F.interpolate(out['Z2'],scale_factor=2,mode='nearest')],1),out['Z3']); put(out,'Z1',*self.h1(h1),self.training); return out

class HierarchicalEncoderK5(nn.Module):
    def __init__(self):
        super().__init__(); self.e5=nn.Linear(1024,1024); self.e4=nn.Sequential(nn.Linear(1536,768),nn.SiLU(),nn.Linear(768,512)); self.e3=FilmBlock(768,256,768); self.h3=Heads(256,128); self.e2=FilmBlock(768,192,768); self.h2=Heads(192,64); self.e1=FilmBlock(576,192,768); self.h1=Heads(192,64)
    def forward(self,e):
        o={}; m,l=self.e5(torch.cat([e['Fg'],e['Dg']],1)).chunk(2,1); put(o,'Z5',m,l.clamp(-12,8),self.training)
        m,l=self.e4(torch.cat([e['Fg'],e['Dg'],o['Z5']],1)).chunk(2,1); put(o,'Z4',m,l.clamp(-12,8),self.training); c=torch.cat([o['Z5'],o['Z4']],1)
        put(o,'Z3',*self.h3(self.e3(torch.cat([e['F4'],e['D4']],1),c)),self.training)
        put(o,'Z2',*self.h2(self.e2(torch.cat([e['F8'],e['D8'],F.interpolate(o['Z3'],scale_factor=2,mode='nearest')],1),c)),self.training)
        put(o,'Z1',*self.h1(self.e1(torch.cat([e['F16'],e['D16'],F.interpolate(o['Z2'],scale_factor=2,mode='nearest')],1),c)),self.training); return o
