"""Small epsilon-prediction diffusion utilities."""
import math, torch
from torch import nn

def timestep_embedding(t, dim):
    half=dim//2; f=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/max(half-1,1)); x=t.float()[:,None]*f[None]; return torch.cat([x.sin(),x.cos()],1)
class DiffusionSchedule(nn.Module):
    def __init__(self, steps=1000):
        super().__init__(); s=.008; x=torch.linspace(0,steps,steps+1); a=torch.cos(((x/steps+s)/(1+s))*math.pi*.5)**2; a=a/a[0]; self.register_buffer('alpha_bar',(a[1:]/a[:-1]).cumprod(0).clamp(1e-5,1))
    def noisy(self,x,t,noise=None):
        noise=torch.randn_like(x) if noise is None else noise; a=self.alpha_bar[t].view(-1,*([1]*(x.ndim-1))); return a.sqrt()*x+(1-a).sqrt()*noise,noise
    @torch.no_grad()
    def sample(self,model,shape,device,conditions=(),steps=20):
        x=torch.randn(shape,device=device); indices=torch.linspace(len(self.alpha_bar)-1,0,steps,device=device).long()
        for t0 in indices:
            t=torch.full((shape[0],),int(t0),device=device,dtype=torch.long); a=self.alpha_bar[t].view(-1,*([1]*(x.ndim-1))); eps=model(x,t,*conditions); x=(x-(1-a).sqrt()*eps)/a.sqrt().clamp_min(1e-4)
            if t0>0: x=a.sqrt()*x+(1-a).sqrt()*torch.randn_like(x)
        return x
