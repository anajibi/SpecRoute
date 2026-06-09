import torch
from torch.nn import functional as F
def kl_loss(mu,logvar): return -.5*(1+logvar-mu.square()-logvar.exp()).mean()
def highpass(x): return x-F.avg_pool2d(x,3,1,1)
def reconstruction_losses(z_hat,z,x_hat,x): return {'z0_det':F.l1_loss(z_hat,z),'img_l1':F.l1_loss(x_hat,x),'hf':F.l1_loss(highpass(x_hat),highpass(x))}
