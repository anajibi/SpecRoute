import torch
from torch.nn import functional as F
from .losses import highpass
def image_metrics(a,b): return {'mse':F.mse_loss(a,b).item(),'hf_l1':F.l1_loss(highpass(a),highpass(b)).item(),'hf_mse':F.mse_loss(highpass(a),highpass(b)).item()}
def cosine(a,b): return F.cosine_similarity(a,b).mean().item()
