"""Stage-one reconstruction losses."""
import torch
import torch.nn.functional as F

def compression_loss(zs): return torch.stack([z.square().mean() for z in zs]).sum()
def reconstruction_loss(z_pred,z_true,x_pred=None,x_true=None,zs=None,lambda_img=.1,lambda_lpips=0.,lpips_model=None,lambda_compress=1e-4):
    parts={"latent":F.mse_loss(z_pred,z_true)}
    if x_pred is not None and x_true is not None: parts["image"]=F.mse_loss(x_pred,x_true)
    if lpips_model is not None and lambda_lpips: parts["lpips"]=lpips_model(x_pred,x_true).mean()
    if zs is not None: parts["compress"]=compression_loss(zs)
    total=parts["latent"]+lambda_img*parts.get("image",0)+lambda_lpips*parts.get("lpips",0)+lambda_compress*parts.get("compress",0)
    return total, parts
