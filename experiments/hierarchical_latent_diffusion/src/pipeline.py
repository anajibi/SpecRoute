"""Shared loading and intervention operations for scripts."""
import torch
from .backbones import DINOv2Backbone,SDVAEBackbone
from .utils import build_trainable,freeze,load_checkpoint

def load_models(cfg,device,need_backbones=True,need_priors=True):
    enc,dec,pri=build_trainable(cfg); out=__import__("pathlib").Path(__file__).parents[1]/cfg["output_dir"]
    s1=load_checkpoint(out/"stage1.pt",device);enc.load_state_dict(s1["encoder"]);dec.load_state_dict(s1["decoder"]);freeze(enc);freeze(dec)
    if need_priors:
        s2=load_checkpoint(out/"stage2.pt",device);pri.load_state_dict(s2["priors"]);freeze(pri)
    dino=sd=None
    if need_backbones:dino=DINOv2Backbone(config=cfg.get("dinov2",{}));sd=SDVAEBackbone(config=cfg.get("sd_vae",{}));dino.to(device);sd.to(device)
    return dino,sd,enc.to(device),dec.to(device),pri.to(device)

@torch.no_grad()
def resample_tail(zs,priors,start,num_steps=50):
    out=[z.clone() for z in zs]
    for i in range(start,len(out)):out[i]=priors.levels[i].sample(out[0].shape[0],None if i==0 else torch.cat(out[:i],-1),out[0].device,num_steps)
    return out
