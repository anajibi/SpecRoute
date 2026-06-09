"""Apply precomputed normalized linear-probe directions; these are pseudo-counterfactual edits."""
import argparse,torch
from _common import *
from dino_vae_hierarchical_diffusion.src.visualization import save_grid
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--directions',required=True);p.add_argument('--level',type=int,default=3);p.add_argument('--alpha',type=float,nargs='+',default=[-2,0,2]);p.add_argument('--max_images',type=int,default=4);a=p.parse_args();c,d,loader,vae,dino,(ev,en,det,s0)=setup(a);out=ensure_output(c);k=c['hierarchy']['K'];load_checkpoint(checkpoint_path(out,'stage1.pt'),{'evidence':ev,'encoder':en,'deterministic':det},d);direction=torch.load(a.directions,map_location=d);x=next(iter(loader))['image'].to(d);_,o=encode(x,vae,dino,ev,en);imgs=[]
with torch.no_grad():
 for alpha in a.alpha:
  q=dict(o);w=direction.expand_as(q[f'Z{a.level}']);q[f'Z{a.level}']=q[f'Z{a.level}']+alpha*w/(w.flatten(1).norm(dim=1).view(-1,*([1]*(w.ndim-1)))+1e-8);imgs.append(vae.decode(det(*levels(q,k))))
save_grid(torch.cat(imgs),out/'direction_grids'/'edits.png')
