import argparse,torch
from _common import *
from dino_vae_hierarchical_diffusion.src.visualization import save_grid
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--max_images',type=int,default=8);a=p.parse_args();c,d,loader,vae,dino,(ev,en,det,s0)=setup(a);out=ensure_output(c);k=c['hierarchy']['K'];load_checkpoint(checkpoint_path(out,'stage1.pt'),{'evidence':ev,'encoder':en,'deterministic':det},d);b=next(iter(loader));x=b['image'].to(d);_,o=encode(x,vae,dino,ev,en);rev=torch.arange(x.shape[0]-1,-1,-1,device=d);variants=[]
with torch.no_grad():
 for cut in range(1,k+1): variants.append(vae.decode(det(*[o[f'Z{i}'] if i>=cut else o[f'Z{i}'][rev] for i in range(k,0,-1)])))
save_grid(torch.cat([x,*variants]),out/'swapping_grids'/'swaps.png')
