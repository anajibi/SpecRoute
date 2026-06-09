import argparse,torch
from _common import *
from dino_vae_hierarchical_diffusion.src.metrics import image_metrics
from dino_vae_hierarchical_diffusion.src.utils import save_csv
from dino_vae_hierarchical_diffusion.src.visualization import save_grid
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--max_images',type=int);p.add_argument('--vae_ceiling',action='store_true');a=p.parse_args();c,d,loader,vae,dino,(ev,en,det,s0)=setup(a);out=ensure_output(c);k=c['hierarchy']['K']; rows=[]
if not a.vae_ceiling: load_checkpoint(checkpoint_path(out,'stage1.pt'),{'evidence':ev,'encoder':en,'deterministic':det,'s0':s0},d)
with torch.no_grad():
 for bi,b in enumerate(loader):
  x=b['image'].to(d);z=vae.encode(x); pred=vae.decode(z) if a.vae_ceiling else vae.decode(det(*levels(encode(x,vae,dino,ev,en)[1],k)));m=image_metrics(pred,x);rows.extend([{'image_id':i,**m} for i in b['image_id']]);save_grid(torch.cat([x,pred]),out/'recon_grids'/f'{bi:05d}.png')
save_csv(rows,out/'metrics'/'reconstruction.csv');print(f'evaluated {len(rows)} images')
