import argparse,torch
from _common import *
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--max_images',type=int);a=p.parse_args();c,d,loader,vae,dino,(ev,en,det,s0)=setup(a);out=ensure_output(c);load_checkpoint(checkpoint_path(out,'stage1.pt'),{'evidence':ev,'encoder':en},d);en.eval();rows=[]
with torch.no_grad():
 for b in loader:
  z,o=encode(b['image'].to(d),vae,dino,ev,en)
  for j,name in enumerate(b['image_id']): rows.append({'image_id':name,'z0':z[j].cpu(),**{q:v[j].cpu() for q,v in o.items()}})
torch.save(rows,out/'latents.pt');print(f'saved {len(rows)} latents')
