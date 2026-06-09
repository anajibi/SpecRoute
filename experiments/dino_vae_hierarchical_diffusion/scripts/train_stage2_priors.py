import argparse,torch
from pathlib import Path
from _common import ROOT
from dino_vae_hierarchical_diffusion.src.utils import load_config,ensure_output
from dino_vae_hierarchical_diffusion.src.priors import VectorDiffusionPrior,SpatialDiffusionPrior
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--epochs',type=int);p.add_argument('--max_images',type=int);a=p.parse_args();c=load_config(a.config);d=torch.device(c['device'] if torch.cuda.is_available() else 'cpu');out=ensure_output(c);data=torch.load(out/'latents.pt',map_location=d);data=data[:a.max_images] if a.max_images else data;k=c['hierarchy']['K']
if k==3: pri={'s3':VectorDiffusionPrior(512).to(d),'s2':SpatialDiffusionPrior(128,8,0,512).to(d),'s1':SpatialDiffusionPrior(64,16,128,512).to(d)}
else: pri={'s5':VectorDiffusionPrior(512).to(d),'s4':VectorDiffusionPrior(256,512).to(d),'s3':SpatialDiffusionPrior(128,4,0,768).to(d),'s2':SpatialDiffusionPrior(64,8,128,768).to(d),'s1':SpatialDiffusionPrior(64,16,64,768).to(d)}
opt=torch.optim.AdamW([p for m in pri.values() for p in m.parameters()],lr=c['optim']['lr'])
for epoch in range(a.epochs or c['train']['stage2_epochs']):
 for r in data:
  z={q:r[q].unsqueeze(0).to(d) for q in r if q.startswith('Z')}; loss=pri[f's{k}'].loss(z[f'Z{k}']);
  if k==3: loss=loss+pri['s2'].loss(z['Z2'],None,z['Z3'])+pri['s1'].loss(z['Z1'],z['Z2'],z['Z3'])
  else:
   g=torch.cat([z['Z5'],z['Z4']],1);loss=loss+pri['s4'].loss(z['Z4'],z['Z5'])+pri['s3'].loss(z['Z3'],None,g)+pri['s2'].loss(z['Z2'],z['Z3'],g)+pri['s1'].loss(z['Z1'],z['Z2'],g)
  opt.zero_grad();loss.backward();opt.step()
 print(f'epoch={epoch+1} loss={loss.item():.4f}')
(out/'checkpoints').mkdir(exist_ok=True);torch.save({n:m.state_dict() for n,m in pri.items()},out/'checkpoints'/'priors.pt')
