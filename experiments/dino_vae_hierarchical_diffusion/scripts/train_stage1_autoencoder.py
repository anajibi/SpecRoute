import argparse,torch
from _common import *
from dino_vae_hierarchical_diffusion.src.losses import kl_loss,highpass
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--max_images',type=int);p.add_argument('--epochs',type=int);a=p.parse_args(); c,d,loader,vae,dino,(ev,en,det,s0)=setup(a); k=c['hierarchy']['K']; out=ensure_output(c); params=[*ev.parameters(),*en.parameters(),*det.parameters(),*s0.parameters()]; opt=torch.optim.AdamW(params,lr=c['optim']['lr'])
for epoch in range(a.epochs or c['train']['stage1_epochs']):
 for batch in loader:
  x=batch['image'].to(d); z,o=encode(x,vae,dino,ev,en); lv=levels(o,k); zd=det(*lv); xd=vae.decode(zd); loss=s0.loss(z,*lv)+torch.nn.functional.l1_loss(zd,z)+.1*torch.nn.functional.l1_loss(xd,x)+.05*torch.nn.functional.l1_loss(highpass(xd),highpass(x))+c['loss']['beta_kl']*sum(kl_loss(o['mu'+str(i)],o['logvar'+str(i)]) for i in range(1,k+1)); opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,c['optim']['grad_clip']);opt.step()
 print(f'epoch={epoch+1} loss={loss.item():.4f}')
(out/'checkpoints').mkdir(parents=True,exist_ok=True); torch.save({'evidence':ev.state_dict(),'encoder':en.state_dict(),'deterministic':det.state_dict(),'s0':s0.state_dict(),'config':c},checkpoint_path(out,'stage1.pt'))
