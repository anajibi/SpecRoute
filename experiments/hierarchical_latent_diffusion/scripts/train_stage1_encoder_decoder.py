#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,torch
from src.backbones import DINOv2Backbone,SDVAEBackbone
from src.datasets import image_loader
from src.losses import reconstruction_loss
from src.utils import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--epochs',type=int);p.add_argument('--max-images',type=int);a=p.parse_args();cfg=load_config(a.config);seed_everything(cfg['seed']);dev=get_device(cfg['device']);out=output_dir(cfg);save_config(cfg,out/'config.yaml')
 dino=DINOv2Backbone(config=cfg.get('dinov2',{})).to(dev);vae=SDVAEBackbone(config=cfg.get('sd_vae',{})).to(dev);enc,dec,_=build_trainable(cfg);enc,dec=enc.to(dev),dec.to(dev);opt=torch.optim.AdamW(list(enc.parameters())+list(dec.parameters()),lr=cfg['stage1']['lr']);epochs=a.epochs or cfg['stage1']['epochs']
 for epoch in range(epochs):
  enc.train();dec.train();total=0
  for x,_ in image_loader(cfg,max_images=a.max_images):
   x=x.to(dev);zs=enc(dino(x));pred=dec(zs);true=vae.encode(x);loss,_=reconstruction_loss(pred,true,zs=zs,lambda_compress=cfg['stage1'].get('lambda_compress',1e-4));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()),1.);opt.step();total+=loss.item()
  append_csv(out/'stage1_metrics.csv',{'epoch':epoch+1,'loss':total});print(f'epoch={epoch+1} loss={total:.5f}')
 save_checkpoint(out/'stage1.pt',encoder=enc.state_dict(),decoder=dec.state_dict(),config=cfg)
if __name__=='__main__':main()
