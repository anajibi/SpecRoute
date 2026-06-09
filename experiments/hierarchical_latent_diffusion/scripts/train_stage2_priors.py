#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,torch
from torch.utils.data import DataLoader
from src.datasets import LatentDataset
from src.utils import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--epochs',type=int);a=p.parse_args();cfg=load_config(a.config);seed_everything(cfg['seed']);dev=get_device(cfg['device']);out=output_dir(cfg);_,_,pri=build_trainable(cfg);pri=pri.to(dev);opt=torch.optim.AdamW(pri.parameters(),lr=cfg['stage2']['lr']);loader=DataLoader(LatentDataset(out/'latents.pt'),batch_size=cfg['stage2']['batch_size'],shuffle=True)
 for epoch in range(a.epochs or cfg['stage2']['epochs']):
  total=0
  for zs in loader:
   zs=[z.to(dev) for z in zs];loss,_=pri.compute_loss(zs);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(pri.parameters(),1.);opt.step();total+=loss.item()
  append_csv(out/'stage2_metrics.csv',{'epoch':epoch+1,'loss':total});print(f'epoch={epoch+1} loss={total:.5f}')
 save_checkpoint(out/'stage2.pt',priors=pri.state_dict(),config=cfg)
if __name__=='__main__':main()
