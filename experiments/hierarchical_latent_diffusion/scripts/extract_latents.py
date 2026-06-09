#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,torch
from src.datasets import image_loader
from src.pipeline import load_models
from src.utils import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--max-images',type=int);a=p.parse_args();cfg=load_config(a.config);dev=get_device(cfg['device']);out=output_dir(cfg);dino,_,enc,_,_=load_models(cfg,dev,need_priors=False);acc=[[] for _ in range(cfg['K'])];ids=[]
 with torch.no_grad():
  for x,names in image_loader(cfg,shuffle=False,max_images=a.max_images):
   for bucket,z in zip(acc,enc(dino(x.to(dev)))):bucket.append(z.cpu())
   ids.extend(names)
 torch.save({'latents':[torch.cat(x) for x in acc],'ids':ids,'config':cfg},out/'latents.pt');print(f'saved {len(ids)} examples')
if __name__=='__main__':main()
