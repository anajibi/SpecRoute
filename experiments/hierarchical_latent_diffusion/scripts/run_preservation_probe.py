#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,torch
from src.datasets import image_loader
from src.metrics import cosine_similarity,mse_distance
from src.pipeline import load_models,resample_tail
from src.utils import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--max-images',type=int,default=10);p.add_argument('--steps',type=int,default=50);a=p.parse_args();cfg=load_config(a.config);dev=get_device(cfg['device']);out=output_dir(cfg);dino,vae,enc,dec,pri=load_models(cfg,dev)
 with torch.no_grad():
  for x,ids in image_loader(cfg,split='test',shuffle=False,max_images=a.max_images):
   x=x.to(dev);zs=enc(dino(x));base=vae.decode(dec(zs));base_f=dino(base)
   for kept in range(cfg['K']+1):
    mod=resample_tail(zs,pri,kept,a.steps) if kept<cfg['K'] else zs;img=vae.decode(dec(mod));sim=cosine_similarity(base_f,dino(img));mse=mse_distance(base,img)
    for i,name in enumerate(ids):append_csv(out/'preservation.csv',{'image_id':name,'K':cfg['K'],'levels_kept':kept,'dinov2_sim':sim[i].item(),'image_mse':mse[i].item()})
if __name__=='__main__':main()
