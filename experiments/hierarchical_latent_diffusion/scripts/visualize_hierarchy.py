#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,torch
from src.datasets import image_loader
from src.batches import batch_images
from src.pipeline import load_models,resample_tail
from src.utils import *
from src.visualization import save_image_grid
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--steps',type=int,default=50);a=p.parse_args();cfg=load_config(a.config);dev=get_device(cfg['device']);out=output_dir(cfg);dino,vae,enc,dec,pri=load_models(cfg,dev);batch=next(iter(image_loader(cfg,split='test',shuffle=False,max_images=1)));x=batch_images(batch,dev);zs=enc(dino(x));imgs=[x,vae.decode(dec(zs))]+[vae.decode(dec(resample_tail(zs,pri,k,a.steps))) for k in range(cfg['K']+1)];save_image_grid(torch.cat(imgs),out/'hierarchy.png',len(imgs))
if __name__=='__main__':main()
