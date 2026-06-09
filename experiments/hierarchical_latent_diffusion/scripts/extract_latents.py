#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,torch
from src.datasets import dataset_attr_names,image_loader
from src.batches import batch_image_ids,batch_images
from src.pipeline import load_models
from src.utils import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--max-images',type=int);p.add_argument('--split',default='train');a=p.parse_args();cfg=load_config(a.config);dev=get_device(cfg['device']);out=output_dir(cfg);dino,_,enc,_,_=load_models(cfg,dev,need_priors=False);acc=[[] for _ in range(cfg['K'])];ids=[];attrs=[]
 with torch.no_grad():
  loader=image_loader(cfg,split=a.split,shuffle=False,max_images=a.max_images)
  for batch in loader:
   for bucket,z in zip(acc,enc(dino(batch_images(batch,dev)))):bucket.append(z.cpu())
   ids.extend(batch_image_ids(batch))
   if 'attributes' in batch:attrs.append(batch['attributes'].cpu())
 payload={'latents':[torch.cat(x) for x in acc],'ids':ids,'config':cfg,'attr_names':dataset_attr_names(loader.dataset)}
 if attrs:payload['attributes']=torch.cat(attrs)
 torch.save(payload,out/'latents.pt');print(f'saved {len(ids)} examples')
if __name__=='__main__':main()
