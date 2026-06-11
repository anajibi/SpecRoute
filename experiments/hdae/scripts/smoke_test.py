#!/usr/bin/env python
"""Tiny end-to-end preprocessing/model/DDIM smoke run; uses first 100 configured images."""
import argparse,copy,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from torchvision.utils import save_image
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.data.preprocess import preprocess
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_hdae_config(a.config,require_data=False);d=cfg.raw['data']
with tempfile.TemporaryDirectory() as td:
    lmdb=str(Path(td)/'tiny.lmdb');attrs=str(Path(td)/'attrs.npz')
    preprocess(d['image_dir'],lmdb,d['attr_path'],d['partition_path'],attrs,64,d['resize_filter'],1,100)
    from experiments.hdae.data.celeba_hq import CelebAHQPacked
    ds=CelebAHQPacked(lmdb,attrs); x=torch.stack([ds[i]['img'] for i in range(2)])
    model=cfg.train_conf.model_conf.make_model(); opt=torch.optim.Adam(model.parameters(),lr=1e-4); model.train()
    for _ in range(2):
        out=model(x,torch.zeros(2,dtype=torch.long),x_start=x).pred;loss=out.square().mean();opt.zero_grad();loss.backward();opt.step()
    assert torch.isfinite(loss);save_image(x.add(1).div(2),Path(cfg.raw['output_dir'])/'smoke_grid.png')
    print('smoke loss',float(loss))
