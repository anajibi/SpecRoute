#!/usr/bin/env python
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.data.preprocess import preprocess
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--num-workers',type=int); p.add_argument('--limit',type=int); a=p.parse_args()
cfg=load_hdae_config(a.config,require_data=False); d=cfg.raw['data']
meta=preprocess(d['image_dir'],d['lmdb_path'],d['attr_path'],d['partition_path'],d['attr_npz'],d['image_size'],d['resize_filter'],a.num_workers or cfg.raw['train']['num_workers'],a.limit)
print(meta)
