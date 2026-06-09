"""Configuration, reproducibility, checkpoint, and model helpers."""
from __future__ import annotations
import csv, json, random
from pathlib import Path
import numpy as np
import torch, yaml
from .encoders import ChainEncoder
from .decoder import HierarchicalDecoder
from .priors import HierarchicalPriorStack

def load_config(path):
    with open(path) as f: cfg=yaml.safe_load(f)
    if cfg["K"] != len(cfg["encoder"]["level_dims"]): raise ValueError("K must match number of level_dims")
    return cfg
def seed_everything(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def get_device(name): return torch.device(name if name != "cuda" or torch.cuda.is_available() else "cpu")
def output_dir(cfg):
    p=Path(__file__).parents[1]/cfg["output_dir"]; p.mkdir(parents=True,exist_ok=True); return p
def save_checkpoint(path, **objects): Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(objects,path)
def load_checkpoint(path, device="cpu"): return torch.load(path,map_location=device,weights_only=False)
def save_config(cfg,path): Path(path).write_text(yaml.safe_dump(cfg,sort_keys=False))
def append_csv(path,row):
    path=Path(path); new=not path.exists(); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(row));
        if new:w.writeheader()
        w.writerow(row)
def build_trainable(cfg):
    enc=ChainEncoder(**cfg["encoder"]); dec=HierarchicalDecoder(level_dims=cfg["encoder"]["level_dims"],**cfg["decoder"]); pri=HierarchicalPriorStack(cfg["encoder"]["level_dims"],cfg["priors"]); return enc,dec,pri
def freeze(module): module.requires_grad_(False); module.eval(); return module
