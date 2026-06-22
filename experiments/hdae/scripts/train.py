#!/usr/bin/env python
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
import torch, pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.data.datamodule import CelebAHQDataModule
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args(); cfg=load_hdae_config(a.config)
torch.set_float32_matmul_precision('high'); pl.seed_everything(cfg.raw['seed'])
d=cfg.raw['data']; t=cfg.raw['train']; dm=CelebAHQDataModule(d['lmdb_path'],d['attr_npz'],t['batch_size_per_gpu'],t['num_workers'],d['flip_aug'])
out=Path(cfg.raw['output_dir']); out.mkdir(parents=True,exist_ok=True)
callbacks=[ModelCheckpoint(dirpath=str(out/'checkpoints'),save_last=True,save_top_k=-1),LearningRateMonitor('step')]
trainer=pl.Trainer(**cfg.lightning_kwargs(),callbacks=callbacks,logger=TensorBoardLogger(str(out),name='logs'))
trainer.fit(HDAELitModule(cfg.train_conf),datamodule=dm)
