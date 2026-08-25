#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path
import warnings

# Filter out the specific pkg_resources deprecation warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*pkg_resources is deprecated as an API.*"
)
# --- Imports and Path Setup ---
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # Note: Consider replacing this with an editable install

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.data.datamodule import CelebAHQDataModule
from experiments.hdae.data.morpho_datamodule import MorphoMNISTDataModule
from experiments.hdae.data.causal3dident_datamodule import Causal3DIdentDataModule


# --- Execution Logic Encapsulated ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    a = p.parse_args()

    cfg = load_hdae_config(a.config)

    torch.set_float32_matmul_precision('high')
    pl.seed_everything(cfg.raw['seed'])

    d = cfg.raw['data']
    t = cfg.raw['train']

    if d.get('type') == 'causal3dident':
        dm = Causal3DIdentDataModule(
            d['h5_path'],
            test_h5_path=d.get('test_h5_path'),
            batch_size=t['batch_size_per_gpu'],
            num_workers=t['num_workers'],
            val_frac=d.get('val_frac', 0.02),
            preload_images=d.get('preload_images', False),
            seed=cfg.raw.get('seed', 0),
        )
    elif d.get('type') == 'morphomnist':
        dm = MorphoMNISTDataModule(
            d['h5_path'],
            t['batch_size_per_gpu'],
            t['num_workers'],
            val_frac=d.get('val_frac', 0.02),
            preload_images=d.get('preload_images', True),
        )
    else:
        dm = CelebAHQDataModule(
            d['lmdb_path'],
            d['attr_npz'],
            t['batch_size_per_gpu'],
            t['num_workers'],
            d['flip_aug']
        )

    out = Path(cfg.raw['output_dir'])
    out.mkdir(parents=True, exist_ok=True)

    checkpoint_path = f'{out}/checkpoints/last.ckpt'
    print('ckpt path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
        print('resume!')
    else:
        resume = None

    callbacks = [
        ModelCheckpoint(
            dirpath=str(out / 'checkpoints'),
            save_last=True,
            save_top_k=1,  # Keep only the 1 most recent step-checkpoint
            every_n_train_steps=1000,  # Save exactly every 1000 batches
            save_on_train_epoch_end=False  # STRICTLY disable the default epoch-end save
        ),
        LearningRateMonitor('step')
    ]

    trainer = pl.Trainer(
        **cfg.lightning_kwargs(),
        resume_from_checkpoint=resume,
        callbacks=callbacks,
        max_epochs=300,
        logger=TensorBoardLogger(str(out), name='logs')
    )

    # This is the line that was causing the recursive spawning loop
    trainer.fit(HDAELitModule(cfg.train_conf), datamodule=dm)


# --- Strict Multiprocessing Guard ---
if __name__ == '__main__':
    from multiprocessing import freeze_support

    # Essential for cross-platform multiprocessing stability
    freeze_support()

    main()
