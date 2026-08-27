#!/usr/bin/env python
import argparse
import glob
import re
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

    # Resume from the NEWEST checkpoint by global_step, not from a hardcoded last.ckpt.
    #
    # Why: if the output dir is seeded with a checkpoint to continue from (as the k=1 run
    # was), Lightning finds `last.ckpt` already taken and writes its own saves to
    # `last-v1.ckpt` instead. A restart that trusted `last.ckpt` would then silently rewind
    # to the seed -- measured here at 50,000 vs the live 86,000, i.e. ~5 hours discarded --
    # and nothing would look wrong in the logs.
    def _step_of(path):
        m = re.search(r'step=(\d+)', os.path.basename(path))
        if m:
            return int(m.group(1))
        try:
            return int(torch.load(path, map_location='cpu').get('global_step', -1))
        except Exception:
            return -1

    ckpt_dir = out / 'checkpoints'
    cands = sorted(glob.glob(str(ckpt_dir / '*.ckpt')))
    resume = None
    if cands:
        scored = [(_step_of(c), c) for c in cands]
        scored.sort()
        best_step, resume = scored[-1]
        print(f'resuming from {os.path.basename(resume)} (global_step={best_step})')
        for st, c in scored[:-1]:
            print(f'  (also present: {os.path.basename(c)} @ {st})')
    else:
        print('no checkpoint found -- training from scratch')

    # Checkpoint cadence is config-driven: save_top_k=-1 keeps every snapshot, so
    # `checkpoint_every_n_steps` alone decides how many land on disk over the run.
    ckpt_every = int(t.get('checkpoint_every_n_steps', 1000))
    keep = int(t.get('save_top_k', 1))
    callbacks = [
        ModelCheckpoint(
            dirpath=str(out / 'checkpoints'),
            save_last=True,
            save_top_k=keep,
            every_n_train_steps=ckpt_every,
            save_on_train_epoch_end=False  # STRICTLY disable the default epoch-end save
        ),
        LearningRateMonitor('step')
    ]
    n_img_ep = int(t.get('log_images_every_n_epochs', 0))
    if n_img_ep > 0:
        from experiments.hdae.hdae.image_logger import ImageLogCallback
        scm = d.get('scm_checkpoint') or 'experiments/hdae/outputs/scm/causal3dident_scm_spline.pt'
        callbacks.append(ImageLogCallback(
            h5_path=d.get('test_h5_path') or d['h5_path'],
            every_n_epochs=n_img_ep,
            n_images=int(t.get('log_images_n', 4)),
            T=int(t.get('T_eval', 100)),
            guidance=float(cfg.raw['conditioning'].get('cfg_guidance_scale', 3.0)),
            scm_path=scm,
            seed=int(cfg.raw.get('seed', 0)),
        ))
        print(f'image logging: every {n_img_ep} epochs, {t.get("log_images_n", 4)} images, '
              f'guidance {cfg.raw["conditioning"].get("cfg_guidance_scale", 3.0)}')

    trainer = pl.Trainer(
        **cfg.lightning_kwargs(),
        resume_from_checkpoint=resume,
        callbacks=callbacks,
        max_epochs=300,
        logger=TensorBoardLogger(str(out), name='logs')
    )

    lit = HDAELitModule(cfg.train_conf)
    if t.get('compile'):
        # train.compile was a dead config key until now -- nothing read it, so setting it
        # true silently did nothing. See HDAELitModule.enable_compile for why only a side
        # handle is compiled and self.model is left alone.
        lit.enable_compile(t['compile'] if isinstance(t['compile'], str) else 'default')

    # This is the line that was causing the recursive spawning loop
    trainer.fit(lit, datamodule=dm)


# --- Strict Multiprocessing Guard ---
if __name__ == '__main__':
    from multiprocessing import freeze_support

    # Essential for cross-platform multiprocessing stability
    freeze_support()

    main()
