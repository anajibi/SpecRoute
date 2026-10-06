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
            seed=cfg.raw.get('seed', 0),
            train_frac=d.get('train_frac', 1.0),
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

    # An explicit `resume_from` seeds a NEW output dir from an existing checkpoint as a TRUE
    # Lightning resume: weights + Adam moments + GradScaler state + global_step, not just
    # weights. This exists because `init_from` is weights-only, and restarting Adam cold on
    # already-converged weights is what destroyed celebahq256_k11_cd015 -- grad norms crept
    # 0.006 -> 0.042 over ~400 steps and then went non-finite, with the loss flat at 0.012
    # the whole way. Only used when the output dir has no checkpoint of its own to resume.
    if resume is None and t.get('resume_from'):
        rf = t['resume_from']
        rf = str(ROOT / rf) if not os.path.isabs(rf) else rf
        if not os.path.exists(rf):
            raise FileNotFoundError(f'resume_from not found: {rf}')
        resume = rf
        print(f'RESUME_FROM: {rf} (full state: weights + optimizer + scaler)')

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
    from experiments.hdae.hdae.nan_tracer import maybe_tracer
    _tr = maybe_tracer()
    if _tr is not None:
        callbacks.append(_tr)
        print('[nan_tracer] ACTIVE ->', _tr.path, 'stop_after=', _tr.stop_after)

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

    # Initialise from a vanilla DiffAE checkpoint when the config asks for it, but ONLY on a
    # fresh run: `resume` already carries the full model, and overwriting it with pretrained
    # weights would silently rewind every step of progress.
    init_from = t.get('init_from')
    if init_from and not resume:
        from experiments.hdae.hdae.pretrained_init import load_pretrained_into
        _p = str(ROOT / init_from) if not Path(init_from).is_absolute() else init_from
        info = load_pretrained_into(lit.model, _p)
        # EMA handling differs between the two kinds of init_from:
        #
        #   vanilla DiffAE checkpoint -> its ema_model IS the weights we just loaded, so copying
        #     the model into EMA is correct and avoids sampling from a half-random network for
        #     the first ~1/(1-decay) steps.
        #   CONTINUATION from one of our own runs -> the checkpoint carries a real EMA that is
        #     better than the raw weights and represents thousands of steps of averaging.
        #     Overwriting it with the model would discard that and make every early checkpoint
        #     look like a regression for no reason. Load it instead.
        import torch as _t
        _raw = _t.load(_p, map_location="cpu")
        _sd = _raw.get("state_dict", _raw)
        _ema = {k[len("ema_model."):]: v for k, v in _sd.items() if k.startswith("ema_model.")}
        _tgt = lit.ema_model.state_dict()
        if _ema and set(_tgt).issubset(_ema):
            lit.ema_model.load_state_dict({k: _ema[k] for k in _tgt})
            print(f"[init] CONTINUATION: EMA carried over from {Path(_p).name} "
                  f"({info['fraction']*100:.1f}% of model weights loaded)")
        else:
            lit.ema_model.load_state_dict(lit.model.state_dict())
            print(f"[init] EMA seeded from the same weights ({info['fraction']*100:.1f}% pretrained)")
        del _raw, _sd, _ema
    elif init_from and resume:
        print(f"[init] ignoring init_from -- resuming from {resume}")
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
