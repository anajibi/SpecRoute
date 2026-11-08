"""Lightning DataModule for the frozen CheXpert v2 row set.

Three things here are protocol, not preference, and each is enforced rather than trusted:

NO HORIZONTAL FLIP. A chest radiograph is not left-right symmetric. The heart sits left of
midline; mirroring teaches the model that situs inversus is ordinary. The CelebA-HQ arm sets
flip_aug: true and that is correct for faces; here the flag does not exist, so it cannot be
switched on by copying a config.

HELD-OUT IS NEVER TRAINED ON. The evaluation cohort's 512 sources are drawn from split == 1.
This module takes its training rows from split == 0 only, and `test_dataloader` exists purely
so the split is addressable -- the trainer never calls it. This is the defect the CelebA-HQ arm
has (build_cohorts.py draws from all 30,000 rows, ~80% of them trained on), and it is not being
reproduced.

MODELED FINDINGS MUST BE CERTAIN. Protocol section 3 step 3 drops rows whose modeled finding is
uncertain. The frozen row set was filtered under v0, when edema was eval-only and its -1 was
folded to 0, and images_224.npy is aligned to that row set by index and cannot be rebuilt. So
the policy is applied here instead: rows flagged uncertain in label_uncertainty.csv for any
attribute in `require_certain` are dropped from training. Without this the generator would be
taught to condition on a label that is really "the report did not commit".

RESOLUTION. The memmap is 224 (resize short side 224 + 224 centre crop). Training runs at 256
because the ffhq256_autoenc architecture is the one the pretrained checkpoint corresponds to:
at img_size 224 the UNet grows an extra decoder ResBlock (output_blocks.5.1, 12 tensors, 3.42%
of parameters) that the checkpoint has no weights for, so a whole block mid-decoder would start
from random. Measured, untouched FFHQ weights on held-out chest radiographs:
    img_size 224  96.6% of weights load   38.08 dB PSNR   0.0098 MAE
    img_size 256  100%  of weights load   38.89 dB PSNR   0.0086 MAE
both better than the 36.35 dB / 0.0224 that justified pretrained init for CelebA-HQ. Upsampling
224 -> 256 costs one bicubic resample; a randomly initialised decoder block costs more.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import pytorch_lightning as pl
    _Base = pl.LightningDataModule
except ImportError:                                    # pragma: no cover
    _Base = object

ATTRS = ["sex", "age_bin", "effusion", "edema"]


class CheXpertFrozen(Dataset):
    def __init__(self, npy, attrs, rows, image_size):
        self.npy, self.rows, self.image_size = npy, np.asarray(rows), int(image_size)
        self.attrs = attrs.astype(np.float32)
        self.arr = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.arr is None:                           # memmap opened per worker, after fork
            self.arr = np.load(self.npy, mmap_mode="r")
        r = int(self.rows[i])
        x = torch.from_numpy(np.asarray(self.arr[r], dtype=np.float32) / 127.5 - 1.0)[None]
        if self.image_size != x.shape[-1]:
            x = F.interpolate(x[None], size=(self.image_size,) * 2, mode="bicubic",
                              align_corners=False)[0].clamp_(-1, 1)
        # DiffAE is a 3-channel model; a radiograph is one channel repeated, which is exactly
        # what the frozen attribute predictors were trained on too.
        return {"img": x.expand(3, -1, -1).contiguous(),
                "index": r,
                "attr": torch.from_numpy(self.attrs[r])}


class CheXpertDataModule(_Base):
    def __init__(self, data_dir, image_size=256, batch_size=4, num_workers=8,
                 n_val=2048, require_certain=("edema",), seed=20261107):
        super().__init__()
        self.data_dir, self.image_size = data_dir, int(image_size)
        self.batch_size, self.num_workers = int(batch_size), int(num_workers)
        self.n_val, self.seed = int(n_val), int(seed)
        self.require_certain = tuple(require_certain or ())
        self.attribute_names = list(ATTRS)
        self.provenance = {}

    def setup(self, stage=None):
        D = self.data_dir
        meta = pd.read_csv(f"{D}/chexpert_meta.csv")
        split = np.load(f"{D}/split.npy")
        if len(split) != len(meta):
            raise ValueError(f"split.npy has {len(split)} rows, meta has {len(meta)}")
        attrs = meta[ATTRS].to_numpy(np.float32)

        train_pool = np.flatnonzero(split == 0)
        heldout = np.flatnonzero(split == 1)

        # certainty policy
        n_unc = 0
        if self.require_certain:
            unc = pd.read_csv(f"{D}/label_uncertainty.csv")
            if not (unc["path"].values == meta["path"].values).all():
                raise ValueError("label_uncertainty.csv is not row-aligned with chexpert_meta.csv")
            keep = np.ones(len(meta), bool)
            for v in self.require_certain:
                col = f"uncertain_{v}"
                if col not in unc.columns:
                    raise ValueError(f"{col} missing from label_uncertainty.csv")
                keep &= unc[col].to_numpy() == 0
            n_unc = int((~keep[train_pool]).sum())
            train_pool = train_pool[keep[train_pool]]

        # validation rows carved out BY PATIENT, so a val image never shares a patient with a
        # training image. A random row split would leak: CheXpert has ~2.8 studies per patient.
        pats = np.sort(meta.loc[train_pool, "patient_id"].unique())
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(pats)
        val_pat, taken = set(), 0
        counts = meta.loc[train_pool].groupby("patient_id").size()
        for p in perm:
            if taken >= self.n_val:
                break
            val_pat.add(p); taken += int(counts.get(p, 0))
        is_val = meta["patient_id"].isin(val_pat).to_numpy()
        val_rows = train_pool[is_val[train_pool]]
        fit_rows = train_pool[~is_val[train_pool]]
        if set(meta.loc[fit_rows, "patient_id"]) & set(meta.loc[val_rows, "patient_id"]):
            raise AssertionError("val and fit share a patient")
        if np.intersect1d(fit_rows, heldout).size or np.intersect1d(val_rows, heldout).size:
            raise AssertionError("a training or validation row is in the held-out split")

        self.train_set = CheXpertFrozen(f"{D}/images_224.npy", attrs, fit_rows, self.image_size)
        self.val_set = CheXpertFrozen(f"{D}/images_224.npy", attrs, val_rows, self.image_size)
        self.test_set = CheXpertFrozen(f"{D}/images_224.npy", attrs, heldout, self.image_size)
        self.provenance = {
            "n_fit": int(len(fit_rows)), "n_val": int(len(val_rows)),
            "n_heldout_never_trained": int(len(heldout)),
            "n_dropped_uncertain": n_unc, "require_certain": list(self.require_certain),
            "image_size": self.image_size, "flip_aug": False,
            "meta_sha256": hashlib.sha256(open(f"{D}/chexpert_meta.csv", "rb").read()).hexdigest(),
            "split_sha256": hashlib.sha256(open(f"{D}/split.npy", "rb").read()).hexdigest(),
            "attrs": list(ATTRS),
        }
        print(f"[chexpert] fit {len(fit_rows):,}  val {len(val_rows):,}  "
              f"heldout {len(heldout):,} (never trained)  "
              f"dropped uncertain {n_unc:,}  size {self.image_size}  no flip")

    def attr_index(self, name):
        return self.attribute_names.index(name)

    def _loader(self, ds, shuffle=False, drop_last=False):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_set, True, True)

    def val_dataloader(self):
        return self._loader(self.val_set)

    def test_dataloader(self):
        return self._loader(self.test_set)
