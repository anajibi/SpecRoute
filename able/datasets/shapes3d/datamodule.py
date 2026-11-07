"""Lightning DataModule for the frozen 3DShapes Graph A layout.

Training reads `train_image_idx.npy` (16,000,000 int32 grid indices) ONE PASS, in the order the
frozen shuffle fixed -- so `max_steps` must equal len/batch exactly and the loader never
reshuffles. `shuffle=False` is therefore load-bearing, not an oversight: the protocol's single
global permutation IS the shuffle, and every method must see the identical stream.

Validation uses the 48,000 HELD-OUT cells, never training rows, so monitoring consumes none of
the 16M budget and the training epoch stays exactly one pass. Held-out factor tuples are decoded
straight from the grid index, so no second labels lookup is needed.
"""
import logging

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import pytorch_lightning as pl
    _Base = pl.LightningDataModule
except ImportError:
    _Base = object

from .scm_exact import LAYOUT_COLUMNS, N_GRID, decode_index


def _to_img(arr):
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1)


class _Images:
    """The 480,000-image grid. Preloaded (5.9 GB) is shared with fork()ed workers copy-on-write;
    every access is a read, so no worker ever duplicates it."""

    def __init__(self, h5_path, preload=True):
        self.h5_path, self._file, self._arr = h5_path, None, None
        if preload:
            with h5py.File(h5_path, "r") as f:
                self._arr = f["images"][:]
            logging.info("3dshapes: preloaded %s images (%.1f GB)",
                         f"{len(self._arr):,}", self._arr.nbytes / 1e9)

    def __getitem__(self, g):
        if self._arr is not None:
            return self._arr[g]
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file["images"][g]

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_file"] = None
        return s


class Shapes3DLayout(Dataset):
    """One training position per row of the frozen layout."""

    def __init__(self, images, image_idx, layout):
        assert len(image_idx) == len(layout), (len(image_idx), len(layout))
        self.images, self.image_idx, self.layout = images, image_idx, layout

    def __len__(self):
        return len(self.image_idx)

    def __getitem__(self, i):
        g = int(self.image_idx[i])
        return {"img": _to_img(self.images[g]),
                "attr": torch.from_numpy(self.layout[i].astype(np.float32)),
                "index": i, "grid_index": g, "partition": 0}


class Shapes3DHeldout(Dataset):
    """The held-out cells, addressed by grid index; factors decoded from the index."""

    def __init__(self, images, cells):
        self.images, self.cells = images, np.asarray(cells, dtype=np.int64)
        Hf, Hw, Ho, C, S, R = decode_index(self.cells)
        # reorder into LAYOUT_COLUMNS: shape, object_hue, scale, floor_hue, wall_hue, orientation
        self.attrs = np.stack([S, Ho, C, Hf, Hw, R], axis=1).astype(np.float32)

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, i):
        g = int(self.cells[i])
        return {"img": _to_img(self.images[g]), "attr": torch.from_numpy(self.attrs[i]),
                "index": i, "grid_index": g, "partition": 2}


class Shapes3DDataModule(_Base):
    def __init__(self, h5_path, layout_dir, batch_size=128, num_workers=4,
                 preload_images=True, n_val=2048, seed=0, train_start=0):
        super().__init__()
        # train_start RESUMES THE FROZEN STREAM, it does not reshuffle it. Lightning 1.9.5
        # without PL_FAULT_TOLERANT_TRAINING restores `batch_progress` but NOT the train
        # sampler position, so a plain mid-epoch resume silently rewinds the loader to row 0:
        # the counter reads 125,000 at the end while the tail of the stream was never seen.
        # Setting train_start to (steps_done * batch) drops exactly the rows already consumed,
        # so the run sees all 16,000,000 draws once, in the frozen order. It must be an exact
        # multiple of the batch size or the batch boundaries shift and the stream is no longer
        # the one the other arms saw.
        self.train_start = int(train_start)
        self.h5_path, self.layout_dir = h5_path, str(layout_dir).rstrip("/")
        self.batch_size, self.num_workers = batch_size, num_workers
        self.preload_images, self.n_val, self.seed = preload_images, int(n_val), int(seed)
        self.attribute_names = list(LAYOUT_COLUMNS)

    def setup(self, stage=None):
        d = self.layout_dir
        image_idx = np.load(f"{d}/train_image_idx.npy")
        layout = np.load(f"{d}/train_layout.npy")
        heldout = np.load(f"{d}/heldout_cells.npy")
        if image_idx.max() >= N_GRID or image_idx.min() < 0:
            raise ValueError("train_image_idx out of grid range")
        held_mask = np.zeros(N_GRID, dtype=bool)
        held_mask[heldout] = True
        if held_mask[image_idx[:1_000_000]].any():
            raise ValueError("training rows fall in held-out cells -- wrong layout_dir?")
        if self.train_start:
            if self.train_start % self.batch_size:
                raise ValueError(
                    f"train_start={self.train_start} is not a multiple of batch_size="
                    f"{self.batch_size}; batch boundaries would shift and the stream would "
                    "stop matching the one the other arms saw.")
            if self.train_start >= len(image_idx):
                raise ValueError(
                    f"train_start={self.train_start} is beyond the frozen stream "
                    f"({len(image_idx)} rows) -- nothing left to train on.")
            image_idx = image_idx[self.train_start:]
            layout = layout[self.train_start:]
            logging.info("3dshapes: RESUMING frozen stream at row %s -- %s rows (%s steps) left",
                         f"{self.train_start:,}", f"{len(image_idx):,}",
                         f"{len(image_idx) // self.batch_size:,}")
        images = _Images(self.h5_path, self.preload_images)
        self.train_set = Shapes3DLayout(images, image_idx, layout)
        val_cells = np.random.RandomState(self.seed).choice(
            heldout, size=min(self.n_val, len(heldout)), replace=False)
        self.val_set = self.test_set = Shapes3DHeldout(images, np.sort(val_cells))
        logging.info("3dshapes: %s train rows (one pass = %s steps at batch %d), %d val cells",
                     f"{len(self.train_set):,}",
                     f"{len(self.train_set) // self.batch_size:,}", self.batch_size,
                     len(self.val_set))

    def attr_index(self, name):
        return self.attribute_names.index(name)

    def _loader(self, ds, shuffle=False, drop_last=False):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        # shuffle=False ON PURPOSE -- the frozen global permutation is the shuffle.
        return self._loader(self.train_set, shuffle=False, drop_last=True)

    def val_dataloader(self):
        return self._loader(self.val_set)

    def test_dataloader(self):
        return self._loader(self.test_set)
