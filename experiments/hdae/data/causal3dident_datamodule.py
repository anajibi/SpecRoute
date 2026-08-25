"""Lightning DataModule for the packed Causal3DIdent dataset (HDAE training).

Validation is carved out of the trainset (val_frac), leaving the shipped 25,200-image
testset completely untouched for later evaluation. No flip augmentation: a mirrored scene
has a different object x-position and rotation, so flipping is not label-preserving here
(same reasoning that rules it out for MorphoMNIST digits).
"""
import numpy as np
from torch.utils.data import DataLoader, Subset

try:
    import pytorch_lightning as pl
    _Base = pl.LightningDataModule
except ImportError:
    _Base = object

from .causal3dident import Causal3DIdentPacked


class Causal3DIdentDataModule(_Base):
    def __init__(self, h5_path, test_h5_path=None, batch_size=32, num_workers=0,
                 val_frac=0.02, preload_images=False, seed=0):
        super().__init__()
        self.h5_path = h5_path
        self.test_h5_path = test_h5_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac
        self.preload_images = preload_images
        self.seed = seed

    def setup(self, stage=None):
        ds = Causal3DIdentPacked(self.h5_path, preload_images=self.preload_images)
        self.attribute_names = ds.attribute_names
        idx = np.arange(len(ds))
        rng = np.random.RandomState(self.seed)
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * self.val_frac))
        self.val_set = Subset(ds, idx[:n_val].tolist())
        self.train_set = Subset(ds, idx[n_val:].tolist())
        if self.test_h5_path:
            test_ds = Causal3DIdentPacked(self.test_h5_path, preload_images=False)
            self.test_set = Subset(test_ds, list(range(len(test_ds))))
        else:
            self.test_set = self.val_set

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
