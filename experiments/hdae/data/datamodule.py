"""Lightning DataModule for the packed CelebA-HQ dataset."""
import logging
import numpy as np
from torch.utils.data import DataLoader, Subset
try:
    import pytorch_lightning as pl
    _Base = pl.LightningDataModule
except ImportError:
    _Base = object
from .celeba_hq import CelebAHQPacked


class CelebAHQDataModule(_Base):
    def __init__(self, lmdb_path, attr_npz, batch_size=32, num_workers=8, flip_aug=True):
        super().__init__(); self.lmdb_path=lmdb_path; self.attr_npz=attr_npz
        self.batch_size=batch_size; self.num_workers=num_workers; self.flip_aug=flip_aug

    def setup(self, stage=None):
        train_ds = CelebAHQPacked(self.lmdb_path, self.attr_npz, self.flip_aug)
        eval_ds = CelebAHQPacked(self.lmdb_path, self.attr_npz, False)
        self.attribute_names = train_ds.attribute_names
        parts = train_ds.partitions
        indices = [np.where(parts == p)[0].tolist() for p in range(3)]
        if any(not x for x in indices):
            logging.warning("partition labels incomplete; using deterministic 90/5/5 index split")
            n=len(train_ds); a=max(1,int(n*.9)); b=max(a+1,int(n*.95)); indices=[list(range(a)),list(range(a,b)),list(range(b,n))]
        self.train_set, self.val_set, self.test_set = Subset(train_ds, indices[0]), Subset(eval_ds, indices[1]), Subset(eval_ds, indices[2])

    def attr_index(self, name): return self.attribute_names.index(name)
    def _loader(self, ds, shuffle=False, drop_last=False):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0)
    def train_dataloader(self): return self._loader(self.train_set, True, True)
    def val_dataloader(self): return self._loader(self.val_set)
    def test_dataloader(self): return self._loader(self.test_set)
