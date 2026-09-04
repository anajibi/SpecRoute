"""Lightning DataModule for the packed MorphoMNIST++ dataset (HDAE training)."""
import numpy as np
from torch.utils.data import DataLoader, Subset
try:
    import pytorch_lightning as pl
    _Base = pl.LightningDataModule
except ImportError:
    _Base = object
from .morphomnist import MorphoMNISTPacked


class MorphoMNISTDataModule(_Base):
    """2-partition packed dataset (0=train, 1=test) -- unlike CelebAHQDataModule's 3-way partition
    split, morphomnist.h5 has no dedicated val partition, so val is carved out of partition 0.
    No flip augmentation (unlike CelebA-HQ): flipping a digit image is not a label-preserving
    transform (e.g. a mirrored "2" is not a valid digit), so it's omitted rather than ported over.
    """

    def __init__(self, h5_path, batch_size=32, num_workers=0, val_frac=0.02, preload_images=True,
                 seed=0, train_frac=1.0):
        super().__init__()
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac
        self.preload_images = preload_images
        self.seed = seed
        # train_frac < 1 subsamples the TRAIN split only, for data-scaling experiments. The
        # subsample is STRATIFIED BY DIGIT and seeded, so a 12.5% run keeps all ten classes in
        # their original proportions rather than losing a class to an unlucky draw. Val and test
        # are never subsampled -- every fraction is scored on exactly the same held-out data,
        # which is the whole point of the comparison.
        self.train_frac = float(train_frac)

    def setup(self, stage=None):
        ds = MorphoMNISTPacked(self.h5_path, preload_images=self.preload_images)
        self.attribute_names = ds.attribute_names
        parts = ds.partitions
        train_all = np.nonzero(parts == 0)[0]
        rng = np.random.RandomState(self.seed)
        rng.shuffle(train_all)
        n_val = max(1, int(len(train_all) * self.val_frac))
        val_idx, train_idx = train_all[:n_val], train_all[n_val:]
        if self.train_frac < 1.0:
            digits = ds.attrs[train_idx, ds.attribute_names.index("digit")].astype(int)
            keep = []
            for d in np.unique(digits):
                cls = train_idx[digits == d]
                n_keep = max(1, int(round(len(cls) * self.train_frac)))
                keep.append(np.random.RandomState(self.seed + 1000 + int(d)).permutation(cls)[:n_keep])
            train_idx = np.sort(np.concatenate(keep))
        test_idx = np.nonzero(parts == 1)[0]
        self.n_train = len(train_idx)
        self.train_set = Subset(ds, train_idx.tolist())
        self.val_set = Subset(ds, val_idx.tolist())
        self.test_set = Subset(ds, test_idx.tolist())

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
