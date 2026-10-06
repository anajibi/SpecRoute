"""Fork/DDP-safe dataset over pre-resized raw RGB LMDB records."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class CelebAHQPacked(Dataset):
    def __init__(self, lmdb_path, attr_npz=None, flip=False):
        self.lmdb_path = str(lmdb_path); self.flip = flip; self._env = None
        self.meta = json.loads((Path(lmdb_path) / "meta.json").read_text())
        arrays = np.load(attr_npz or self.meta["attr_npz"])
        self.attrs = arrays["attrs"]; self.partitions = arrays["partitions"]
        self.attribute_names = [str(x) for x in arrays["attribute_names"]]
        if len(self.attrs) != self.meta["num_images"]:
            raise ValueError("attribute array and LMDB length differ")

    def _open(self):
        if self._env is None:
            import lmdb
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                                  readahead=False, meminit=False, subdir=True)
        return self._env

    def __len__(self): return self.meta["num_images"]

    def __getitem__(self, index):
        with self._open().begin() as txn:
            raw = txn.get(f"{index:08d}".encode())
        s = self.meta["image_size"]
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(s, s, 3).copy()
        img = torch.from_numpy(arr).permute(2, 0, 1).float().div_(127.5).sub_(1)
        if self.flip and torch.rand(()) < .5: img = img.flip(-1)
        return {"img": img, "index": index, "attr": torch.from_numpy(self.attrs[index].copy()),
                "partition": int(self.partitions[index])}

    def __getstate__(self):
        state = self.__dict__.copy(); state["_env"] = None; return state
