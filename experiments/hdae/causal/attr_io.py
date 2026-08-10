"""Load a packed attribute table (`attrs`/`attribute_names`, optionally `partitions`) from
either the CelebA-HQ `.npz` convention or the MorphoMNIST++ `.h5` convention -- same field
names, different container. Dispatches on file extension so callers (`train_scm.py`, the SCM
verification scripts) stay dataset-agnostic without an `if morpho` branch of their own.
"""
from typing import Optional, Tuple

import h5py
import numpy as np


def load_attr_table(path: str) -> Tuple[np.ndarray, list, Optional[np.ndarray]]:
    """Returns (attrs: (N,D) float32, attribute_names: list[str], partitions: (N,) int64 or None)."""
    if path.endswith(".npz"):
        arrays = np.load(path, allow_pickle=True)
        attrs = arrays["attrs"]
        names = [str(x) for x in arrays["attribute_names"]]
        partitions = arrays["partitions"] if "partitions" in arrays else None
        return attrs, names, partitions
    if path.endswith(".h5") or path.endswith(".hdf5"):
        with h5py.File(path, "r") as f:
            attrs = f["attrs"][:]
            names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["attribute_names"][:]]
            partitions = f["partitions"][:] if "partitions" in f else None
        return attrs, names, partitions
    raise ValueError(f"unrecognized attribute table format: {path!r} (expected .npz or .h5/.hdf5)")
