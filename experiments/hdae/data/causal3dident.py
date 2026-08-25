"""Packed Causal3DIdent dataset for HDAE training.

Reads the HDF5 written by scripts/build_causal3dident.py. Mirrors MorphoMNISTPacked's
``{"img", "attr"}`` batch contract so the same LitModule/datamodule machinery applies.

Attribute columns exposed to the model, in this order:

    class        1 col   categorical, 7 object classes (raw value IS the class index)
    pos_spl      1 col   continuous
    pos_obj_0..2 3 cols  continuous (raw latent columns 0,1,2)
    rot_obj_0..2 3 cols  continuous (raw latent columns 3,4,5)
    hue_obj      1 col   continuous, UNMODELLED (not conditioned on)
    hue_spl      1 col   continuous, UNMODELLED
    hue_bg       1 col   continuous, UNMODELLED

The three hues are still returned so they are available as the FC_unobserved pool later;
which columns the model actually conditions on is decided by `conditioning_attrs` in the
config, expanded per-component by lit_module._conditioning_attr_indices.

Every latent is already in [-1, 1] as shipped, so no rescaling happens here.
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# raw_latents column layout (see the dataset's own docs / clevr_dataset.py change_list)
_POS_OBJ = [0, 1, 2]
_ROT_OBJ = [3, 4, 5]
_POS_SPL = 6
_HUE_OBJ, _HUE_SPL, _HUE_BG = 7, 8, 9

ATTRIBUTE_NAMES = (["class", "pos_spl"]
                   + [f"pos_obj_{j}" for j in range(3)]
                   + [f"rot_obj_{j}" for j in range(3)]
                   + ["hue_obj", "hue_spl", "hue_bg"])

CLASS_NAMES = ["teapot", "hare", "dragon", "cow", "armadillo", "horse", "head"]


class Causal3DIdentPacked(Dataset):
    def __init__(self, h5_path, preload_images=False):
        self.h5_path = str(h5_path)
        self.preload_images = bool(preload_images)
        self._h5 = None
        with h5py.File(self.h5_path, "r") as h:
            self.n = int(h["images"].shape[0])
            self.image_size = int(h["images"].shape[1])
            lat = np.asarray(h["latents"][:], dtype=np.float32)
            cls = np.asarray(h["class"][:], dtype=np.float32)
            self._images = np.asarray(h["images"][:]) if self.preload_images else None
        self.attr = np.concatenate([
            cls[:, None],
            lat[:, [_POS_SPL]],
            lat[:, _POS_OBJ],
            lat[:, _ROT_OBJ],
            lat[:, [_HUE_OBJ, _HUE_SPL, _HUE_BG]],
        ], axis=1).astype(np.float32)
        assert self.attr.shape[1] == len(ATTRIBUTE_NAMES), (self.attr.shape, len(ATTRIBUTE_NAMES))
        self.attribute_names = list(ATTRIBUTE_NAMES)

    def _images_handle(self):
        if self._images is not None:
            return self._images
        if self._h5 is None:                      # opened lazily, per worker process
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5["images"]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = np.asarray(self._images_handle()[i], dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1) * 2.0 - 1.0   # HWC uint8 -> CHW in [-1, 1]
        return {"img": img, "attr": torch.from_numpy(self.attr[i]), "index": i}
