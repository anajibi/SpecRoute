"""Pack Causal3DIdent into a single HDF5 at a chosen resolution.

Streams straight from the Zenodo tarballs (`tarfile` in `r|gz` sequential mode) --
decode, resize, write -- so the ~7.8GB of full-size PNGs are never materialised on
disk. Mirrors the layout of `morphomnist_70k.h5` so the datamodule looks familiar.

Source: Zenodo 10.5281/zenodo.4784282 (CC-BY-4.0), von Kuegelgen et al., NeurIPS 2021.
Layout inside each tarball:
    {split}/images_{i}/{NNNNN}.png     i = object class 0..6, 5-digit index
    {split}/raw_latents_{i}.npy        (N_i, 10) float32, all in [-1, 1]

The 10 raw latent columns group into the attributes the model conditions on:
    pos_obj = cols 0,1,2   rot_obj = cols 3,4,5   pos_spl = col 6
    hue_obj = col 7        hue_spl = col 8        hue_bg  = col 9   (unmodelled)
plus `class`, which is the images_{i} folder index.

Output datasets:
    images   (N, S, S, 3) uint8
    latents  (N, 10)      float32   -- raw_latents, unchanged
    class    (N,)         int64
"""
import argparse
import io
import os
import re
import sys
import tarfile
import time

import h5py
import numpy as np
from PIL import Image

MEMBER_RE = re.compile(r"^[^/]+/images_(\d)/(\d+)\.png$")
N_CLASSES = 7
N_LATENTS = 10


def load_latents(split_dir, n_classes=N_CLASSES):
    per_class = []
    for i in range(n_classes):
        p = os.path.join(split_dir, f"raw_latents_{i}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing -- extract the *.npy files first: "
                                    f"tar xzf <split>.tar.gz --wildcards '*/*.npy'")
        per_class.append(np.load(p))
    return per_class


def pack(tar_path, split_dir, out_path, size, report_every=20000):
    per_class = load_latents(split_dir)
    counts = [a.shape[0] for a in per_class]
    offsets = np.cumsum([0] + counts)
    total = int(offsets[-1])
    print(f"{os.path.basename(tar_path)}: {total} images across {N_CLASSES} classes {counts}", flush=True)

    tmp = out_path + ".partial"
    with h5py.File(tmp, "w") as h5:
        dimg = h5.create_dataset("images", (total, size, size, 3), dtype="uint8",
                                 chunks=(min(64, total), size, size, 3), compression=None)
        dlat = h5.create_dataset("latents", (total, N_LATENTS), dtype="float32")
        dcls = h5.create_dataset("class", (total,), dtype="int64")
        for i, a in enumerate(per_class):
            dlat[offsets[i]:offsets[i + 1]] = a.astype("float32")
            dcls[offsets[i]:offsets[i + 1]] = i
        h5.attrs["source"] = "zenodo 10.5281/zenodo.4784282 (Causal3DIdent)"
        h5.attrs["image_size"] = size
        h5.attrs["resample"] = "PIL LANCZOS"
        h5.attrs["latent_names"] = np.array(
            ["pos_x", "pos_y", "pos_z", "rot_a", "rot_b", "rot_g",
             "pos_spl", "hue_obj", "hue_spl", "hue_bg"], dtype=h5py.string_dtype())
        h5.attrs["class_names"] = np.array(
            ["teapot", "hare", "dragon", "cow", "armadillo", "horse", "head"],
            dtype=h5py.string_dtype())

        seen = np.zeros(total, dtype=bool)
        done = 0
        t0 = time.time()
        with tarfile.open(tar_path, mode="r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                mm = MEMBER_RE.match(m.name)
                if mm is None:
                    continue                       # .npy, .ipynb_checkpoints, dirs
                cls, idx = int(mm.group(1)), int(mm.group(2))
                if idx >= counts[cls]:
                    raise ValueError(f"{m.name}: index {idx} beyond latents rows {counts[cls]}")
                f = tf.extractfile(m)
                im = Image.open(io.BytesIO(f.read())).convert("RGB")
                if im.size != (size, size):
                    im = im.resize((size, size), Image.LANCZOS)
                row = int(offsets[cls]) + idx
                dimg[row] = np.asarray(im, dtype=np.uint8)
                seen[row] = True
                done += 1
                if done % report_every == 0:
                    el = time.time() - t0
                    print(f"  {done}/{total}  {done/el:.0f} img/s  eta {(total-done)/(done/el)/60:.1f} min",
                          flush=True)
        missing = int((~seen).sum())
        if missing:
            raise RuntimeError(f"{missing} rows never written -- tar is incomplete")
        h5.attrs["n_images"] = total
    os.replace(tmp, out_path)
    print(f"wrote {out_path}  ({os.path.getsize(out_path)/1e9:.2f} GB, {total} images)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/hdae/data/causal3dident")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--splits", nargs="+", default=["testset", "trainset"])
    args = ap.parse_args()
    for split in args.splits:
        tar = os.path.join(args.root, f"{split}.tar.gz")
        out = os.path.join(args.root, f"causal3dident_{split}_{args.size}.h5")
        if os.path.exists(out):
            print(f"{out} exists, skipping", flush=True)
            continue
        pack(tar, os.path.join(args.root, split), out, args.size)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
