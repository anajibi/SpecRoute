"""CheXpert protocol v0, section 4: images -> one uint8 memmap.

Streams the JPEGs straight out of the Kaggle archive and never extracts it.
That is a disk decision, not a stylistic one: the archive is 11.5 GB, the
memmap is 8.6 GB, and an extraction would be another ~11 GB on a volume with
about 12 GB free. Each worker opens its own ZipFile handle and seeks to its
own members, so the archive is read concurrently without being unpacked.

Per image: decode -> grayscale -> resize the SHORT side to 224 -> centre crop
224x224 -> uint8. Nothing else. No normalisation and no augmentation here --
those are model-side so that the stored pixels stay the one fixed artifact
every arm and every predictor sees.

Row i of the memmap is row i of chexpert_meta.csv. The alignment is positional
and nothing in the .npy records it, so the meta sha256 is written into the
sidecar JSON and consumers are expected to check it.

The CSV's Path column is prefixed "CheXpert-v1.0-small/" while the archive's
members are not; the prefix is stripped to get the member name. The archive
also carries one macOS AppleDouble entry (train/patient00001/study1/._view1_
frontal.jpg) which is metadata, not an image, and is never referenced by the
CSV -- it is reached only if a path is mangled, so a decode failure is raised
rather than skipped.
"""
import argparse, hashlib, json, os, sys, time, zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

PREFIX = "CheXpert-v1.0-small/"
SIZE = 224
_W = {}


def _init(zip_path, npy_path, members):
    _W["z"] = zipfile.ZipFile(zip_path)
    _W["a"] = np.load(npy_path, mmap_mode="r+")
    _W["m"] = members


def _one(buf):
    im = Image.open(buf)
    im.draft("L", None)          # let libjpeg decode straight to grayscale
    im = im.convert("L")
    w, h = im.size
    s = SIZE / min(w, h)
    # round, not floor: floor can leave the short side at 223 and break the crop
    nw, nh = max(SIZE, int(round(w * s))), max(SIZE, int(round(h * s)))
    im = im.resize((nw, nh), Image.BICUBIC)
    l, t = (nw - SIZE) // 2, (nh - SIZE) // 2
    im = im.crop((l, t, l + SIZE, t + SIZE))
    a = np.asarray(im, dtype=np.uint8)
    if a.shape != (SIZE, SIZE):
        raise ValueError(f"got {a.shape} after crop")
    return a


def _chunk(job):
    lo, hi = job
    z, arr, mem = _W["z"], _W["a"], _W["m"]
    blank = []
    for i in range(lo, hi):
        with z.open(mem[i]) as f:
            a = _one(f)
        if a.max() == a.min():
            blank.append(i)
        arr[i] = a
    arr.flush()
    return lo, hi, blank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()

    meta = pd.read_csv(args.meta, usecols=["path"])
    paths = meta["path"].tolist()
    n = len(paths)

    bad_prefix = [p for p in paths[:n] if not p.startswith(PREFIX)]
    if bad_prefix:
        raise SystemExit(f"{len(bad_prefix)} paths lack the {PREFIX!r} prefix, e.g. {bad_prefix[0]}")
    members = [p[len(PREFIX):] for p in paths]

    with zipfile.ZipFile(args.zip) as z:
        have = set(z.namelist())
    missing = [m for m in members if m not in have]
    if missing:
        raise SystemExit(f"{len(missing)} referenced images absent from the archive, e.g. {missing[:3]}")
    if any(os.path.basename(m).startswith("._") for m in members):
        raise SystemExit("a path resolved to an AppleDouble metadata entry")
    print(f"{n} images, all present in {args.zip}")

    free = os.statvfs(os.path.dirname(os.path.abspath(args.out))).f_bavail * \
           os.statvfs(os.path.dirname(os.path.abspath(args.out))).f_frsize
    need = n * SIZE * SIZE
    print(f"memmap {need/1e9:.2f} GB, free {free/1e9:.2f} GB")
    if free < need + (1 << 30):
        raise SystemExit("not enough free space for the memmap plus 1 GB headroom")

    arr = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.uint8, shape=(n, SIZE, SIZE))
    del arr

    jobs = [(lo, min(lo + args.chunk, n)) for lo in range(0, n, args.chunk)]
    t0 = time.time()
    done = 0
    blanks = []
    with Pool(args.workers, initializer=_init, initargs=(args.zip, args.out, members)) as pool:
        for lo, hi, bl in pool.imap_unordered(_chunk, jobs):
            done += hi - lo
            blanks += bl
            if done % 20000 < args.chunk:
                el = time.time() - t0
                print(f"  {done}/{n}  {done/el:.0f} img/s  eta {(n-done)/max(done/el,1)/60:.1f} min", flush=True)
    print(f"decoded {done} images in {(time.time()-t0)/60:.1f} min")
    if done != n:
        raise SystemExit(f"wrote {done} rows, expected {n}")

    # integrity: no constant row, and the per-row mean/std are retained so the
    # section-7 test does not have to re-read 8.6 GB to find a dead image
    a = np.load(args.out, mmap_mode="r")
    mu = np.empty(n, np.float32)
    sd = np.empty(n, np.float32)
    for lo in range(0, n, 4096):
        blk = np.asarray(a[lo:lo + 4096], dtype=np.float32)
        mu[lo:lo + 4096] = blk.mean(axis=(1, 2))
        sd[lo:lo + 4096] = blk.std(axis=(1, 2))
    const = np.flatnonzero(sd == 0)
    np.save(args.out.replace(".npy", "_rowstats.npy"), np.stack([mu, sd]))

    h = hashlib.sha256()
    with open(args.out, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    mh = hashlib.sha256(open(args.meta, "rb").read()).hexdigest()

    rec = {
        "n": n, "shape": [n, SIZE, SIZE], "dtype": "uint8",
        "pipeline": "decode -> grayscale(L) -> resize short side 224 (PIL BICUBIC) -> centre crop 224 -> uint8",
        "source_zip": os.path.abspath(args.zip),
        "source_zip_sha256_note": "see ladder/chexpert_fetch.log",
        "path_prefix_stripped": PREFIX,
        "meta": os.path.abspath(args.meta), "meta_sha256": mh,
        "npy_sha256": h.hexdigest(),
        "constant_rows": const.tolist()[:50],
        "n_constant_rows": int(const.size),
        "pixel_mean": round(float(mu.mean()), 4),
        "pixel_std_of_row_means": round(float(mu.std()), 4),
    }
    with open(args.out + ".json", "w") as f:
        json.dump(rec, f, indent=2)
    print(f"constant (dead) rows: {const.size}")
    print(f"pixel mean {rec['pixel_mean']}  sha256 {rec['npy_sha256']}")
    print(f"wrote {args.out} and {args.out}.json")
    if const.size:
        print("WARNING: constant rows present -- section 7 image-integrity test fails", file=sys.stderr)


if __name__ == "__main__":
    main()
