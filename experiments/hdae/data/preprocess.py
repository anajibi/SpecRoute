"""One-time, resumable raw-uint8 CelebA-HQ LMDB packer."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional


def enumerate_images(image_dir, limit: Optional[int] = None):
    paths = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
    paths.sort(key=lambda p: (int(p.stem) if p.stem.isdigit() else float('inf'), p.name))
    return paths[:limit] if limit else paths


def _resize(args):
    path, image_size, resize_filter = args
    from PIL import Image
    filters = {"lanczos": Image.Resampling.LANCZOS, "bicubic": Image.Resampling.BICUBIC}
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), filters[resize_filter])
        return image.tobytes()


def preprocess(image_dir, lmdb_path, attr_path, partition_path, attr_npz,
               image_size=64, resize_filter="bicubic", num_workers=1, limit=None,
               map_size=1 << 40):
    import lmdb
    from .attributes import align_attributes
    paths = enumerate_images(image_dir, limit)
    if not paths:
        raise FileNotFoundError(f"no images found in {image_dir}")
    lmdb_path, meta_path = Path(lmdb_path), Path(lmdb_path) / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("image_size") == image_size and meta.get("num_images") == len(paths) and Path(attr_npz).exists():
            return meta
    lmdb_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_path), map_size=map_size, subdir=True)
    image_ids = [p.stem for p in paths]
    with env.begin() as txn:
        completed = int((txn.get(b"__completed__") or b"0").decode())
    tasks = ((str(p), image_size, resize_filter) for p in paths[completed:])
    iterator = map(_resize, tasks) if num_workers <= 1 else ProcessPoolExecutor(num_workers).map(_resize, tasks)
    for offset, raw in enumerate(iterator, start=completed):
        with env.begin(write=True) as txn:
            key = f"{offset:08d}".encode()
            txn.put(key, raw)
            txn.put(key + b":id", image_ids[offset].encode())
            txn.put(b"__completed__", str(offset + 1).encode())
    env.sync(); env.close()
    aligned = align_attributes(image_ids, attr_path, partition_path, attr_npz)
    meta = {"image_size": image_size, "num_images": len(paths), "image_ids": image_ids,
            "filter": resize_filter, "storage": "raw_uint8_rgb", "source_dir": str(image_dir),
            "attr_npz": str(attr_npz), **aligned}
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta
