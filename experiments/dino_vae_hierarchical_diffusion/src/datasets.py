"""CelebA attribute, image-path, synthetic-image, and latent datasets."""
from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

SPLIT_TO_PARTITION = {
    "train": 0,
    "val": 1,
    "valid": 1,
    "validation": 1,
    "test": 2,
    "all": None,
}


def _build_image_transform(image_size: int):
    transforms = importlib.import_module("torchvision.transforms")
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def read_celeba_attributes(attr_path: str | Path) -> tuple[list[str], list[dict[str, object]]]:
    attr_path = Path(attr_path)
    with attr_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"Invalid CelebA attribute file: {attr_path}")
    attr_names = lines[1].split()
    rows: list[dict[str, object]] = []
    for line in lines[2:]:
        parts = line.split()
        image_id, values = parts[0], parts[1:]
        if len(values) != len(attr_names):
            raise ValueError(
                f"Attribute count mismatch for {image_id}: expected {len(attr_names)}, got {len(values)}"
            )
        row: dict[str, object] = {"image_id": image_id}
        row.update({name: 1.0 if int(value) > 0 else 0.0 for name, value in zip(attr_names, values)})
        rows.append(row)
    return attr_names, rows


def read_celeba_partitions(partition_path: str | Path | None) -> dict[str, int]:
    if partition_path is None or not Path(partition_path).exists():
        return {}
    partitions: dict[str, int] = {}
    with Path(partition_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) == 2:
                partitions[parts[0]] = int(parts[1])
    return partitions


class CelebAAttributeDataset(Dataset):
    """CelebA-style images joined to the official attribute and partition files."""

    def __init__(
        self,
        image_dir: str | Path,
        attr_path: str | Path,
        partition_path: str | Path | None = None,
        split: str = "all",
        image_size: int = 256,
        attr_names: Sequence[str] | None = None,
    ) -> None:
        if split not in SPLIT_TO_PARTITION:
            raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLIT_TO_PARTITION)}")
        self.image_dir = Path(image_dir)
        all_attr_names, rows = read_celeba_attributes(attr_path)
        partitions = read_celeba_partitions(partition_path)
        partition_id = SPLIT_TO_PARTITION[split]
        if partitions and partition_id is not None:
            rows = [row for row in rows if partitions.get(str(row["image_id"])) == partition_id]
        self.attr_names = list(attr_names) if attr_names is not None else all_attr_names
        missing = sorted(set(self.attr_names) - set(all_attr_names))
        if missing:
            raise ValueError(f"Attributes not present in {attr_path}: {missing}")
        self.records = rows
        self.transform = _build_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.records[index]
        image_id = str(row["image_id"])
        image_path = self.image_dir / image_id
        image = Image.open(image_path).convert("RGB")
        attrs = torch.tensor([float(row[name]) for name in self.attr_names], dtype=torch.float32)
        return {
            "image": self.transform(image),
            "attributes": attrs,
            "image_id": image_id,
            "image_path": str(image_path),
            "index": index,
        }


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Sequence[str | Path], image_size: int = 256) -> None:
        self.image_paths = [Path(path) for path in image_paths]
        self.transform = _build_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.image_paths[index]
        return {
            "image": self.transform(Image.open(path).convert("RGB")),
            "image_id": path.name,
            "image_path": str(path),
            "index": index,
        }


class SyntheticImageDataset(Dataset):
    def __init__(self, n: int = 100, image_size: int = 256, seed: int = 0) -> None:
        self.n, self.image_size, self.seed = n, image_size, seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict[str, object]:
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.rand(3, self.image_size, self.image_size, generator=generator) * 2 - 1
        return {"image": image, "image_id": f"synthetic_{index:06d}", "index": index}


class LatentDataset(Dataset):
    """Load the batched hierarchy cache emitted by ``extract_latents.py``."""

    def __init__(self, path: str | Path) -> None:
        self.data = torch.load(path, map_location="cpu", weights_only=False)
        if "latents" not in self.data:
            raise ValueError(f"Latent cache does not contain a 'latents' entry: {path}")
        self.zs = self.data["latents"]
        if not self.zs:
            raise ValueError(f"Latent cache contains no hierarchy levels: {path}")

    def __len__(self) -> int:
        return self.zs[0].shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return tuple(z[index] for z in self.zs)


def build_subset(dataset: Dataset, num_images: int | None) -> Subset:
    size = len(dataset) if num_images is None else min(num_images, len(dataset))
    return Subset(dataset, list(range(size)))


def build_image_dataset(cfg: dict, split: str | None = None, max_images: int | None = None) -> Dataset:
    dataset_cfg = cfg["dataset"]
    if dataset_cfg.get("synthetic", False):
        dataset: Dataset = SyntheticImageDataset(
            dataset_cfg.get("synthetic_size", 100), dataset_cfg["image_size"], cfg["seed"]
        )
    elif "image_dir" in dataset_cfg and "attr_path" in dataset_cfg:
        dataset = CelebAAttributeDataset(
            image_dir=dataset_cfg["image_dir"],
            attr_path=dataset_cfg["attr_path"],
            partition_path=dataset_cfg.get("partition_path"),
            split=split or dataset_cfg.get("split", "all"),
            image_size=dataset_cfg["image_size"],
            attr_names=dataset_cfg.get("attr_names"),
        )
    elif "image_paths" in dataset_cfg:
        dataset = ImagePathDataset(dataset_cfg["image_paths"], dataset_cfg["image_size"])
    else:
        root = Path(dataset_cfg.get(f"{split}_root", dataset_cfg.get("root", "")))
        paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if not paths:
            raise FileNotFoundError(f"No images found under {root}")
        dataset = ImagePathDataset(paths, dataset_cfg["image_size"])
    return build_subset(dataset, max_images)


def image_loader(
    cfg: dict,
    split: str | None = None,
    max_images: int | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    dataset = build_image_dataset(cfg, split=split, max_images=max_images)
    effective_split = split or cfg["dataset"].get("split", "all")
    stage1_cfg = cfg.get("stage1", {})
    return DataLoader(
        dataset,
        batch_size=stage1_cfg.get("batch_size", cfg["dataset"].get("batch_size", 32)),
        shuffle=(effective_split == "train" if shuffle is None else shuffle),
        num_workers=cfg["dataset"].get("num_workers", 4),
        pin_memory=True,
    )


def write_dicts_csv(rows: Sequence[dict[str, object]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dataset_attr_names(dataset: Dataset) -> list[str] | None:
    """Return selected attribute names through any Subset wrappers."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    names = getattr(dataset, "attr_names", None)
    return list(names) if names is not None else None
