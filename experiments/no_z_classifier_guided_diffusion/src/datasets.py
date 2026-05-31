from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset


def _build_image_transform(image_size: int):
    transforms = importlib.import_module("torchvision.transforms")
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

SPLIT_TO_PARTITION = {"train": 0, "val": 1, "valid": 1, "validation": 1, "test": 2, "all": None}


def read_celeba_attributes(attr_path: str | Path) -> tuple[list[str], list[dict[str, object]]]:
    attr_path = Path(attr_path)
    with open(attr_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"Invalid CelebA attribute file: {attr_path}")
    attr_names = lines[1].split()
    rows: list[dict[str, object]] = []
    for line in lines[2:]:
        parts = line.split()
        image_id, values = parts[0], parts[1:]
        if len(values) != len(attr_names):
            raise ValueError(f"Attribute count mismatch for {image_id}: expected {len(attr_names)}, got {len(values)}")
        row: dict[str, object] = {"image_id": image_id}
        row.update({name: 1.0 if int(value) > 0 else 0.0 for name, value in zip(attr_names, values)})
        rows.append(row)
    return attr_names, rows


def read_celeba_partitions(partition_path: str | Path | None) -> dict[str, int]:
    if partition_path is None or not Path(partition_path).exists():
        return {}
    partitions: dict[str, int] = {}
    with open(partition_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) == 2:
                partitions[parts[0]] = int(parts[1])
    return partitions


class CelebAAttributeDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        attr_path: str | Path,
        partition_path: str | Path | None = None,
        split: str = "all",
        image_size: int = 256,
        attr_names: Sequence[str] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        all_attr_names, rows = read_celeba_attributes(attr_path)
        partitions = read_celeba_partitions(partition_path)
        partition_id = SPLIT_TO_PARTITION.get(split)
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


def build_subset(dataset: Dataset, num_images: int | None) -> Subset:
    if num_images is None or num_images >= len(dataset):
        return Subset(dataset, list(range(len(dataset))))
    return Subset(dataset, list(range(num_images)))


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Sequence[str | Path], image_size: int = 256) -> None:
        self.image_paths = [Path(path) for path in image_paths]
        self.transform = _build_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.image_paths[index]
        return {"image": self.transform(Image.open(path).convert("RGB")), "image_path": str(path), "index": index}


def write_dicts_csv(rows: Sequence[dict[str, object]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
