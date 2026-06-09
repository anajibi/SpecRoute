import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import torch
from PIL import Image

import src.datasets as datasets


def _write_metadata(tmp_path):
    attrs = tmp_path / "attrs.txt"
    attrs.write_text("3\nSmiling Young\n000001.jpg 1 -1\n000002.jpg -1 1\n000003.jpg 1 1\n")
    partitions = tmp_path / "partitions.txt"
    partitions.write_text("000001.jpg 0\n000002.jpg 1\n000003.jpg 2\n")
    images = tmp_path / "images"
    images.mkdir()
    for name in ["000001.jpg", "000002.jpg", "000003.jpg"]:
        Image.new("RGB", (4, 4)).save(images / name)
    return images, attrs, partitions


def test_attribute_parser_and_partition_filter(tmp_path, monkeypatch):
    images, attrs, partitions = _write_metadata(tmp_path)
    monkeypatch.setattr(datasets, "_build_image_transform", lambda _: lambda image: torch.zeros(3, 4, 4))
    dataset = datasets.CelebAAttributeDataset(images, attrs, partitions, split="test", attr_names=["Smiling"])
    assert len(dataset) == 1
    item = dataset[0]
    assert item["image_id"] == "000003.jpg"
    assert item["attributes"].tolist() == [1.0]
    assert item["image"].shape == (3, 4, 4)


def test_build_dataset_uses_requested_split(tmp_path, monkeypatch):
    images, attrs, partitions = _write_metadata(tmp_path)
    monkeypatch.setattr(datasets, "_build_image_transform", lambda _: lambda image: torch.zeros(3, 4, 4))
    cfg = {
        "seed": 1,
        "dataset": {
            "image_dir": str(images),
            "attr_path": str(attrs),
            "partition_path": str(partitions),
            "split": "test",
            "image_size": 4,
        },
    }
    assert len(datasets.build_image_dataset(cfg)) == 1
    assert len(datasets.build_image_dataset(cfg, split="train")) == 1
