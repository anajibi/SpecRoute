from pathlib import Path

import torch
from PIL import Image

from dino_vae_hierarchical_diffusion.src.datasets import (
    CelebAAttributeDataset,
    LatentDataset,
    SyntheticImageDataset,
    build_image_dataset,
    dataset_attr_names,
    read_celeba_attributes,
    write_dicts_csv,
)


def _write_celeba_fixture(root: Path) -> tuple[Path, Path, Path]:
    image_dir = root / "images"
    image_dir.mkdir()
    for name, color in [("000001.jpg", "red"), ("000002.jpg", "blue")]:
        Image.new("RGB", (12, 16), color).save(image_dir / name)
    attr_path = root / "list_attr_celeba.txt"
    attr_path.write_text("2\nSmiling Eyeglasses\n000001.jpg 1 -1\n000002.jpg -1 1\n", encoding="utf-8")
    partition_path = root / "list_eval_partition.txt"
    partition_path.write_text("000001.jpg 0\n000002.jpg 2\n", encoding="utf-8")
    return image_dir, attr_path, partition_path


def test_celeba_attributes_and_partition_split(tmp_path):
    image_dir, attr_path, partition_path = _write_celeba_fixture(tmp_path)
    names, rows = read_celeba_attributes(attr_path)
    assert names == ["Smiling", "Eyeglasses"]
    assert rows[0]["Smiling"] == 1.0
    dataset = CelebAAttributeDataset(image_dir, attr_path, partition_path, split="train", image_size=8)
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["image"].shape == (3, 8, 8)
    assert sample["image_id"] == "000001.jpg"
    assert sample["attributes"].tolist() == [1.0, 0.0]


def test_build_dataset_supports_celeba_config_and_selected_attributes(tmp_path):
    image_dir, attr_path, partition_path = _write_celeba_fixture(tmp_path)
    cfg = {
        "seed": 7,
        "dataset": {
            "image_dir": str(image_dir),
            "attr_path": str(attr_path),
            "partition_path": str(partition_path),
            "split": "test",
            "attr_names": ["Eyeglasses"],
            "image_size": 8,
        },
    }
    dataset = build_image_dataset(cfg, max_images=1)
    assert len(dataset) == 1
    assert dataset_attr_names(dataset) == ["Eyeglasses"]
    assert dataset[0]["image_id"] == "000002.jpg"


def test_synthetic_and_latent_datasets_are_deterministic(tmp_path):
    synthetic = SyntheticImageDataset(n=2, image_size=8, seed=4)
    assert torch.equal(synthetic[0]["image"], synthetic[0]["image"])
    latent_path = tmp_path / "latents.pt"
    z3, z2 = torch.randn(3, 4), torch.randn(3, 2, 2, 2)
    torch.save({"latents": (z3, z2)}, latent_path)
    latents = LatentDataset(latent_path)
    assert len(latents) == 3
    assert torch.equal(latents[1][0], z3[1])


def test_write_dicts_csv_unions_fields(tmp_path):
    path = tmp_path / "metrics.csv"
    write_dicts_csv([{"a": 1}, {"a": 2, "b": 3}], path)
    assert path.read_text(encoding="utf-8").splitlines()[0] == "a,b"
