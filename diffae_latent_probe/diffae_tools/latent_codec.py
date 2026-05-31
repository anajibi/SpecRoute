from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import csv
import json
import math

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class LatentBundle:
    semantic: np.ndarray
    stochastic: Any
    image_ids: list[str]
    metadata: dict


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_array(path: str | Path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.dtype == torch.float16:
            value = value.float()
        try:
            np.save(path, value.numpy())
            return path
        except Exception:
            torch.save(value, path.with_suffix(".pt"))
            return path.with_suffix(".pt")
    if isinstance(value, np.ndarray):
        try:
            np.save(path, value)
            return path
        except Exception:
            torch.save(torch.from_numpy(value), path.with_suffix(".pt"))
            return path.with_suffix(".pt")
    torch.save(value, path.with_suffix(".pt"))
    return path.with_suffix(".pt")


def load_array(path: str | Path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu")
    if path.exists() and path.is_file():
        try:
            return np.load(path, allow_pickle=False)
        except Exception:
            return torch.load(path, map_location="cpu")
    raise FileNotFoundError(path)


def save_latent_bundle(
    output_dir: str | Path,
    semantic,
    stochastic,
    image_ids: Iterable[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> LatentBundle:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic_np = _to_numpy(semantic).astype(np.float32)
    image_ids = list(image_ids)
    semantic_path = save_array(output_dir / "z_sem.npy", semantic_np)
    stochastic_path = save_array(output_dir / "stochastic.npy", stochastic)
    with (output_dir / "image_ids.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id"])
        for image_id in image_ids:
            writer.writerow([image_id])
    metadata = dict(metadata or {})
    metadata.update(
        {
            "semantic_path": str(semantic_path),
            "stochastic_path": str(stochastic_path),
            "num_images": len(image_ids),
            "semantic_shape": list(semantic_np.shape),
            "stochastic_shape": list(_to_numpy(stochastic).shape),
        }
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return LatentBundle(semantic=semantic_np, stochastic=stochastic, image_ids=image_ids, metadata=metadata)


def load_latent_bundle(latent_dir: str | Path) -> LatentBundle:
    latent_dir = Path(latent_dir)
    semantic = load_array(latent_dir / "z_sem.npy")
    stochastic_path = latent_dir / "stochastic.npy"
    if not stochastic_path.exists():
        stochastic_path = latent_dir / "stochastic.pt"
    stochastic = load_array(stochastic_path)
    image_ids_csv = latent_dir / "image_ids.csv"
    image_ids = []
    with image_ids_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_ids.append(row["image_id"])
    metadata = json.loads((latent_dir / "metadata.json").read_text()) if (latent_dir / "metadata.json").exists() else {}
    return LatentBundle(semantic=_to_numpy(semantic), stochastic=stochastic, image_ids=image_ids, metadata=metadata)


def _adaptive_pool_features(latent: torch.Tensor, size: int = 8) -> torch.Tensor:
    latent = latent.float()
    if latent.ndim == 4:
        pooled = F.adaptive_avg_pool2d(latent, (size, size))
        return pooled.flatten(1)
    if latent.ndim == 3:
        pooled = F.adaptive_avg_pool1d(latent.transpose(1, 2), size).transpose(1, 2)
        return pooled.flatten(1)
    return latent.flatten(1)


def flatten_latent_for_probe(
    latent,
    mode: str = "auto",
    target_dim: int = 512,
    random_state: int = 0,
    pca_model=None,
    max_flat_features: int = 8192,
):
    """Convert a latent tensor/array into a 2D feature matrix for probes.

    Supported modes:
    - auto: use identity for 2D latents, spatial pooling for large 4D/3D latents.
    - flatten: flatten the latent if it is not too large.
    - gap: global average pooling; for 4D returns mean+std, for 3D returns mean+std.
    - random_projection: flatten then project with Gaussian random projection.
    - pca: flatten then project with PCA.
    - summary: combine pooled summaries and statistics.
    """
    from sklearn.decomposition import PCA
    from sklearn.random_projection import GaussianRandomProjection

    is_torch = torch.is_tensor(latent)
    x = latent.detach().cpu() if is_torch else torch.as_tensor(np.asarray(latent))
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim == 2:
        feats = x.float().numpy()
    elif x.ndim == 3:
        # (N, T, D) or (N, C, L)
        if mode in {"summary", "auto", "gap"}:
            mean = x.float().mean(dim=1)
            std = x.float().std(dim=1)
            feats = torch.cat([mean, std], dim=1).numpy()
        else:
            flat = x.flatten(1).float().numpy()
            feats = flat
    elif x.ndim == 4:
        if mode in {"gap", "summary", "auto"}:
            mean = x.float().mean(dim=(2, 3))
            std = x.float().std(dim=(2, 3))
            feats = torch.cat([mean, std], dim=1).numpy()
            if mode == "summary":
                pooled = _adaptive_pool_features(x.float(), size=8).numpy()
                feats = np.concatenate([feats, pooled], axis=1)
        else:
            feats = x.flatten(1).float().numpy()
    else:
        feats = x.flatten(1).float().numpy()

    if mode == "auto":
        if feats.shape[1] > max_flat_features:
            mode = "gap"
        else:
            return feats.astype(np.float32), None

    if mode == "flatten":
        if feats.shape[1] > max_flat_features:
            raise ValueError(
                f"Latent has {feats.shape[1]} features; use gap/random_projection/pca instead"
            )
        return feats.astype(np.float32), None

    if mode == "gap":
        return feats.astype(np.float32), None

    if mode == "summary":
        return feats.astype(np.float32), None

    if mode == "random_projection":
        projector = GaussianRandomProjection(n_components=min(target_dim, feats.shape[1]), random_state=random_state)
        feats = projector.fit_transform(feats)
        return feats.astype(np.float32), projector

    if mode == "pca":
        if pca_model is None:
            n_components = min(target_dim, feats.shape[0], feats.shape[1])
            pca_model = PCA(n_components=n_components, random_state=random_state)
            feats = pca_model.fit_transform(feats)
        else:
            feats = pca_model.transform(feats)
        return feats.astype(np.float32), pca_model

    raise ValueError(f"Unknown feature mode: {mode}")


def latent_shape(latent) -> Tuple[int, ...]:
    if torch.is_tensor(latent):
        return tuple(latent.shape)
    return tuple(np.asarray(latent).shape)

