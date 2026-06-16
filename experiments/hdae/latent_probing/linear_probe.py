"""Train one linear binary probe per (latent level, CelebA attribute).

The probe dataset is an ``.npz`` written by ``extract_latents.py`` with keys:
``z_level_0``, ..., ``attrs`` in {-1,+1}, ``partitions`` in {0,1,2}, and
``attribute_names``.  Latent levels are kept separate on purpose so a run with
40 attributes and 3 levels trains and reports 120 independent classifiers.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import csv
import json

import numpy as np


@dataclass(frozen=True)
class ProbeJob:
    level: int
    attribute_index: int
    attribute_name: str
    latent_key: str


def latent_keys(arrays: Dict[str, np.ndarray]) -> List[str]:
    keys = [k for k in arrays.keys() if k.startswith("z_level_")]
    return sorted(keys, key=lambda k: int(k.rsplit("_", 1)[1]))


def make_probe_jobs(attribute_names: Sequence[str], num_levels: int) -> List[ProbeJob]:
    return [
        ProbeJob(level=level,
                 attribute_index=attr_idx,
                 attribute_name=str(attr_name),
                 latent_key=f"z_level_{level}")
        for level in range(num_levels)
        for attr_idx, attr_name in enumerate(attribute_names)
    ]


def split_indices(partitions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.where(partitions == 0)[0]
    val = np.where(partitions == 1)[0]
    test = np.where(partitions == 2)[0]
    if min(len(train), len(val), len(test)) == 0:
        n = len(partitions)
        first = max(1, int(0.8 * n))
        second = max(first + 1, int(0.9 * n))
        train, val, test = np.arange(first), np.arange(first, second), np.arange(second, n)
    return train, val, test


def standardize(train_x: np.ndarray, *others: np.ndarray):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((train_x - mean) / std, *[(x - mean) / std for x in others], mean, std)


def binary_metrics(logits: np.ndarray, targets01: np.ndarray) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= 0.5).astype(np.int64)
    target = targets01.astype(np.int64)
    acc = float((pred == target).mean())
    pos = target == 1
    neg = ~pos
    pos_acc = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    neg_acc = float((pred[neg] == 0).mean()) if neg.any() else float("nan")
    balanced = float(np.nanmean([pos_acc, neg_acc]))
    return {"accuracy": acc, "balanced_accuracy": balanced,
            "positive_accuracy": pos_acc, "negative_accuracy": neg_acc}


def _train_one_torch(train_x, train_y, val_x, val_y, lr, weight_decay, max_epochs,
                     batch_size, patience, device, seed):
    import torch
    torch.manual_seed(seed)
    x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_y[:, None], dtype=torch.float32, device=device)
    vx = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    vy = torch.as_tensor(val_y[:, None], dtype=torch.float32, device=device)
    model = torch.nn.Linear(train_x.shape[1], 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_state, best_loss, bad = None, float("inf"), 0
    for _epoch in range(max_epochs):
        perm = torch.randperm(len(x), device=device)
        for start in range(0, len(x), batch_size):
            idx = perm[start:start + batch_size]
            loss = loss_fn(model(x[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            val_loss = float(loss_fn(model(vx), vy).detach().cpu())
        if val_loss < best_loss - 1e-6:
            best_loss, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.cpu(), best_loss


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def train_all_probes(latents_npz: str, output_dir: str, *, lr: float = 1e-3,
                     weight_decay: float = 1e-4, max_epochs: int = 200,
                     batch_size: int = 256, patience: int = 20,
                     device: str = "cpu", seed: int = 0) -> List[Dict[str, float]]:
    """Train and save every independent level/attribute linear classifier."""
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    arrays = np.load(latents_npz, allow_pickle=True)
    keys = latent_keys(arrays)
    attrs = arrays["attrs"].astype(np.int64)
    partitions = arrays["partitions"].astype(np.int64)
    attribute_names = [str(x) for x in arrays["attribute_names"]]
    jobs = make_probe_jobs(attribute_names, len(keys))
    train_idx, val_idx, test_idx = split_indices(partitions)
    out = Path(output_dir); weights_dir = out / "weights"; weights_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in jobs:
        x_all = arrays[job.latent_key].astype(np.float32)
        y_all = (attrs[:, job.attribute_index] > 0).astype(np.float32)
        train_x, val_x, test_x, mean, std = standardize(x_all[train_idx], x_all[val_idx], x_all[test_idx])
        train_y, val_y, test_y = y_all[train_idx], y_all[val_idx], y_all[test_idx]
        model, best_val_loss = _train_one_torch(train_x, train_y, val_x, val_y, lr,
                                                weight_decay, max_epochs, batch_size,
                                                patience, device, seed + job.level * 1000 + job.attribute_index)
        with torch.no_grad():
            val_logits = model(torch.as_tensor(val_x, dtype=torch.float32)).squeeze(1).numpy()
            test_logits = model(torch.as_tensor(test_x, dtype=torch.float32)).squeeze(1).numpy()
        row = {"level": job.level, "latent_key": job.latent_key,
               "attribute_index": job.attribute_index, "attribute_name": job.attribute_name,
               "num_train": len(train_idx), "num_val": len(val_idx), "num_test": len(test_idx),
               "best_val_loss": best_val_loss}
        row.update({f"val_{k}": v for k, v in binary_metrics(val_logits, val_y).items()})
        row.update({f"test_{k}": v for k, v in binary_metrics(test_logits, test_y).items()})
        rows.append(row)
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                    "job": job.__dict__, "row": row},
                   weights_dir / f"level{job.level:02d}_attr{job.attribute_index:02d}_{_safe_name(job.attribute_name)}.pt")
    with open(out / "probe_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    summary = {"num_levels": len(keys), "num_attributes": len(attribute_names),
               "num_classifiers": len(rows), "metrics_csv": str(out / "probe_metrics.csv")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return rows
