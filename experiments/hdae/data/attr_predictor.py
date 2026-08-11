"""Reusable, dataset-agnostic per-attribute CNN predictors.

Estimates a single scalar factor from an image -- e.g. "how thick is this
digit's stroke" or "what's this face's smile probability" -- via a small
CNN trained end-to-end on (image, logged value) pairs. Deliberately not one
shared backbone with multiple output heads: each attribute gets its own,
fully independent network (same architecture template, separate weights).
That costs more compute than a shared trunk, but it means no attribute's
gradient can interfere with another's, and any one predictor can be
trained, checkpointed, and swapped out on its own.

Works against any dataset exposing the repo's `{"img", "attr", ...}`
contract (`data/celeba_hq.py`'s `CelebAHQPacked`, `data/morphomnist.py`'s
`MorphoMNISTPacked`, or a future one) -- nothing here is MorphoMNIST- or
CelebA-specific. A dataset-specific driver script picks the `AttrSpec`
list and checkpoint paths; this module only knows about
(image, scalar-or-angle target) pairs.
"""
import dataclasses
import json
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision("high")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclasses.dataclass
class AttrSpec:
    """One target attribute's prediction contract.

    `kind="scalar"`: regressed directly, after min-max normalization to
    [-1, 1] using `lo`/`hi` (required). Use for anything without wraparound
    (thickness, intensity, translate_x, rotation, scale, ...).

    `kind="circular"`: the raw value is an angle that wraps -- e.g. hue in
    [0, 1) or a phase in [0, 2*pi). Predicted as a (sin, cos) pair (trained
    on MSE in that space, which has no discontinuity at the wrap point) and
    decoded back to the original units via atan2. `period` is the wrap
    period (1.0 for hue, 2*pi for a radian phase).

    `kind="categorical"`: `[lo, hi)` is divided into `num_bins` equal-width
    bins and treated as a classification problem (cross-entropy over bin
    logits), not a regression. Use when the regression loss landscape is
    the actual problem -- e.g. an angle confounded with unrelated shape
    variation, where MSE rewards hedging toward the mean more than it
    rewards getting individual examples right. Decoded back to raw units
    as the predicted bin's center, `lo + (argmax + 0.5) * (hi-lo)/num_bins`.
    """
    name: str
    kind: str  # "scalar" | "circular" | "categorical"
    lo: Optional[float] = None
    hi: Optional[float] = None
    period: Optional[float] = None
    num_bins: Optional[int] = None

    def __post_init__(self):
        if self.kind == "scalar" and (self.lo is None or self.hi is None):
            raise ValueError(f"AttrSpec {self.name!r}: kind='scalar' needs lo/hi")
        if self.kind == "circular" and self.period is None:
            raise ValueError(f"AttrSpec {self.name!r}: kind='circular' needs period")
        if self.kind == "categorical" and (self.lo is None or self.hi is None or not self.num_bins):
            raise ValueError(f"AttrSpec {self.name!r}: kind='categorical' needs lo/hi/num_bins")
        if self.kind not in ("scalar", "circular", "categorical"):
            raise ValueError(f"AttrSpec {self.name!r}: unknown kind {self.kind!r}")

    @property
    def output_dim(self) -> int:
        if self.kind == "scalar":
            return 1
        if self.kind == "circular":
            return 2
        return self.num_bins

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        """raw value(s), shape (B,) -> network target space:
        scalar: [-1,1] float, shape (B,1). circular: (sin,cos) float, shape (B,2).
        categorical: bin index, LongTensor shape (B,) (for F.cross_entropy, not one-hot)."""
        if self.kind == "scalar":
            unit = (2.0 * (raw - self.lo) / (self.hi - self.lo) - 1.0).clamp(-1.0, 1.0)
            return unit.unsqueeze(-1)
        if self.kind == "categorical":
            frac = (raw - self.lo) / (self.hi - self.lo)
            return (frac * self.num_bins).long().clamp(0, self.num_bins - 1)
        angle = raw * (2 * torch.pi / self.period)
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)

    def denormalize(self, pred: torch.Tensor) -> torch.Tensor:
        """network output -> raw units, shape (B,). Inverse of `normalize`.
        scalar/circular: pred is (B, output_dim) float (network output).
        categorical: pred is (B, num_bins) logits -- decoded via argmax, not softmax-expectation,
        so the reported value is what the network actually committed to, not a hedge."""
        if self.kind == "scalar":
            return (pred.squeeze(-1).clamp(-1.0, 1.0) + 1.0) / 2.0 * (self.hi - self.lo) + self.lo
        if self.kind == "categorical":
            bin_idx = pred.argmax(dim=-1).float()
            return self.lo + (bin_idx + 0.5) * (self.hi - self.lo) / self.num_bins
        angle = torch.atan2(pred[..., 0], pred[..., 1])
        return (angle % (2 * torch.pi)) * (self.period / (2 * torch.pi))


torch.serialization.add_safe_globals([AttrSpec])  # this dataclass rides along in checkpoint hparams (torch>=2.6
                                                  # defaults torch.load to weights_only=True, which otherwise
                                                  # rejects it as an un-allowlisted global)


@dataclasses.dataclass
class AugmentConfig:
    """Optional robustness augmentation, applied to the input image only (targets are untouched).
    Disabled by default (every probability/strength at 0) -- wire up non-zero values to make a
    predictor robust to blur/noise/brightness at inference time; nothing here is enabled unless
    explicitly configured.
    """
    gaussian_blur_prob: float = 0.0
    gaussian_blur_sigma_range: tuple = (0.1, 1.5)
    gaussian_noise_prob: float = 0.0
    gaussian_noise_std: float = 0.05
    brightness_jitter_prob: float = 0.0
    brightness_jitter_strength: float = 0.2

    @property
    def enabled(self) -> bool:
        return self.gaussian_blur_prob > 0 or self.gaussian_noise_prob > 0 or self.brightness_jitter_prob > 0

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        """img: (C,H,W) float tensor in [-1,1]. Applied independently per-call (per-sample)."""
        if torch.rand(()) < self.gaussian_blur_prob:
            lo, hi = self.gaussian_blur_sigma_range
            sigma = float(torch.empty(()).uniform_(lo, hi))
            k = max(3, int(sigma * 4) | 1)  # odd kernel size, scaled to sigma
            img = torchvision_gaussian_blur(img, k, sigma)
        if torch.rand(()) < self.gaussian_noise_prob:
            img = img + torch.randn_like(img) * self.gaussian_noise_std
        if torch.rand(()) < self.brightness_jitter_prob:
            factor = 1.0 + float(torch.empty(()).uniform_(-self.brightness_jitter_strength,
                                                          self.brightness_jitter_strength))
            img = (img * factor).clamp(-1.0, 1.0)
        return img


def torchvision_gaussian_blur(img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    from torchvision.transforms.functional import gaussian_blur
    return gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)


class AugmentedDataset(Dataset):
    """Wraps a base `{"img", ...}` dataset, applying `augment` to `img` on read. A no-op pass-through
    when `augment` is None or disabled -- always safe to wrap with, on or off."""

    def __init__(self, base: Dataset, augment: Optional[AugmentConfig] = None):
        self.base = base
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        if self.augment is not None and self.augment.enabled:
            item["img"] = self.augment.apply(item["img"])
        return item


class SmallCNN(nn.Module):
    """One self-contained backbone+head network, sized for small (~64px) images. Every AttrSpec
    gets its own instance of this class -- no parameters are shared across attributes.

    ``dropout`` sits between pooled features and the head's first linear layer. It matters more
    than it looks: every attribute trained under this template overfits hard (train MSE reaches
    ~1e-4-5e-3 while val MSE plateaus 10-1000x higher and *stays flat*, i.e. more capacity isn't
    the bottleneck) -- dropout plus the optimizer's weight decay (see
    ``AttrRegressionModule``) target that gap directly instead of changing model size."""

    def __init__(self, in_channels: int = 3, output_dim: int = 1, base_channels: int = 32,
                dropout: float = 0.0):
        super().__init__()
        c = base_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.GroupNorm(8, c * 2), nn.SiLU(),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.GroupNorm(8, c * 4), nn.SiLU(),
            nn.Conv2d(c * 4, c * 8, 3, stride=2, padding=1), nn.GroupNorm(8, c * 8), nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(c * 8, c * 4), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(c * 4, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.pool(self.conv(x)).flatten(1)
        return self.head(feat)


class AttrRegressionModule(pl.LightningModule):
    """One attribute's SmallCNN, wrapped for pl.Trainer. `attr_col` indexes into the dataset's
    `attr` vector (see `attribute_names`) to pull the target for this specific attribute."""

    def __init__(self, spec: AttrSpec, attr_col: int, in_channels: int = 3, base_channels: int = 32,
                lr: float = 1e-3, weight_decay: float = 0.0, dropout: float = 0.0,
                lr_plateau_patience: int = 0):
        super().__init__()
        self.save_hyperparameters(ignore=[])
        self.spec = spec
        self.attr_col = attr_col
        self.model = SmallCNN(in_channels=in_channels, output_dim=spec.output_dim, base_channels=base_channels,
                              dropout=dropout)
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_plateau_patience = lr_plateau_patience

    def _step(self, batch):
        raw_target = batch["attr"][:, self.attr_col].float()
        target = self.spec.normalize(raw_target)
        pred = self.model(batch["img"])
        if self.spec.kind == "categorical":
            loss = F.cross_entropy(pred, target)
        else:
            loss = F.mse_loss(pred, target)
        return loss, pred, raw_target

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch["img"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, raw_target = self._step(batch)
        pred_raw = self.spec.denormalize(pred)
        if self.spec.kind == "circular":
            diff = (pred_raw - raw_target) % self.spec.period
            err = torch.minimum(diff, self.spec.period - diff)
        else:
            err = (pred_raw - raw_target).abs()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["img"].shape[0])
        self.log("val_mae_raw", err.mean(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["img"].shape[0])
        if self.spec.kind == "categorical":
            target = self.spec.normalize(raw_target)
            acc = (pred.argmax(dim=-1) == target).float().mean()
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["img"].shape[0])
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_plateau_patience <= 0:
            return opt
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                                patience=self.lr_plateau_patience)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    @torch.no_grad()
    def predict_raw(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B,C,H,W). Returns raw-unit predictions, (B,) for scalar or (B,) angle for circular."""
        self.eval()
        pred = self.model(img.to(self.device))
        return self.spec.denormalize(pred).cpu()


def train_attr_predictor(spec: AttrSpec, attr_col: int, train_dataset: Dataset, val_dataset: Dataset,
                         output_dir: str, in_channels: int = 3, base_channels: int = 32, lr: float = 1e-3,
                         weight_decay: float = 0.0, dropout: float = 0.0, lr_plateau_patience: int = 0,
                         augment: "AugmentConfig | None" = None,
                         batch_size: int = 256, max_epochs: int = 100, patience: int = 8,
                         num_workers: int = 4, accelerator: str = "auto") -> str:
    """Trains one independent AttrRegressionModule and returns its best checkpoint path.

    `val_dataset` drives both early stopping and checkpoint selection -- callers are responsible
    for keeping it disjoint from any data used for final evaluation (this function has no
    knowledge of a held-out test split; that's the caller's split to protect).

    `weight_decay`/`dropout`/`lr_plateau_patience` exist because every attribute trained under
    this template overfits (train loss reaches ~1e-4-5e-3 while val loss plateaus far higher and
    flat) -- these three are what actually address that, not more capacity. `augment`, if given,
    wraps `train_dataset` only (never `val_dataset` -- validation must see undistorted images).
    """
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from torch.utils.data import DataLoader

    if augment is not None:
        train_dataset = AugmentedDataset(train_dataset, augment)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=num_workers > 0)

    module = AttrRegressionModule(spec, attr_col, in_channels=in_channels, base_channels=base_channels, lr=lr,
                                  weight_decay=weight_decay, dropout=dropout,
                                  lr_plateau_patience=lr_plateau_patience)
    ckpt_dir = Path(output_dir) / spec.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(dirpath=str(ckpt_dir), filename="best", monitor="val_loss", mode="min",
                                    save_top_k=1)
    early_stop_cb = EarlyStopping(monitor="val_loss", mode="min", patience=patience)

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=1,
                         callbacks=[checkpoint_cb, early_stop_cb], enable_progress_bar=True,
                         logger=False, enable_model_summary=False)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    spec_path = ckpt_dir / "spec.json"
    spec_path.write_text(json.dumps(dataclasses.asdict(spec)))
    return checkpoint_cb.best_model_path


def load_attr_predictor(checkpoint_path: str, attr_col: int) -> AttrRegressionModule:
    """`in_channels`/`base_channels`/`dropout`/etc are deliberately NOT passed here -- they're
    restored from the checkpoint's own saved hyperparameters (`save_hyperparameters` in
    `AttrRegressionModule.__init__`), so a checkpoint trained with a non-default architecture
    (e.g. a wider `base_channels` for a harder attribute) loads correctly without the caller
    needing to know or track what architecture each individual checkpoint used."""
    spec_path = Path(checkpoint_path).parent / "spec.json"
    spec = AttrSpec(**json.loads(spec_path.read_text()))
    module = AttrRegressionModule.load_from_checkpoint(checkpoint_path, spec=spec, attr_col=attr_col)
    module.eval()
    return module
