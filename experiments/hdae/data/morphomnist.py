"""MorphoMNIST++ generator + packed dataset loader (TODO item 3, scoped pass).

Built from scratch for this repo: raw digits come only from torchvision's
public MNIST download (no other local project's data/code is used or
referenced, per explicit instruction). Real thickness/intensity
perturbations (not synthetic placeholders) are applied directly to pixels
via grayscale morphology, with `intensity` causally dependent on
`thickness` -- matching the classic MorphoMNIST design and this
experiment's declared causal edge (see `configs/causal_graph_morpho.yaml`).

Every factor -- modeled (digit, thickness, intensity, hue) and
injected-unobserved (slant, rotation, scale, translation, background field,
texture seed) -- is sampled deterministically from a per-index seed and
**measured from the actual rendered pixels** where physically meaningful
(thickness, intensity), not just the requested target, so the logged
ground truth always matches what's on screen. `render()` is a pure
function of (base digit, factor dict) -> image, independent of
`sample_factors()`, so a factor record can be re-rendered exactly (Phase 1
gate: round-trip identity) and re-rendered *with one factor changed* (the
Phase 6 renderer cross-check, later).

Deferred (not built this pass, see EXP_NOTES.md): local swelling,
fractures, stroke-width modulation -- these need skeleton-based morphology,
not just dilation/erosion, and aren't load-bearing for the dataset/graph
setup gate.
"""
import colorsys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from skimage.morphology import dilation, erosion, disk
    from skimage.transform import AffineTransform, warp
except ImportError as e:  # pragma: no cover
    raise ImportError("MorphoMNIST++ generation needs scikit-image (pip install scikit-image)") from e

IMAGE_SIZE = 32  # 28 (raw MNIST) + 2px pad each side, matches CelebA-HQ pipeline's power-of-2-friendly sizing
PAD = 2

# Attribute order is fixed and shared by the dataset, the causal graph config, and the SCM.
MODELED_ATTRS = ["digit", "thickness", "intensity", "hue"]
UNOBSERVED_ATTRS = ["slant", "rotation", "scale", "translate_x", "translate_y",
                    "bg_freq", "bg_phase", "bg_amplitude", "texture_seed", "texture_amplitude"]
ATTRIBUTE_NAMES = MODELED_ATTRS + UNOBSERVED_ATTRS

# Priors for sampled (not measured) factors. Deliberately generous but bounded so digits stay
# recognizable -- these are modeling choices, not derived from any external source.
THICKNESS_TARGET_RANGE = (1.2, 5.5)          # px, average stroke half-width * 2
INTENSITY_BASE_RANGE = (140.0, 220.0)        # 0-255 mean foreground gray level before the thickness effect
# Gain/noise tuned empirically (not the first values tried): the achieved-pixel correlation is
# substantially weaker than the target-space correlation (measurement noise from the morphology
# ops attenuates it), so the target-space signal needs real headroom above "just detectable" for
# the *achieved* thickness/intensity correlation -- what CF1 eval will actually measure -- to be
# robustly negative. GAIN=6/NOISE=12 gave target corr -0.33 but achieved corr only ~-0.06; this
# setting gives target corr ~-0.76, achieved ~-0.67. See data/verify_morphomnist.py check 2.
INTENSITY_THICKNESS_GAIN = 22.0
INTENSITY_NOISE_STD = 5.0
SLANT_RANGE = (-25.0, 25.0)                  # degrees, shear
ROTATION_RANGE = (-20.0, 20.0)               # degrees
SCALE_RANGE = (0.85, 1.15)
TRANSLATE_RANGE = (-3.0, 3.0)                # px, post-pad
BG_AMPLITUDE_RANGE = (0.0, 18.0)              # 0-255 scale of the additive background field
TEXTURE_AMPLITUDE_RANGE = (0.0, 10.0)         # 0-255 scale of additive per-pixel noise


def measure_thickness(gray_u8: np.ndarray) -> float:
    """Average stroke width in px, via the distance transform on the binarized digit."""
    from scipy.ndimage import distance_transform_edt
    binary = gray_u8 > 20
    if not binary.any():
        return 0.0
    dist = distance_transform_edt(binary)
    return float(2.0 * dist[binary].mean())


def measure_intensity(gray_u8: np.ndarray) -> float:
    """Mean gray level over foreground (stroke) pixels only."""
    fg = gray_u8[gray_u8 > 20]
    return float(fg.mean()) if fg.size else 0.0


def _set_thickness(gray_u8: np.ndarray, target: float, max_iters: int = 6) -> np.ndarray:
    """Iteratively dilate/erode (grayscale, so anti-aliased edges are preserved) towards ``target``.

    Thin digits (thickness already near/below the low end of
    ``THICKNESS_TARGET_RANGE``) can vanish under even a single radius-1
    erosion. Guard against that explicitly: never apply a step that erases
    the digit, backing off to a smaller radius first and, failing that,
    accepting the current (non-erased) thickness rather than hitting an
    unreachable target. A vanished digit would make ``thickness``/
    ``intensity`` undefined and silently corrupt the dataset.
    """
    img = gray_u8.copy()
    for _ in range(max_iters):
        current = measure_thickness(img)
        if current <= 1e-6:
            break
        diff = target - current
        if abs(diff) < 0.15:
            break
        radius = max(1, round(abs(diff) / 2))
        candidate = None
        for r in range(radius, 0, -1):
            trial = dilation(img, disk(r)) if diff > 0 else erosion(img, disk(r))
            if measure_thickness(trial) > 1e-6:
                candidate = trial
                break
        if candidate is None:
            break  # even radius=1 would erase the digit; stop, keep current thickness
        img = candidate
    return img


def _set_intensity(gray_u8: np.ndarray, target: float) -> np.ndarray:
    """Rescale foreground pixel values so their mean hits ``target`` (background stays 0)."""
    fg_mask = gray_u8 > 20
    if not fg_mask.any():
        return gray_u8
    current = gray_u8[fg_mask].astype(np.float32).mean()
    if current < 1e-3:
        return gray_u8
    scale = float(np.clip(target / current, 0.2, 5.0))
    out = gray_u8.astype(np.float32)
    out[fg_mask] = np.clip(out[fg_mask] * scale, 1, 255)
    return out.astype(np.uint8)


def _apply_hue(gray_u8: np.ndarray, hue: float) -> np.ndarray:
    """Colorize a grayscale stroke image; black background is preserved (0 * color = 0)."""
    rgb_gain = np.array(colorsys.hsv_to_rgb(hue, 1.0, 1.0), dtype=np.float32)
    return np.clip(gray_u8[..., None].astype(np.float32) * rgb_gain[None, None, :], 0, 255).astype(np.uint8)


def _apply_geometry(rgb_u8: np.ndarray, slant: float, rotation: float, scale: float,
                    translate_x: float, translate_y: float) -> np.ndarray:
    """One combined affine warp (shear + rotate + scale + translate) to avoid multi-pass blur."""
    h, w = rgb_u8.shape[:2]
    center = np.array([w / 2.0, h / 2.0])
    shear = np.deg2rad(slant)
    theta = np.deg2rad(rotation)
    shear_mat = np.array([[1.0, np.tan(shear)], [0.0, 1.0]])
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mat2x2 = (rot_mat @ shear_mat) * scale
    # Build the full 3x3 by hand: rotate/shear/scale about center, then translate.
    full = np.eye(3)
    full[:2, :2] = mat2x2
    full[:2, 2] = center - mat2x2 @ center + np.array([translate_x, translate_y])
    tform = AffineTransform(matrix=full)
    out = np.stack([
        warp(rgb_u8[..., c], tform.inverse, order=1, mode="constant", cval=0.0, preserve_range=True)
        for c in range(rgb_u8.shape[2])
    ], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_background(rgb_u8: np.ndarray, freq: float, phase: float, amplitude: float) -> np.ndarray:
    """Low-frequency additive gradient field, added everywhere then reclipped (visible in background,
    subtle over the digit's ink -- matches 'background field' as a coarse-scale nuisance)."""
    if amplitude <= 1e-6:
        return rgb_u8
    h, w = rgb_u8.shape[:2]
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    field = amplitude * np.sin(2 * np.pi * freq * (xx + yy) + phase)
    return np.clip(rgb_u8.astype(np.float32) + field[..., None], 0, 255).astype(np.uint8)


def _apply_texture(rgb_u8: np.ndarray, seed: int, amplitude: float) -> np.ndarray:
    """Additive per-pixel noise from a deterministic seed -- regenerable from the logged seed alone."""
    if amplitude <= 1e-6:
        return rgb_u8
    rng = np.random.RandomState(int(seed))
    noise = rng.normal(0.0, amplitude, size=rgb_u8.shape[:2])
    return np.clip(rgb_u8.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)


@dataclass
class Factors:
    digit: int
    thickness: float
    intensity: float
    hue: float
    slant: float
    rotation: float
    scale: float
    translate_x: float
    translate_y: float
    bg_freq: float
    bg_phase: float
    bg_amplitude: float
    texture_seed: int
    texture_amplitude: float

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in ATTRIBUTE_NAMES], dtype=np.float32)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "Factors":
        kwargs = dict(zip(ATTRIBUTE_NAMES, vec.tolist()))
        kwargs["digit"] = int(round(kwargs["digit"]))
        kwargs["texture_seed"] = int(round(kwargs["texture_seed"]))
        return cls(**kwargs)


def sample_targets(index: int, digit: int) -> Dict[str, float]:
    """Deterministic per-index sampling of *requested* factor values (thickness/intensity are
    targets the renderer then measures and corrects for -- see ``render()``)."""
    rng = np.random.RandomState(seed=index)
    thickness_target = rng.uniform(*THICKNESS_TARGET_RANGE)
    intensity_base = rng.uniform(*INTENSITY_BASE_RANGE)
    intensity_noise = rng.normal(0.0, INTENSITY_NOISE_STD)
    intensity_target = intensity_base - INTENSITY_THICKNESS_GAIN * thickness_target + intensity_noise
    hue = rng.uniform(0.0, 1.0)
    slant = rng.uniform(*SLANT_RANGE)
    rotation = rng.uniform(*ROTATION_RANGE)
    scale = rng.uniform(*SCALE_RANGE)
    tx = rng.uniform(*TRANSLATE_RANGE)
    ty = rng.uniform(*TRANSLATE_RANGE)
    bg_freq = rng.uniform(1.0, 4.0)
    bg_phase = rng.uniform(0.0, 2 * np.pi)
    bg_amplitude = rng.uniform(*BG_AMPLITUDE_RANGE)
    texture_seed = int(rng.randint(0, 2 ** 31 - 1))
    texture_amplitude = rng.uniform(*TEXTURE_AMPLITUDE_RANGE)
    return dict(digit=digit, thickness=thickness_target, intensity=intensity_target, hue=hue, slant=slant,
               rotation=rotation, scale=scale, translate_x=tx, translate_y=ty, bg_freq=bg_freq,
               bg_phase=bg_phase, bg_amplitude=bg_amplitude, texture_seed=texture_seed,
               texture_amplitude=texture_amplitude)


def render(base_digit_u8: np.ndarray, factors: Factors, measure: bool = True) -> Tuple[np.ndarray, Factors]:
    """Pure function: (raw padded grayscale digit, factor record) -> RGB image.

    If ``measure`` is True, ``thickness``/``intensity`` in the returned
    ``Factors`` are the values actually achieved on the rendered pixels
    (may differ slightly from the input targets -- morphology doesn't hit
    an exact target every time). Pass ``measure=False`` only when
    re-rendering from an already-measured record (round-trip check).
    """
    gray = _set_thickness(base_digit_u8, factors.thickness)
    gray = _set_intensity(gray, factors.intensity)
    achieved_thickness = measure_thickness(gray) if measure else factors.thickness
    achieved_intensity = measure_intensity(gray) if measure else factors.intensity

    rgb = _apply_hue(gray, factors.hue)
    rgb = _apply_geometry(rgb, factors.slant, factors.rotation, factors.scale,
                          factors.translate_x, factors.translate_y)
    rgb = _apply_background(rgb, factors.bg_freq, factors.bg_phase, factors.bg_amplitude)
    rgb = _apply_texture(rgb, factors.texture_seed, factors.texture_amplitude)

    out_factors = Factors(**{**factors.__dict__, "thickness": achieved_thickness, "intensity": achieved_intensity})
    return rgb, out_factors


def pad_to_32(img28_u8: np.ndarray) -> np.ndarray:
    return np.pad(img28_u8, PAD, mode="constant", constant_values=0)


class MorphoMNISTPacked(Dataset):
    """Packed MorphoMNIST++ dataset -- mirrors CelebAHQPacked's ``{"img","attr"}`` contract."""

    def __init__(self, npz_path):
        arrays = np.load(npz_path, allow_pickle=True)
        self.images = arrays["images"]  # (N, 32, 32, 3) uint8
        self.attrs = arrays["attrs"]  # (N, len(attribute_names)) float32
        self.attribute_names = [str(x) for x in arrays["attribute_names"]]
        self.partitions = arrays["partitions"]  # (N,) int, 0=train 1=test

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        img = torch.from_numpy(self.images[index].copy()).permute(2, 0, 1).float().div_(127.5).sub_(1)
        return {"img": img, "index": index, "attr": torch.from_numpy(self.attrs[index].copy()),
               "partition": int(self.partitions[index])}
