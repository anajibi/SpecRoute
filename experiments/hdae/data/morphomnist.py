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
texture seed) -- is sampled deterministically from a per-index seed, from a
distribution declared per-attribute in `configs/morphomnist_factors.yaml`
(see `load_factor_config`/`sample_from_spec`), and **measured from the
actual rendered pixels** where physically meaningful (thickness,
intensity), not just the requested target, so the logged ground truth
always matches what's on screen. `render()` is a pure function of (base
digit, factor record) -> image, independent of `sample_targets()`, so a
factor record can be re-rendered exactly (Phase 1 gate: round-trip
identity) and re-rendered *with one factor changed* (the Phase 6 renderer
cross-check, later).

Canvas is `image.digit_size + 2*image.pad` (default 28+2*18=64) -- the
padding matters beyond framing: geometric transforms (rotation/shear/scale/
translation) need real margin around the digit to avoid clipping content at
the canvas edge (see `render()`), and a bigger canvas is what makes larger,
more visually distinct transform ranges possible at all.

Deferred (not built this pass, see EXP_NOTES.md): local swelling,
fractures, stroke-width modulation -- these need skeleton-based morphology,
not just dilation/erosion, and aren't load-bearing for the dataset/graph
setup gate.
"""
import colorsys
from dataclasses import dataclass
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

try:
    from skimage.morphology import dilation, erosion, disk
    from skimage.transform import AffineTransform, warp
except ImportError as e:  # pragma: no cover
    raise ImportError("MorphoMNIST++ generation needs scikit-image (pip install scikit-image)") from e

DEFAULT_FACTOR_CONFIG_PATH = "experiments/hdae/configs/morphomnist_factors.yaml"

# Attribute order is fixed and shared by the dataset, the causal graph config, and the SCM.
MODELED_ATTRS = ["digit", "thickness", "intensity", "hue"]
# slant and texture_seed were DROPPED (2026-08-28). Both are still rendered -- slant at a
# constant 0, texture at a constant seed -- they are simply no longer stored or treated as
# attributes, because neither carries usable information:
#   slant        was already constant 0.0 in every one of the 70,000 images. A constant column
#                is dead weight in every predictor, every FC pool, and every variance analysis.
#   texture_seed is a per-image random INTEGER SEED, not a physical quantity. Its effect on
#                pixels is pseudo-random by construction, so no model can predict it and no
#                amount of extra data makes it learnable -- it is pure irreducible noise that
#                only widens the error bars on everything else. Fixing the seed makes the
#                texture pattern deterministic, so texture_amplitude stays a real, learnable
#                nuisance factor while the unpredictable part goes away.
UNOBSERVED_ATTRS = ["rotation", "scale", "translate_x", "translate_y",
                    "bg_freq", "bg_phase", "bg_amplitude", "texture_amplitude"]

# The fixed texture pattern seed. Any constant works; it is recorded here so the rendering is
# reproducible and so nobody mistakes the constant for a leftover per-image value.
FIXED_TEXTURE_SEED = 20260828
ATTRIBUTE_NAMES = MODELED_ATTRS + UNOBSERVED_ATTRS


def load_factor_config(path: str = DEFAULT_FACTOR_CONFIG_PATH) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    img = config["image"]
    expected = img["digit_size"] + 2 * img["pad"]
    if img["canvas_size"] != expected:
        raise ValueError(f"{path}: canvas_size={img['canvas_size']} must equal "
                         f"digit_size + 2*pad = {expected}")
    required = [n for n in UNOBSERVED_ATTRS if n != "texture_seed"] + \
        ["thickness", "intensity_noise", "hue"]
    missing = [n for n in required if n not in config["factors"]]
    if missing:
        raise ValueError(f"{path}: missing factor specs for {missing}")
    missing_formula = [k for k in ("scale", "noise_coef", "thickness_coef", "bias", "floor")
                       if k not in config.get("intensity_formula", {})]
    if missing_formula:
        raise ValueError(f"{path}: missing intensity_formula.{{{','.join(missing_formula)}}}")
    return config


def sample_from_spec(rng: np.random.RandomState, spec: dict) -> float:
    """`spec["offset"]`, if present, is added to the sampled value regardless of distribution --
    e.g. thickness's `t := 0.5 + Gamma(10,5)` (Pawlowski et al. 2020) is `distribution: gamma` with
    `offset: 0.5`, not a distribution of its own."""
    dist = spec["distribution"]
    p = spec.get("params", {})
    if dist == "uniform":
        val = rng.uniform(p["low"], p["high"])
    elif dist == "normal":
        val = rng.normal(p.get("mean", 0.0), p.get("std", 1.0))
    elif dist == "loguniform":
        val = np.exp(rng.uniform(np.log(p["low"]), np.log(p["high"])))
    elif dist == "constant":
        val = p["value"]
    elif dist == "gamma":
        # numpy's gamma() takes (shape, scale); the literature convention here is (shape, rate),
        # scale = 1/rate.
        val = rng.gamma(p["shape"], scale=1.0 / p["rate"])
    elif dist == "discrete_bins":
        # Uniform over `num_bins` fixed bin-center values spanning [low, high) -- not a continuous
        # uniform. Used so a downstream classifier's bin edges land exactly on generated values
        # (no residual intra-class variation from a value sampled anywhere within its bin).
        num_bins = int(p["num_bins"])
        centers = p["low"] + (np.arange(num_bins) + 0.5) * (p["high"] - p["low"]) / num_bins
        val = rng.choice(centers)
    else:
        raise ValueError(f"unknown distribution {dist!r} "
                         "(expected uniform/normal/loguniform/constant/gamma/discrete_bins)")
    return float(val + spec.get("offset", 0.0))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


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


def _set_thickness(gray_u8: np.ndarray, target: float, max_iters: int = 6,
                   min_pixel_fraction: float = 0.6) -> np.ndarray:
    """Iteratively dilate/erode (grayscale, so anti-aliased edges are preserved) towards ``target``.

    Two failure modes guarded against, both found by visual review, not just
    by thickness reaching zero:
    - Thin digits can vanish entirely under even a single radius-1 erosion.
    - Short of total vanishing, erosion can strip a digit down to a few
      residual fragments (measurable, nonzero "thickness", but visually
      destroyed) -- e.g. a recognizable "9" eroded to ~10% of its original
      ink. Guard on *foreground pixel count*, not just thickness>0: never
      apply a step that drops below ``min_pixel_fraction`` of the original
      foreground pixel count, backing off to a smaller radius first and,
      failing that, accepting the current thickness over an unreachable target.
    """
    img = gray_u8.copy()
    original_fg = int((gray_u8 > 20).sum())
    min_fg = max(1, int(min_pixel_fraction * original_fg))
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
            if measure_thickness(trial) > 1e-6 and int((trial > 20).sum()) >= min_fg:
                candidate = trial
                break
        if candidate is None:
            break  # even radius=1 would erase/gut the digit; stop, keep current thickness
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
    rotation: float
    scale: float
    translate_x: float
    translate_y: float
    bg_freq: float
    bg_phase: float
    bg_amplitude: float
    texture_amplitude: float
    # Rendered but NOT stored -- see UNOBSERVED_ATTRS. Defaults make from_vector() work on the
    # 12-column records without either of them being present.
    slant: float = 0.0
    texture_seed: int = FIXED_TEXTURE_SEED
    # Not part of ATTRIBUTE_NAMES / the logged record -- generation-time-only so render() can
    # compute intensity's target from thickness's *achieved* value (see render()'s docstring for
    # why: the requested thickness target and what morphology actually achieves can diverge, and
    # computing intensity from the request rather than the outcome breaks the causal link for
    # every image where they diverge). This is `eps_I` in Pawlowski et al. 2020's SCM formula.
    intensity_noise: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in ATTRIBUTE_NAMES], dtype=np.float32)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "Factors":
        kwargs = dict(zip(ATTRIBUTE_NAMES, vec.tolist()))
        kwargs["digit"] = int(round(kwargs["digit"]))
        if "texture_seed" in kwargs:
            kwargs["texture_seed"] = int(round(kwargs["texture_seed"]))
        return cls(**kwargs)


def sample_targets(index: int, digit: int, config: dict) -> Dict[str, float]:
    """Deterministic per-index sampling of *requested* factor values (thickness/intensity are
    targets the renderer then measures and corrects for -- see ``render()``), per the
    distributions declared in ``config`` (``load_factor_config``).

    thickness/intensity follow Pawlowski, Castro & Glocker 2020 ("Deep Structural Causal Models
    for Tractable Counterfactual Inference"): t := 0.5 + Gamma(10,5), i := 191*sigmoid(0.5*eps_I +
    2*t - 5) + 64, eps_I ~ N(0,1). `intensity_noise` sampled here is eps_I; the actual intensity
    formula runs in render() against thickness's *achieved* value, not this requested one."""
    rng = np.random.RandomState(seed=index)
    f = config["factors"]
    thickness_target = sample_from_spec(rng, f["thickness"])
    intensity_noise = sample_from_spec(rng, f["intensity_noise"])
    hue = sample_from_spec(rng, f["hue"])
    slant = sample_from_spec(rng, f["slant"])
    rotation = sample_from_spec(rng, f["rotation"])
    scale = sample_from_spec(rng, f["scale"])
    tx = sample_from_spec(rng, f["translate_x"])
    ty = sample_from_spec(rng, f["translate_y"])
    bg_freq = sample_from_spec(rng, f["bg_freq"])
    bg_phase = sample_from_spec(rng, f["bg_phase"])
    bg_amplitude = sample_from_spec(rng, f["bg_amplitude"])
    rng.randint(0, 2 ** 31 - 1)          # draw kept so the RNG stream is unchanged from the
                                         # previous build; only its USE is dropped
    texture_seed = FIXED_TEXTURE_SEED    # deterministic texture pattern -- see UNOBSERVED_ATTRS
    texture_amplitude = sample_from_spec(rng, f["texture_amplitude"])
    return dict(digit=digit, thickness=thickness_target, intensity=0.0, hue=hue, slant=slant,
               rotation=rotation, scale=scale, translate_x=tx, translate_y=ty, bg_freq=bg_freq,
               bg_phase=bg_phase, bg_amplitude=bg_amplitude, texture_seed=texture_seed,
               texture_amplitude=texture_amplitude, intensity_noise=intensity_noise)


def render(base_digit_u8: np.ndarray, factors: Factors, config: dict, measure: bool = True
          ) -> Tuple[np.ndarray, Factors]:
    """Pure function: (raw padded grayscale digit, factor record, factor config) -> RGB image.

    If ``measure`` is True, ``thickness``/``intensity`` in the returned
    ``Factors`` are the values actually achieved on the rendered pixels
    (may differ slightly from the input targets -- morphology doesn't hit
    an exact target every time). Pass ``measure=False`` only when
    re-rendering from an already-measured record (round-trip check) --
    ``config`` is unused in that path (kept for a uniform call signature).

    When ``measure`` is True, intensity's target is computed from thickness's
    *achieved* value (Pawlowski et al. 2020's formula, via ``factors.intensity_noise``
    and ``config["intensity_formula"]``), not the originally requested
    ``factors.thickness`` -- morphology's vanish-guard (``_set_thickness``)
    can leave the achieved value well short of a hard-to-reach request, and
    computing intensity from the request rather than the outcome breaks the
    causal link on every image where they diverge.
    """
    gray = _set_thickness(base_digit_u8, factors.thickness)
    achieved_thickness = measure_thickness(gray) if measure else factors.thickness
    if measure:
        ifm = config["intensity_formula"]
        z = ifm["noise_coef"] * factors.intensity_noise + ifm["thickness_coef"] * achieved_thickness + ifm["bias"]
        intensity_target = ifm["scale"] * _sigmoid(z) + ifm["floor"]
    else:
        intensity_target = factors.intensity
    gray = _set_intensity(gray, intensity_target)
    achieved_intensity = measure_intensity(gray) if measure else factors.intensity

    rgb = _apply_hue(gray, factors.hue)
    # Geometry needs margin: a tight canvas around the digit makes rotation/shear/scale/
    # translation clip real digit content at the array edge (mode="constant" fills the clipped
    # area with background, silently deleting strokes -- see TODO-List item 3's bug log). Pad
    # out, warp on the padded canvas where there's real headroom, then crop back to the original
    # size. Margin scales with canvas size so it stays generous for wider transform ranges.
    canvas = rgb.shape[0]
    margin = max(16, canvas // 2)
    rgb_padded = np.pad(rgb, ((margin, margin), (margin, margin), (0, 0)), mode="constant", constant_values=0)
    rgb_padded = _apply_geometry(rgb_padded, factors.slant, factors.rotation, factors.scale,
                                 factors.translate_x, factors.translate_y)
    rgb = rgb_padded[margin:-margin, margin:-margin]
    rgb = _apply_background(rgb, factors.bg_freq, factors.bg_phase, factors.bg_amplitude)
    rgb = _apply_texture(rgb, factors.texture_seed, factors.texture_amplitude)

    out_factors = Factors(**{**factors.__dict__, "thickness": achieved_thickness, "intensity": achieved_intensity})
    return rgb, out_factors


def pad_digit(img28_u8: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(img28_u8, pad, mode="constant", constant_values=0)


class MorphoMNISTPacked(Dataset):
    """Packed MorphoMNIST++ dataset -- mirrors CelebAHQPacked's ``{"img","attr"}`` contract.

    Backed by HDF5, opened lazily per-process by default (see ``__getstate__``/``_open``) --
    an ``h5py.File`` handle inherited across a DataLoader ``fork`` produces
    silent garbage reads, not an error, so it must never be opened in
    ``__init__`` and pickled into worker processes. Same fork-safety pattern
    as ``celeba_hq.py``'s LMDB env.

    ``attrs``/``attribute_names``/``partitions`` are small (a few hundred KB
    total) and loaded eagerly into memory regardless.

    ``preload_images=True`` reads the entire (~860MB decompressed) image
    array into RAM up front instead of decompressing per-item lzf chunks on
    every access -- fine for this dataset's size, and the right choice when
    training many small independent networks against it concurrently (one
    dataset copy per process; per-item access becomes a plain slice, no
    per-worker HDF5/lzf overhead, no DataLoader worker fan-out needed).
    Leave it False for anything closer to CelebA-HQ scale.
    """

    def __init__(self, h5_path, preload_images: bool = False):
        self.h5_path = str(h5_path)
        self._file = None
        self._preloaded = None
        with h5py.File(self.h5_path, "r") as f:
            self.attrs = f["attrs"][:]  # (N, len(attribute_names)) float32
            self.attribute_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                                    for x in f["attribute_names"][:]]
            self.partitions = f["partitions"][:]  # (N,) int, 0=train 1=test
            self._n = f["images"].shape[0]
            if preload_images:
                self._preloaded = f["images"][:]

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return self._n

    def __getitem__(self, index):
        image = self._preloaded[index] if self._preloaded is not None else self._open()["images"][index]
        img = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1)
        return {"img": img, "index": index, "attr": torch.from_numpy(self.attrs[index].copy()),
               "partition": int(self.partitions[index])}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state
