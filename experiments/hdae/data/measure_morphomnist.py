"""Measure MorphoMNIST++ factors directly from a rendered image, blind to the
generator's logged ground truth -- i.e. what you could recover from a
counterfactual image you didn't generate yourself.

`digit` is explicitly out of scope: recognizing a digit from pixels needs a
trained classifier, not a closed-form measurement (mirrors how CelebA's FC
eval leans on `attr_classifier.py` for exactly this reason). Everything else
here is a real, reusable estimator, not a stub -- and this is the first
concrete piece of the "how do we measure unobserved factors on a generated
image" gap flagged in EXP_NOTES.md for Phase 5 (CF1 eval).

Reliability is genuinely uneven across factors, and that unevenness is
informative, not a defect to hide:
- thickness, intensity, hue: well-defined statistics of the foreground
  pixels. Directly measurable, though `measure_thickness` on a *transformed*
  image reflects thickness AFTER scale/rotation, not the pre-geometry value
  the generator logged -- a real, expected discrepancy (see module-level
  note in `morphomnist.py`), not noise.
- translate_x/y: foreground centroid offset from canvas center. Reliable
  for roughly symmetric digits; biased for asymmetric ones (a digit's own
  "weight" isn't centered on its bounding box).
- rotation, slant: recovered via the second-moment (PCA) principal axis of
  the foreground mask. This conflates the *applied* transform with the
  *digit's own shape asymmetry* -- a "7" has a diagonal stroke at zero
  applied rotation -- so it is systematically noisier than the others. Not
  a bug in the measurement code; a real limit of single-image estimation
  without a reference frame.
- scale: no reference frame exists in a single rendered image (nothing here
  knows the specific digit's own un-transformed size), so this is a proxy
  (sqrt of foreground pixel area, in px) rather than a direct estimate of
  the generator's multiplicative scale factor. Expect a real but loose
  relationship with ground truth, not a tight match.
- bg_freq/bg_phase/bg_amplitude, texture_amplitude: separated by spatial
  frequency -- a Gaussian blur of the background region isolates the
  low-frequency sinusoidal field; the high-frequency residual is attributed
  to texture noise. freq/phase are then fit to the field's known sinusoidal
  form by least squares.
"""
import colorsys
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from experiments.hdae.data.morphomnist import measure_intensity, measure_thickness

FOREGROUND_THRESHOLD = 20


def _foreground_mask(rgb_u8: np.ndarray) -> np.ndarray:
    return rgb_u8.max(axis=-1) > FOREGROUND_THRESHOLD


def measure_thickness_from_image(rgb_u8: np.ndarray) -> float:
    return measure_thickness(rgb_u8.max(axis=-1))


def measure_intensity_from_image(rgb_u8: np.ndarray) -> float:
    return measure_intensity(rgb_u8.max(axis=-1))


def measure_hue_from_image(rgb_u8: np.ndarray) -> float:
    """Circular mean hue over foreground pixels (hue wraps at 1.0, so a plain mean is wrong)."""
    mask = _foreground_mask(rgb_u8)
    fg = rgb_u8[mask].astype(np.float32) / 255.0
    if fg.size == 0:
        return 0.0
    hues = np.array([colorsys.rgb_to_hsv(*px)[0] for px in fg])
    angles = hues * 2 * np.pi
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return float((mean_angle / (2 * np.pi)) % 1.0)


def measure_translation_from_image(rgb_u8: np.ndarray) -> Tuple[float, float]:
    """Foreground centroid offset from canvas center, in px."""
    mask = _foreground_mask(rgb_u8)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, 0.0
    h, w = mask.shape
    return float(xs.mean() - w / 2.0), float(ys.mean() - h / 2.0)


def measure_orientation_from_image(rgb_u8: np.ndarray) -> float:
    """Second-moment (PCA) principal-axis angle in degrees, range (-90, 90].

    Conflates applied rotation/shear with the digit's own shape asymmetry --
    see module docstring. Returned as a single combined angle; there is no
    way to separately recover rotation vs. slant from one image.
    """
    mask = _foreground_mask(rgb_u8)
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return 0.0
    coords = np.stack([xs - xs.mean(), ys - ys.mean()], axis=0).astype(np.float64)
    cov = np.cov(coords)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    angle = float(np.degrees(np.arctan2(principal[1], principal[0])))
    if angle > 90:
        angle -= 180
    if angle <= -90:
        angle += 180
    return angle


def measure_scale_proxy_from_image(rgb_u8: np.ndarray) -> float:
    """sqrt(foreground pixel area), px -- a size proxy, not a calibrated scale-factor estimate."""
    return float(np.sqrt(_foreground_mask(rgb_u8).sum()))


def measure_background_from_image(rgb_u8: np.ndarray) -> Dict[str, float]:
    """Separate the low-frequency sinusoidal background field from high-frequency texture noise
    by Gaussian blur, then fit freq/phase/amplitude of the field by least squares."""
    mask = _foreground_mask(rgb_u8)
    bg = rgb_u8.max(axis=-1).astype(np.float32)
    if mask.all():
        return {"bg_freq": float("nan"), "bg_phase": float("nan"), "bg_amplitude": 0.0, "texture_amplitude": 0.0}
    fill_value = float(bg[~mask].mean())
    filled = np.where(mask, fill_value, bg)
    smooth = gaussian_filter(filled, sigma=2.0)
    residual = filled - smooth

    h, w = bg.shape
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    s = (xx + yy)[~mask]
    vals = smooth[~mask] - smooth[~mask].mean()
    bg_amplitude_est = float(smooth[~mask].std() * np.sqrt(2))

    def _model(s, freq, phase, amp):
        return amp * np.sin(2 * np.pi * freq * s + phase)

    try:
        popt, _ = curve_fit(_model, s, vals, p0=[2.0, 0.0, max(bg_amplitude_est, 1e-3)], maxfev=4000)
        freq_est, phase_est, _ = popt
        phase_est = float(phase_est % (2 * np.pi))
        freq_est = float(freq_est)
    except RuntimeError:
        freq_est, phase_est = float("nan"), float("nan")

    texture_amplitude_est = float(residual[~mask].std())
    return {"bg_freq": freq_est, "bg_phase": phase_est, "bg_amplitude": bg_amplitude_est,
           "texture_amplitude": texture_amplitude_est}


def measure_all(rgb_u8: np.ndarray) -> Dict[str, float]:
    """Everything measurable from a rendered image alone. Excludes `digit` (needs a classifier)
    and `texture_seed` (an arbitrary seed, not a physical quantity with a comparable "measured"
    notion)."""
    tx, ty = measure_translation_from_image(rgb_u8)
    out = {
        "thickness": measure_thickness_from_image(rgb_u8),
        "intensity": measure_intensity_from_image(rgb_u8),
        "hue": measure_hue_from_image(rgb_u8),
        "translate_x": tx,
        "translate_y": ty,
        "orientation": measure_orientation_from_image(rgb_u8),  # combines rotation + slant, see docstring
        "scale_proxy": measure_scale_proxy_from_image(rgb_u8),  # px, not directly comparable to ground-truth scale
    }
    out.update(measure_background_from_image(rgb_u8))
    return out
