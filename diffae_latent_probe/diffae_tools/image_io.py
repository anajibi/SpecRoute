from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import logging
import os

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from torchvision.utils import save_image

LOGGER = logging.getLogger(__name__)


@dataclass
class DiffAEAlignmentResult:
    success: bool
    original_path: str
    aligned_path: str
    strategy: str
    failure_reason: Optional[str] = None


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def list_image_files(folder: str | Path, recursive: bool = True):
    folder = Path(folder)
    if recursive:
        for path in sorted(folder.rglob("*")):
            if path.suffix in _IMAGE_EXTS and path.is_file():
                yield path
    else:
        for path in sorted(folder.iterdir()):
            if path.suffix in _IMAGE_EXTS and path.is_file():
                yield path


def sanitize_image_id(path: str | Path, base_dir: str | Path | None = None) -> str:
    path = Path(path)
    if base_dir is not None:
        try:
            rel = path.relative_to(base_dir)
        except Exception:
            rel = path.name
    else:
        rel = path.name
    rel = str(rel).replace(os.sep, "__")
    stem = Path(rel).stem
    return f"{stem}.png"


def load_pil_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def center_crop_resize(path: str | Path, output_size: int = 256) -> Image.Image:
    """Weak fallback preprocessing: square center crop + resize."""
    img = load_pil_image(path)
    img = ImageOps.fit(img, (output_size, output_size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    return img


def pil_to_tensor(img: Image.Image, normalize: bool = True) -> torch.Tensor:
    tensor = transforms.ToTensor()(img)
    if normalize:
        tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(tensor)
    return tensor


def tensor_to_pil(tensor: torch.Tensor, denormalize: bool = True) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().float()
    if denormalize:
        tensor = tensor.clamp(-1, 1)
        tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def load_image_tensor(path: str | Path, image_size: int = 256, normalize: bool = True) -> torch.Tensor:
    img = center_crop_resize(path, output_size=image_size)
    return pil_to_tensor(img, normalize=normalize)


def save_tensor_image(tensor: torch.Tensor, path: str | Path, denormalize: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = tensor_to_pil(tensor, denormalize=denormalize)
    img.save(path, format="PNG")


def official_ffhq_align(
    src_path: str | Path,
    dst_path: str | Path,
    repo_root: str | Path,
    output_size: int = 256,
) -> Tuple[bool, Optional[str]]:
    """Try to use the upstream DiffAE align.py implementation.

    Returns (success, failure_reason). This function only succeeds if dlib and a
    usable 68-point landmarks model are available.
    """
    repo_root = Path(repo_root)
    dst_path = Path(dst_path)
    try:
        import sys

        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from align import LandmarksDetector, image_align  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency path
        return False, f"official align.py unavailable: {exc}"

    landmarks_candidates = [
        repo_root / "shape_predictor_68_face_landmarks.dat",
        repo_root / "temp" / "shape_predictor_68_face_landmarks.dat",
        repo_root / "temp" / "shape_predictor_68_face_landmarks.dat.bz2",
    ]
    landmarks_path = None
    for cand in landmarks_candidates:
        if cand.exists() and cand.suffix != ".bz2":
            landmarks_path = cand
            break
    if landmarks_path is None:
        return False, "68-point landmark model not found; use fallback preprocessing"

    try:
        detector = LandmarksDetector(str(landmarks_path))
        src_path = str(src_path)
        for face_landmarks in detector.get_landmarks(src_path):
            image_align(src_path, str(dst_path), face_landmarks, output_size=output_size)
            return True, None
        return False, "no face landmarks detected"
    except Exception as exc:  # pragma: no cover - depends on dlib/image content
        return False, f"official alignment failed: {exc}"


def align_image_with_fallback(
    src_path: str | Path,
    dst_path: str | Path,
    repo_root: str | Path,
    output_size: int = 256,
    prefer_official: bool = True,
) -> DiffAEAlignmentResult:
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    reason = None

    if prefer_official:
        ok, reason = official_ffhq_align(src_path, dst_path, repo_root, output_size=output_size)
        if ok:
            return DiffAEAlignmentResult(True, str(src_path), str(dst_path), "official_align.py")
        LOGGER.warning("Official alignment unavailable for %s: %s", src_path, reason)

    img = center_crop_resize(src_path, output_size=output_size)
    img.save(dst_path, format="PNG")
    return DiffAEAlignmentResult(
        success=False,
        original_path=str(src_path),
        aligned_path=str(dst_path),
        strategy="center_crop_resize",
        failure_reason=reason if prefer_official else "official alignment disabled",
    )


def normalize_for_diffae(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.float()
    if tensor.max() > 1.5 or tensor.min() < -0.5:
        tensor = tensor / 255.0
    if tensor.min() >= 0.0 and tensor.max() <= 1.0:
        tensor = tensor * 2.0 - 1.0
    return tensor


def denormalize_from_diffae(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.float().clamp(-1, 1)
    return (tensor + 1.0) / 2.0


