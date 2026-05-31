from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw


def _load_tile(path: str | Path, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size))


def _label_tile(text: str, width: int, height: int = 24) -> Image.Image:
    tile = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(tile)
    draw.text((4, 4), text, fill="black")
    return tile


def save_guidance_grid(
    original_path: str | Path,
    reconstruction_path: str | Path,
    edited_records: Sequence[dict[str, object]],
    guidance_scales: Sequence[float],
    output_path: str | Path,
    tile_size: int = 256,
) -> Path:
    cols = len(guidance_scales)
    rows = 3
    label_h = 24
    grid = Image.new("RGB", (cols * tile_size, rows * (tile_size + label_h)), "white")
    original = _load_tile(original_path, tile_size)
    recon = _load_tile(reconstruction_path, tile_size)
    by_scale = {float(record["guidance_scale"]): record for record in edited_records}
    for col, scale in enumerate(guidance_scales):
        x = col * tile_size
        grid.paste(_label_tile(f"original | scale={scale}", tile_size, label_h), (x, 0))
        grid.paste(original, (x, label_h))
        y = tile_size + label_h
        grid.paste(_label_tile("DDIM reconstruction", tile_size, label_h), (x, y))
        grid.paste(recon, (x, y + label_h))
        y = 2 * (tile_size + label_h)
        if float(scale) in by_scale:
            edited = _load_tile(str(by_scale[float(scale)]["edited_path"]), tile_size)
            grid.paste(_label_tile(f"edited | scale={scale}", tile_size, label_h), (x, y))
            grid.paste(edited, (x, y + label_h))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path
