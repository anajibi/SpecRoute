"""Section 7's image-integrity check: render random rows for a human to look at.

A dead-pixel test catches a constant row; it does not catch an image that
decoded into noise, got cropped off the chest, or landed on the wrong meta
row. That needs eyes, so this writes a labelled contact sheet -- the caption
under each panel is read straight from chexpert_meta.csv, which makes a
row-alignment slip visible rather than merely absent from a checksum.
"""
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--cell", type=int, default=224)
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()

    meta = pd.read_csv(args.meta)
    arr = np.load(args.npy, mmap_mode="r")
    assert len(arr) == len(meta), f"{len(arr)} images vs {len(meta)} meta rows"
    rows = np.sort(np.random.default_rng(args.seed).choice(len(meta), args.n, replace=False))

    cols = args.cols
    nrow = (args.n + cols - 1) // cols
    cell, cap, gap = args.cell, 34, 6
    W = cols * (cell + gap) + gap
    H = nrow * (cell + cap + gap) + gap
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    f = font(10)

    for i, r in enumerate(rows):
        cx, cy = i % cols, i // cols
        x = gap + cx * (cell + gap)
        y = gap + cy * (cell + cap + gap)
        im = Image.fromarray(np.asarray(arr[r])).convert("RGB")
        if im.size != (cell, cell):
            im = im.resize((cell, cell), Image.NEAREST)
        sheet.paste(im, (x, y))
        m = meta.iloc[r]
        d.text((x, y + cell + 1), f"row {r}  pat {m.patient_id} study {m.study_id}", font=f, fill="black")
        d.text((x, y + cell + 12), f"{'M' if m.sex else 'F'} age {m.age_years} (bin {m.age_bin})"
                                   f"  {'AP' if not m.ap_pa else 'PA'}", font=f, fill="black")
        tags = [k for k in ("effusion", "cardiomegaly", "edema", "lung_opacity", "consolidation",
                            "atelectasis", "pneumothorax", "support_devices") if m[k]]
        # measure rather than guess a character count: a long tag list used to
        # run straight across the neighbouring panel's caption
        txt = "+" + ",".join(tags) if tags else "no positive label"
        while len(txt) > 4 and d.textlength(txt, font=f) > cell:
            txt = txt[:-2] + "\u2026" if not txt.endswith("\u2026") else txt[:-2] + "\u2026"
        d.text((x, y + cell + 23), txt, font=f, fill=(140, 0, 0) if tags else (110, 110, 110))

    sheet.save(args.out)
    print(f"wrote {args.out}  rows {rows.tolist()}")


if __name__ == "__main__":
    main()
