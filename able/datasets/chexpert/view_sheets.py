"""Render the frozen CheXpert memmap in the protocol's own terms.

contact_sheet.py already draws random rows for section 7's integrity check.
These are different: they show the dataset organised by the variables the SCM
actually models, which is what you want to look at before trusting a
counterfactual on them.

  cells_24.png      one image per joint cell of (sex, age_bin, effusion, edema)
                    -- exactly the 24 cells the evaluation cohort is stratified
                    over, drawn from real cohort SOURCE rows so the panel is
                    the thing ABLE will be asked to intervene on.

  contrast_*.png    a finding held against its own absence with the OTHER
                    modeled attributes pinned. An unmatched grid confounds the
                    contrast: older patients have more effusion AND more
                    edema, so "effusion present vs absent" drawn at random is
                    partly an age comparison. Pinning makes the panel show the
                    finding and not the correlation.

Every caption is read from chexpert_meta.csv, so a row-alignment slip is
visible rather than merely absent from a checksum.
"""
import argparse, os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

AGE = {0: "<45", 1: "45-65", 2: ">65"}
SEX = {0: "F", 1: "M"}


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def sheet(arr, meta, rows, captions, cols, cell, title, out, pad=6, cap_h=34, title_h=46):
    n = len(rows)
    r = (n + cols - 1) // cols
    W = cols * (cell + pad) + pad
    H = title_h + r * (cell + cap_h + pad) + pad
    im = Image.new("RGB", (W, H), (14, 14, 16))
    d = ImageDraw.Draw(im)
    d.text((pad + 2, 12), title, fill=(235, 235, 240), font=font(20))
    for i, (row, cap) in enumerate(zip(rows, captions)):
        cy, cx = divmod(i, cols)
        x = pad + cx * (cell + pad)
        y = title_h + cy * (cell + cap_h + pad)
        g = np.asarray(arr[int(row)], dtype=np.uint8)
        p = Image.fromarray(g, "L").convert("RGB")
        if cell != p.size[0]:
            p = p.resize((cell, cell), Image.LANCZOS)
        im.paste(p, (x, y))
        for j, line in enumerate(cap.split("\n")):
            d.text((x + 2, y + cell + 2 + j * 15), line,
                   fill=(170, 200, 230) if j else (240, 240, 245), font=font(13 if j else 14))
    im.save(out)
    print(f"wrote {out}  ({im.size[0]}x{im.size[1]}, {n} panels)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/exouser/able_data/chexpert")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=int, default=224)
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    D = args.data
    meta = pd.read_csv(f"{D}/chexpert_meta.csv")
    split = np.load(f"{D}/split.npy")
    unc = pd.read_csv(f"{D}/label_uncertainty.csv")
    arr = np.load(f"{D}/images_224.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    certain = unc["uncertain_edema"].to_numpy() == 0

    # ---- 24 joint cells, from held-out certain-edema rows (the cohort's own pool)
    pool = meta.loc[(split == 1) & certain]
    rows, caps = [], []
    for s in (0, 1):
        for a in (0, 1, 2):
            for e in (0, 1):
                for d_ in (0, 1):
                    g = pool.index[(pool.sex == s) & (pool.age_bin == a)
                                   & (pool.effusion == e) & (pool.edema == d_)].to_numpy()
                    r = int(rng.choice(g))
                    rows.append(r)
                    caps.append(f"{SEX[s]}  age {AGE[a]}\n"
                                f"effusion {e}   edema {d_}")
    sheet(arr, meta, rows, caps, 8, args.cell,
          "CheXpert v2  |  the 24 cohort cells: (sex, age_bin, effusion, edema), one held-out source each",
          f"{args.out}/cells_24.png")

    # ---- matched contrasts: pin everything except the finding under test
    def contrast(var, other, pin, title, out, k=7):
        sel, cp = [], []
        for v in (0, 1):
            q = (pool[var] == v) & (pool[other] == pin["other"]) \
                & (pool.sex == pin["sex"]) & (pool.age_bin == pin["age_bin"])
            g = pool.index[q].to_numpy()
            take = rng.choice(g, min(k, len(g)), replace=False)
            for r in take:
                sel.append(int(r))
                cp.append(f"{var} = {v}\n{SEX[pin['sex']]} age {AGE[pin['age_bin']]}  {other} {pin['other']}")
        sheet(arr, meta, sel, cp, k, args.cell, title, out)

    contrast("effusion", "edema", {"other": 0, "sex": 1, "age_bin": 2},
             "CheXpert v2  |  do(effusion) target: absent (top) vs present (bottom), "
             "matched on sex=M, age>65, edema=0",
             f"{args.out}/contrast_effusion.png")
    contrast("edema", "effusion", {"other": 0, "sex": 1, "age_bin": 2},
             "CheXpert v2  |  do(edema) target: absent (top) vs present (bottom), "
             "matched on sex=M, age>65, effusion=0",
             f"{args.out}/contrast_edema.png")

    # ---- radiologist_val, the clean reference set
    rmeta = pd.read_csv(f"{D}/radiologist_val_meta.csv")
    rarr = np.load(f"{D}/radiologist_val_224.npy", mmap_mode="r")
    pick = np.sort(np.random.default_rng(args.seed + 5).choice(len(rmeta), 16, replace=False))
    rc = [f"{SEX[rmeta.at[i,'sex']]}  age {rmeta.at[i,'age_years']}\n"
          f"eff {rmeta.at[i,'effusion']}  edema {rmeta.at[i,'edema']}  "
          f"cm {rmeta.at[i,'cardiomegaly']}" for i in pick]
    sheet(rarr, rmeta, pick, rc, 8, args.cell,
          "radiologist_val  |  200 frontal images, labels read off the IMAGES by radiologists "
          "(the clean reference, 16 shown)",
          f"{args.out}/radiologist_val_16.png")


if __name__ == "__main__":
    main()
