"""CheXzero zero-shot on radiologist_val, as an independent off-the-shelf reader.

Requested as a third opinion on support_devices, which the XRV noise audit
could not reach (XRV has no Support Devices output at all).

TWO DEVIATIONS FROM CHEXZERO'S NATIVE PIPELINE, both forced, both stated:

 1. Input pixels. CheXzero's own preprocessing letterboxes the full radiograph
    onto a 320x320 canvas (resize longest side, paste centred, black bars) and
    then resizes to 224. radiologist_val_224.npy is resize-short-side-224 plus
    a 224 centre crop, so the periphery CheXzero would have kept is already
    gone. The original JPEGs cannot be recovered -- the 11.5 GB archive was
    deleted to make room for the image memmap. Feeding the existing 224 crops
    is the only option, and it has one genuine advantage: CheXzero and this
    project's classifier then see IDENTICAL pixels, so the comparison is
    controlled on input even though it is not CheXzero's native geometry.
    The centre crop is the direction that would hurt support_devices most --
    lines, tubes and pacemakers sit near the edges.

 2. Everything downstream is CheXzero's own: Normalize(101.48761, 83.43944) on
    the 0-255 scale over 3 repeated channels, CLIP ViT-B/32, the ("{}",
    "no {}") prompt pair, cosine similarity, softmax over the pair, and the
    10-checkpoint ensemble averaged.

So the number is gated before it is believed. CheXzero's published strength is
the five CheXpert competition pathologies; all five are scored here against the
same radiologist labels. If they land in a sane range the pipeline is sound
enough to read support_devices off it. If they collapse, deviation 1 broke it
and the support_devices number is not reportable. That is the same discipline
the XRV audit's S2 gate used.

Checkpoints are loaded with weights_only=True: a .pt is a pickle, and these came
off a Google Drive link.
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--data", default="/home/exouser/able_data/chexpert")
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

sys.path.insert(0, args.repo)
import clip                                     # noqa: E402  (repo-local)

MEAN, STD = 101.48761, 83.43944
PAIR = ("{}", "no {}")
# the five CheXpert competition pathologies CheXzero reports, as the gate,
# plus the label actually being asked about
GATE_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
TARGET = ["Support Devices", "Enlarged Cardiomediastinum"]
LABELS = GATE_LABELS + TARGET

dev = args.device if torch.cuda.is_available() else "cpu"

rmeta = pd.read_csv(f"{args.data}/radiologist_val_meta.csv")
valid = pd.read_csv(f"{args.data}/source_csv/valid.csv").set_index("Path")
Y = {}
for L in LABELS:
    v = pd.to_numeric(valid.loc[rmeta["path"].values, L], errors="coerce").to_numpy()
    assert np.isin(v, [0.0, 1.0]).all(), f"{L}: radiologist label is not clean 0/1"
    Y[L] = v.astype(np.int64)

arr = np.load(f"{args.data}/radiologist_val_224.npy", mmap_mode="r")
x = torch.from_numpy(np.asarray(arr, dtype=np.float32))          # (N,224,224), 0-255
x = ((x - MEAN) / STD)[:, None].repeat(1, 3, 1, 1)               # CheXzero's Normalize
print(f"images {tuple(x.shape)}  range [{x.min():.2f}, {x.max():.2f}]  device {dev}")


def text_weights(model, labels, template):
    with torch.no_grad():
        W = []
        for c in labels:
            t = clip.tokenize([template.format(c)], context_length=77).to(dev)
            e = model.encode_text(t)
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(dim=0)
            W.append(e / e.norm())
        return torch.stack(W, dim=1)


def image_feats(model, x, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            f = model.encode_image(x[i:i + bs].to(dev))
            out.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(out)


paths = sorted(glob.glob(os.path.join(args.weights, "*.pt")))
print(f"{len(paths)} checkpoints")
preds = []
for p in paths:
    model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
    sd = torch.load(p, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model = model.to(dev).eval()
    F = image_feats(model, x)
    pos = (F @ text_weights(model, LABELS, PAIR[0])).float().cpu().numpy()
    neg = (F @ text_weights(model, LABELS, PAIR[1])).float().cpu().numpy()
    y = np.exp(pos) / (np.exp(pos) + np.exp(neg))
    preds.append(y)
    print(f"  {os.path.basename(p)}  done")
    del model
    torch.cuda.empty_cache()

P = np.mean(preds, axis=0)
np.save(f"{args.out}/chexzero_radval_probs.npy", P)


def boot(y, p, n=2000, seed=7):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return None, None, None
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            v.append(roc_auc_score(y[i], p[i]))
    return (round(float(roc_auc_score(y, p)), 4),
            round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4))


res = {}
print(f"\n{'label':30s} {'AUC':>7s}  {'95% CI':>18s}  {'n_pos':>5s}  role")
for j, L in enumerate(LABELS):
    a, lo, hi = boot(Y[L], P[:, j])
    role = "GATE" if L in GATE_LABELS else "target"
    res[L] = {"auc": a, "ci95": [lo, hi], "n": int(len(Y[L])),
              "n_pos": int(Y[L].sum()), "role": role}
    print(f"{L:30s} {a:7.4f}  [{lo:.4f}, {hi:.4f}]  {int(Y[L].sum()):5d}  {role}")

g = [res[L]["auc"] for L in GATE_LABELS]
rec = {
    "reader": "CheXzero (rajpurkarlab/CheXzero), zero-shot, 10-checkpoint ensemble",
    "prompt_pair": list(PAIR),
    "set": "radiologist_val (200 frontal images, radiologist-labelled)",
    "n_checkpoints": len(paths),
    "checkpoints": [os.path.basename(p) for p in paths],
    "input_deviation": ("scored on radiologist_val_224.npy (resize-short-side-224 + 224 "
                        "centre crop), not CheXzero's letterbox-to-320-then-224. The source "
                        "JPEGs are gone with the deleted archive. Identical pixels to this "
                        "project's own classifier, so the comparison is controlled on input."),
    "gate": {"labels": GATE_LABELS, "mean_auc": round(float(np.mean(g)), 4),
             "min_auc": round(float(np.min(g)), 4),
             "published_chexzero_range_on_chexpert": "roughly .82-.92 on these five",
             "interpretation": "if min_auc is far below ~.70 the input deviation broke the "
                               "pipeline and the target numbers are not reportable"},
    "labels": res,
}
json.dump(rec, open(f"{args.out}/chexzero_radval.json", "w"), indent=2)
print(f"\ngate mean {rec['gate']['mean_auc']}  min {rec['gate']['min_auc']}")
print(f"wrote {args.out}/chexzero_radval.json")
