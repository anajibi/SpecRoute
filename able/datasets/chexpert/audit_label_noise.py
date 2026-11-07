"""CheXpert label-noise audit: report-derived labels vs an independent reader.

The question is narrow and worth stating precisely. CheXpert's training labels
were produced by a rule-based labeller run over free-text radiology REPORTS,
not by anyone looking at the images. So a "negative" can mean the finding was
absent, or that the radiologist simply did not mention it. The 234-image valid
set and the 668-image test set were labelled by radiologists reading the
IMAGES, and those are the clean references.

The reader is TorchXRayVision's `densenet121-res224-all`, trained on NIH +
PadChest + CheXpert + MIMIC + RSNA. `densenet121-res224-chex` is deliberately
NOT used for the audit: it learned the same report labels, so agreeing with
them is not evidence about them. It is run once at the end as a control --
if report-label noise is real, the chex model should score HIGHER against the
report labels and no better against the radiologists.

Nothing here trains anything.

Stages: s2 (gate on valid), s4 (train sweep), report (S5/S6).
"""
import argparse, io, json, os, sys, time, zipfile
import numpy as np
import pandas as pd
import skimage.io
import torch
import torchxrayvision as xrv
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

PREFIX = "CheXpert-v1.0-small/"

# CSV column -> XRV output name. Support Devices has no XRV counterpart.
LABEL_MAP = {
    "Pleural Effusion": "Effusion",
    "Cardiomegaly": "Cardiomegaly",
    "Edema": "Edema",
    "Consolidation": "Consolidation",
    "Lung Opacity": "Lung Opacity",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Lung Lesion": "Lung Lesion",
    "Fracture": "Fracture",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
}
NOT_AUDITABLE = ["Support Devices"]

# S2 gate: published XRV-all ranges on CheXpert sit around 0.80-0.93
GATE = {"Effusion": 0.85, "Cardiomegaly": 0.80, "Edema": 0.85,
        "Atelectasis": 0.75, "Consolidation": 0.80, "Pneumothorax": 0.80}


def xrv_preprocess(data):
    """Exactly XRV's own pipeline. Getting this wrong silently ruins every AUC."""
    img = skimage.io.imread(io.BytesIO(data))
    if img.ndim == 3:
        img = img.mean(2)
    img = xrv.datasets.normalize(img, 255)          # uint8 -> [-1024, 1024]
    img = img[None, ...]
    img = xrv.datasets.XRayCenterCrop()(img)
    img = xrv.datasets.XRayResizer(224)(img)
    return img.astype(np.float32)


class ZipXray(Dataset):
    def __init__(self, zip_path, members):
        self.zip_path, self.members, self.z = zip_path, members, None

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        if self.z is None:
            self.z = zipfile.ZipFile(self.zip_path)
        with self.z.open(self.members[i]) as f:
            data = f.read()
        return torch.from_numpy(xrv_preprocess(data)), i


def boot_auc(y, p, n=1000, seed=0):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return None, None, None, int(len(y))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(roc_auc_score(y, p)), 4), round(float(lo), 4), round(float(hi), 4), int(len(y))


@torch.no_grad()
def sweep(zip_path, members, weights, dev, workers, batch, age=False):
    model = xrv.models.DenseNet(weights=weights).to(dev).eval()
    agem = xrv.baseline_models.riken.AgeModel().to(dev).eval() if age else None
    dl = DataLoader(ZipXray(zip_path, members), batch_size=batch, num_workers=workers,
                    pin_memory=True, shuffle=False)
    P = np.zeros((len(members), len(model.pathologies)), np.float32)
    A = np.zeros(len(members), np.float32) if age else None
    t0, done = time.time(), 0
    for x, idx in dl:
        x = x.to(dev, non_blocking=True)
        P[idx.numpy()] = model(x).cpu().numpy()
        if agem is not None:
            A[idx.numpy()] = agem(x).cpu().numpy().ravel()
        done += len(idx)
        if done % 20000 < batch:
            el = time.time() - t0
            print(f"  {done}/{len(members)}  {done/el:.0f} img/s  "
                  f"eta {(len(members)-done)/max(done/el,1)/60:.1f} min", flush=True)
    print(f"  swept {done} in {(time.time()-t0)/60:.1f} min")
    return P, A, list(model.pathologies)


def members_of(df):
    bad = [p for p in df["Path"] if not p.startswith(PREFIX)]
    if bad:
        raise SystemExit(f"{len(bad)} paths lack {PREFIX!r}")
    return [p[len(PREFIX):] for p in df["Path"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["s1", "s2", "s4", "report"])
    ap.add_argument("--zip", default="/home/exouser/able_data/chexpert_src/chexpert.zip")
    ap.add_argument("--csv-dir", default="/home/exouser/able_data/chexpert/source_csv")
    ap.add_argument("--out", default="/home/exouser/SpecRoute/outputs/chexpert_noise")
    ap.add_argument("--weights", default="densenet121-res224-all")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag or ("" if args.weights == "densenet121-res224-all" else "_chex")

    # ---------------- S0 inputs
    train = pd.read_csv(f"{args.csv_dir}/train.csv")
    valid = pd.read_csv(f"{args.csv_dir}/valid.csv")
    with zipfile.ZipFile(args.zip) as z:
        have = set(z.namelist())
    miss_t = [m for m in members_of(train) if m not in have]
    miss_v = [m for m in members_of(valid) if m not in have]
    s0 = {"len_train": len(train), "len_valid": len(valid),
          "train_images_missing": len(miss_t), "valid_images_missing": len(miss_v),
          "pass": len(train) == 223414 and len(valid) == 234 and not miss_t and not miss_v}
    print(f"S0: train {len(train)} valid {len(valid)} missing {len(miss_t)}/{len(miss_v)} "
          f"-> {'PASS' if s0['pass'] else 'FAIL'}")
    if not s0["pass"]:
        raise SystemExit("S0 failed")

    # ---------------- S1 preprocessing check
    if args.stage == "s1":
        with zipfile.ZipFile(args.zip) as z:
            m = members_of(valid)[0]
            img = xrv_preprocess(z.read(m))
        ok = img.shape == (1, 224, 224) and img.min() >= -1024 and img.max() <= 1024
        print(f"S1: {m}\n    shape {img.shape}  range [{img.min():.1f}, {img.max():.1f}]  "
              f"dtype {img.dtype} -> {'PASS' if ok else 'FAIL'}")
        model = xrv.models.DenseNet(weights=args.weights)
        print(f"    model pathologies ({len(model.pathologies)}): {list(model.pathologies)}")
        unmapped = [v for v in LABEL_MAP.values() if v not in model.pathologies]
        print(f"    unmapped label targets: {unmapped or 'none'}")
        json.dump({"S0": s0, "S1": {"member": m, "shape": list(img.shape),
                                    "range": [float(img.min()), float(img.max())],
                                    "pathologies": list(model.pathologies),
                                    "unmapped": unmapped, "pass": bool(ok and not unmapped)}},
                  open(f"{args.out}/s0_s1.json", "w"), indent=2)
        if not ok or unmapped:
            raise SystemExit("S1 failed")
        return

    # ---------------- S2 gate on valid (radiologist labels)
    if args.stage == "s2":
        P, _, paths = sweep(args.zip, members_of(valid), args.weights, dev, args.workers, args.batch)
        np.save(f"{args.out}/valid_xrv_probs{tag}.npy", P)
        col = {k: paths.index(v) for k, v in LABEL_MAP.items()}
        res, fails = {}, []
        for csvcol, xname in LABEL_MAP.items():
            y = pd.to_numeric(valid[csvcol], errors="coerce").to_numpy()
            keep = np.isin(y, [0, 1])
            a, lo, hi, n = boot_auc(y[keep], P[keep, col[csvcol]])
            g = GATE.get(xname)
            ok = None if g is None else (a is not None and a >= g)
            res[xname] = {"csv_column": csvcol, "auc": a, "ci95": [lo, hi], "n": n,
                          "n_pos": int((y[keep] == 1).sum()), "gate": g, "pass": ok}
            if ok is False:
                fails.append(xname)
            print(f"  {xname:28s} AUC {a} [{lo},{hi}]  n={n} pos={int((y[keep]==1).sum())}"
                  + (f"  gate {g} {'PASS' if ok else 'FAIL'}" if g else ""))
        json.dump({"weights": args.weights, "set": "valid(234)", "labels": res,
                   "failed_gate": fails, "pass": not fails},
                  open(f"{args.out}/s2_gate{tag}.json", "w"), indent=2)
        print(f"\nS2 {'PASS' if not fails else 'FAIL: ' + str(fails)} -> {args.out}/s2_gate{tag}.json")
        if fails:
            raise SystemExit("S2 gate failed: fix the pipeline (normalisation, crop, "
                             "channel order, label alignment) before S4")
        return

    # ---------------- S4 training sweep, frontal only
    if args.stage == "s4":
        fr = train[train["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
        print(f"S4: {len(fr)} frontal training images")
        P, A, paths = sweep(args.zip, members_of(fr), args.weights, dev, args.workers,
                            args.batch, age=(tag == ""))
        np.save(f"{args.out}/train_xrv_probs{tag}.npy", P)
        fr[["Path"]].to_csv(f"{args.out}/train_xrv_rows{tag}.csv", index=False)
        if A is not None:
            np.save(f"{args.out}/train_xrv_age{tag}.npy", A)
        json.dump({"weights": args.weights, "n": len(fr), "pathologies": paths},
                  open(f"{args.out}/s4_meta{tag}.json", "w"), indent=2)
        print(f"wrote {args.out}/train_xrv_probs{tag}.npy")
        return


if __name__ == "__main__":
    main()
