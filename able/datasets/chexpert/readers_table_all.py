"""Three readers x every CheXpert attribute x train / test / radiologist_val.

Readers
  ours10    the frozen v2 predictor (best.ckpt). Trained on the FIT rows of
            the train split, epoch chosen on SELECT, never on heldout.
  ours11    the measurement checkpoint with an 11th head for enlarged
            cardiomediastinum. Same recipe, same split, same seed.
  xrv       TorchXRayVision densenet121-res224-all. Its training mix INCLUDES
            CheXpert, so on train and test it is partly scoring labels it
            already learned. Not independent. No Support Devices output.
  chexzero  CheXzero zero-shot, 10-checkpoint ensemble. Trained on MIMIC-CXR
            report TEXT only -- never on CheXpert, never on a label column.
            The only reader here that is independent of the thing being scored.

What the numbers mean, and it differs by column:

  train / test  labels are REPORT-derived and noisy, with -1 folded to 0 the
                way protocol section 3 step 4 does for eval-only findings. An
                AUC here is agreement-with-the-report, not accuracy. For
                ours10/ours11 the train column also contains the rows they
                took gradients on, so it is partly fit quality; fit and select
                are broken out separately in the JSON for that reason.
  radval        labels are radiologist-assigned from the images. This is the
                only column where AUC means what people usually mean by it,
                and it is the column the v2 gate is defined on.

A reader that is high on train/test but low on radval has learned the report,
not the finding. That gap is the quantity of interest.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

OURS10 = ["sex", "effusion", "cardiomegaly", "support_devices", "ap_pa",
          "edema", "lung_opacity", "consolidation", "atelectasis", "pneumothorax"]
OURS11 = OURS10 + ["enlarged_cardiomediastinum"]

# CheXpert column -> our key
CHEX = {
    "Pleural Effusion": "effusion", "Cardiomegaly": "cardiomegaly",
    "Support Devices": "support_devices", "Edema": "edema",
    "Lung Opacity": "lung_opacity", "Consolidation": "consolidation",
    "Atelectasis": "atelectasis", "Pneumothorax": "pneumothorax",
    "Enlarged Cardiomediastinum": "enlarged_cardiomediastinum",
    "Lung Lesion": "lung_lesion", "Fracture": "fracture",
    "Pneumonia": "pneumonia", "Pleural Other": "pleural_other",
}
FINDINGS = list(CHEX.values())

XRV_PATH = ["Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
            "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
            "Cardiomegaly", "Nodule", "Mass", "Hernia", "Lung Lesion", "Fracture",
            "Lung Opacity", "Enlarged Cardiomediastinum"]
XRV = {"atelectasis": "Atelectasis", "consolidation": "Consolidation",
       "pneumothorax": "Pneumothorax", "edema": "Edema", "effusion": "Effusion",
       "pneumonia": "Pneumonia", "cardiomegaly": "Cardiomegaly",
       "lung_lesion": "Lung Lesion", "fracture": "Fracture",
       "lung_opacity": "Lung Opacity",
       "enlarged_cardiomediastinum": "Enlarged Cardiomediastinum"}

CZ_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion",
             "Enlarged Cardiomediastinum", "Lung Opacity", "Lung Lesion", "Pneumonia",
             "Pneumothorax", "Pleural Other", "Fracture", "Support Devices"]
CZ = {CHEX[L]: i for i, L in enumerate(CZ_LABELS)}


def auc(y, s):
    """Mann-Whitney AUC with tie-averaged ranks. Same value as sklearn, faster."""
    y = np.asarray(y, np.int8)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return None
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def boot(y, s, n, seed=7):
    y, s = np.asarray(y), np.asarray(s)
    a = auc(y, s)
    if a is None or n == 0:
        return a, None, None
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        b = auc(y[i], s[i])
        if b is not None:
            v.append(b)
    if not v:
        return a, None, None
    return (round(a, 4), round(float(np.percentile(v, 2.5)), 4),
            round(float(np.percentile(v, 97.5)), 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/exouser/able_data/chexpert")
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261107)
    a = ap.parse_args()
    D, Pd = a.data, a.probs
    os.makedirs(a.out, exist_ok=True)

    meta = pd.read_csv(f"{D}/chexpert_meta.csv")
    split = np.load(f"{D}/split.npy")
    rmeta = pd.read_csv(f"{D}/radiologist_val_meta.csv")

    # --- reproduce the predictors' own fit/select sub-split, same rule, same seed
    train_rows = np.flatnonzero(split == 0)
    held_rows = np.flatnonzero(split == 1)
    tp = np.sort(meta.loc[train_rows, "patient_id"].unique())
    perm = np.random.default_rng(a.seed + 1).permutation(tp)
    sel_pat = set(perm[:int(round(len(tp) * 0.10))].tolist())
    is_sel = meta["patient_id"].isin(sel_pat).to_numpy()
    fit_rows = train_rows[~is_sel[train_rows]]
    sel_rows = train_rows[is_sel[train_rows]]
    print(f"fit {len(fit_rows)}  select {len(sel_rows)}  test {len(held_rows)}")

    # --- labels. 13 findings off the source CSV, -1 folded to 0 (section 3 step 4)
    src = pd.read_csv(f"{D}/source_csv/train.csv").set_index("Path")
    src = src.loc[meta["path"].values]
    Y, UNC = {}, {}
    for L, k in CHEX.items():
        v = pd.to_numeric(src[L], errors="coerce").to_numpy()
        UNC[k] = (v == -1)
        Y[k] = np.where(np.isnan(v), 0, np.where(v == -1, 0, v)).astype(np.int8)
    for k in ("sex", "ap_pa"):
        Y[k] = meta[k].to_numpy(np.int8); UNC[k] = np.zeros(len(meta), bool)
    AGE = meta["age_bin"].to_numpy(np.int64)

    valid = pd.read_csv(f"{D}/source_csv/valid.csv").set_index("Path")
    vs = valid.loc[rmeta["path"].values]
    RY = {}
    for L, k in CHEX.items():
        v = pd.to_numeric(vs[L], errors="coerce").to_numpy()
        RY[k] = np.where(np.isnan(v), 0, v).astype(np.int8)
    for k in ("sex", "ap_pa"):
        RY[k] = rmeta[k].to_numpy(np.int8)
    RAGE = rmeta["age_bin"].to_numpy(np.int64)

    # --- reader scores on the frozen rows
    S, SA = {}, {}
    S["ours10"] = np.load(f"{Pd}/ours10_frozen_bin.npy"); SA["ours10"] = np.load(f"{Pd}/ours10_frozen_age.npy")
    S["ours11"] = np.load(f"{Pd}/ours11_frozen_bin.npy"); SA["ours11"] = np.load(f"{Pd}/ours11_frozen_age.npy")
    S["chexzero"] = np.load(f"{Pd}/chexzero_images_224_probs.npy")
    xr = pd.read_csv(f"{D}/noise_audit/train_xrv_rows.csv")
    xp = np.load(f"{D}/noise_audit/train_xrv_probs.npy")
    pos = pd.Series(np.arange(len(xr)), index=xr["Path"].values)
    S["xrv"] = xp[pos.loc[meta["path"].values].to_numpy()]

    R, RA = {}, {}
    R["ours10"] = np.load(f"{Pd}/ours10_radval_bin.npy"); RA["ours10"] = np.load(f"{Pd}/ours10_radval_age.npy")
    R["ours11"] = np.load(f"{Pd}/ours11_radval_bin.npy"); RA["ours11"] = np.load(f"{Pd}/ours11_radval_age.npy")
    R["chexzero"] = np.load(f"{Pd}/chexzero_radiologist_val_224_probs.npy")
    vxp = np.load(f"{D}/noise_audit/valid_xrv_probs.npy")
    vpos = pd.Series(np.arange(len(valid)), index=valid.index.values)
    R["xrv"] = vxp[vpos.loc[rmeta["path"].values].to_numpy()]

    def col(reader, k, radval=False):
        """score column for attribute k, or None if this reader cannot read it"""
        M = (R if radval else S)
        if reader == "ours10":
            return M["ours10"][:, OURS10.index(k)] if k in OURS10 else None
        if reader == "ours11":
            return M["ours11"][:, OURS11.index(k)] if k in OURS11 else None
        if reader == "xrv":
            return M["xrv"][:, XRV_PATH.index(XRV[k])] if k in XRV else None
        if reader == "chexzero":
            return M["chexzero"][:, CZ[k]] if k in CZ else None
        return None

    SETS = [("fit", fit_rows, 100), ("select", sel_rows, 300),
            ("train", train_rows, 100), ("test", held_rows, 1000)]
    READERS = ["ours10", "ours11", "xrv", "chexzero"]
    ATTRS = FINDINGS + ["sex", "ap_pa"]

    out = {}
    for k in ATTRS:
        out[k] = {}
        for reader in READERS:
            e = {}
            c = col(reader, k)
            for name, rows, nb in SETS:
                if c is None:
                    e[name] = None; continue
                yy, ss = Y[k][rows], c[rows]
                av, lo, hi = boot(yy, ss, nb)
                e[name] = {"auc": av, "ci95": [lo, hi], "n": int(len(rows)),
                           "n_pos": int(yy.sum()),
                           "uncertain_folded_to_0": int(UNC[k][rows].sum())}
                # certain-rows-only variant: drops the -1 rows instead of folding
                m = ~UNC[k][rows]
                if m.sum() < len(rows):
                    ac = auc(Y[k][rows][m], c[rows][m])
                    e[name]["auc_certain_only"] = None if ac is None else round(ac, 4)
                    e[name]["n_certain"] = int(m.sum())
            rc = col(reader, k, radval=True)
            if rc is None:
                e["radval"] = None
            else:
                av, lo, hi = boot(RY[k], rc, 2000)
                e["radval"] = {"auc": av, "ci95": [lo, hi], "n": int(len(RY[k])),
                               "n_pos": int(RY[k].sum())}
            out[k][reader] = e
        print(f"  {k} done", flush=True)

    # age_bin, ours only (XRV and CheXzero have no age output)
    out["age_bin"] = {}
    for reader in ("ours10", "ours11"):
        e = {}
        for name, rows, nb in SETS:
            from sklearn.metrics import roc_auc_score
            e[name] = {"auc_ovr_macro": round(float(roc_auc_score(
                AGE[rows], SA[reader][rows], multi_class="ovr", average="macro")), 4),
                "n": int(len(rows))}
        e["radval"] = {"auc_ovr_macro": round(float(roc_auc_score(
            RAGE, RA[reader], multi_class="ovr", average="macro")), 4), "n": int(len(RAGE))}
        out["age_bin"][reader] = e

    rec = {"what": "three readers x all CheXpert attributes x train/test/radiologist_val",
           "date": "2026-11-08",
           "label_semantics": {
               "train/select/fit/test": "report-derived CheXpert labels, -1 folded to 0 "
                                        "(protocol section 3 step 4); auc_certain_only drops "
                                        "the -1 rows instead",
               "radval": "radiologist-assigned from the images; the gate column"},
           "readers": {
               "ours10": "frozen v2 predictor best.ckpt; fit rows were trained on",
               "ours11": "measurement checkpoint, 11th head for enlarged_cardiomediastinum",
               "xrv": "TorchXRayVision densenet121-res224-all; CheXpert IS in its training "
                      "mix, so train/test are not independent; no Support Devices output",
               "chexzero": "CheXzero zero-shot 10-checkpoint ensemble; MIMIC-CXR report text "
                           "only, independent of every label column here"},
           "bootstrap": "percentile, seed 7; resamples: fit/train 100, select 300, "
                        "test 1000, radval 2000",
           "sets": {n: int(len(r)) for n, r, _ in SETS},
           "table": out}
    json.dump(rec, open(f"{a.out}/readers_all.json", "w"), indent=2)
    print(f"wrote {a.out}/readers_all.json")
