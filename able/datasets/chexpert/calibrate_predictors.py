"""CheXpert protocol v0, rest of section 5: T6 calibration and the FC correction.

Two things the raw AUCs do not give you.

T6 -- the predictor's own ceiling and floor. A counterfactual image is scored
by running this predictor over it, so the score a PERFECT generator could
reach is not 1.0, it is whatever the predictor achieves on REAL images of the
same kind. Measuring TPR and FPR on held-out real radiographs puts that
ceiling and floor on record, so CC and FC are read against what the instrument
can do rather than against an oracle that does not exist. Without it a model
is charged for the predictor's errors.

r_k(a) -- the correlation correction for FC. FC asks whether attributes other
than the intervened one stayed put. But in real chest radiographs the
attributes are not independent: images with effusion read higher on edema
whether or not anything caused it. If do(effusion) shifts the edema reading,
part of that shift is the association the predictor learned from real data,
not damage the generator did. r_k(a) measures that association on real
held-out images -- the difference in the predictor's mean reading of k between
the groups of a -- so FC can discount it.

Thresholds are tuned by Youden's J on the SELECTION set (inside the train
split), never on the held-out split, because the evaluation cohort is drawn
from there. Tuning on the cohort pool is not forbidden by the standing rule --
only attribute SELECTION is -- but there is no reason to spend the
independence when a clean set is sitting right there.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_predictors import BINARY, Chex, Net, predict

ATTRS = ["sex", "age_bin", "effusion", "cardiomegaly"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/exouser/able_data/chexpert")
    ap.add_argument("--out", default="/home/exouser/SpecRoute/experiments/hdae/outputs/attr_predictors_chexpert")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--select-frac", type=float, default=0.10)
    args = ap.parse_args()
    dev = "cuda"

    meta = pd.read_csv(f"{args.data}/chexpert_meta.csv")
    split = np.load(f"{args.data}/split.npy")
    train_rows = np.flatnonzero(split == 0)
    held_rows = np.flatnonzero(split == 1)

    # the same fit/select partition train_predictors.py made, reproduced from the seed
    tp = np.sort(meta.loc[train_rows, "patient_id"].unique())
    perm = np.random.default_rng(args.seed + 1).permutation(tp)
    sel_pat = set(perm[:int(round(len(tp) * args.select_frac))].tolist())
    is_sel = meta["patient_id"].isin(sel_pat).to_numpy()
    sel_rows = train_rows[is_sel[train_rows]]

    ck = torch.load(f"{args.out}/best.ckpt", map_location="cpu")
    if ck["binary"] != BINARY:
        raise SystemExit(f"checkpoint label order {ck['binary']} != {BINARY}")
    net = Net().to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    npy = f"{args.data}/images_224.npy"
    mk = lambda rows: DataLoader(Chex(npy, meta, rows, False), batch_size=256,
                                 num_workers=args.workers, pin_memory=True)
    p_sel, a_sel, y_sel, ab_sel = predict(net, mk(sel_rows), dev)
    p_hel, a_hel, y_hel, ab_hel = predict(net, mk(held_rows), dev)

    rmeta = pd.read_csv(f"{args.data}/radiologist_val_meta.csv")
    p_rad, _, y_rad, _ = predict(net, DataLoader(
        Chex(f"{args.data}/radiologist_val_224.npy", rmeta, np.arange(len(rmeta)), False),
        batch_size=128, num_workers=2), dev)

    rep = json.load(open(f"{args.out}/predictor_report.json"))
    kept = rep["kept_labels"]

    # Bootstrap CIs on both AUCs. radiologist_val is 200 images, so a point
    # estimate there carries a wide interval and should not be read as if it
    # were the 17,740-image held-out number. The two sets also disagree on
    # WHAT is being measured: held-out labels come from the report-text
    # labeller, radiologist_val labels come from radiologists reading the
    # image. A label that scores far higher on radiologist_val is a label
    # whose held-out AUC is partly bounded by noise in its own targets.
    def boot_auc(y, p, n=2000, seed=0):
        if len(np.unique(y)) < 2:
            return None, None, None
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(y), size=(n, len(y)))
        vals = []
        for row in idx:
            yy = y[row]
            if len(np.unique(yy)) < 2:
                continue
            vals.append(roc_auc_score(yy, p[row]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return round(float(roc_auc_score(y, p)), 4), round(float(lo), 4), round(float(hi), 4)

    ci = {}
    for j, k in enumerate(BINARY):
        a_h, lo_h, hi_h = boot_auc(y_hel[:, j], p_hel[:, j], n=400, seed=1)
        a_r, lo_r, hi_r = boot_auc(y_rad[:, j], p_rad[:, j], n=2000, seed=2)
        ci[k] = {"heldout_auc": a_h, "heldout_ci95": [lo_h, hi_h],
                 "radiologist_val_auc": a_r, "radiologist_val_ci95": [lo_r, hi_r],
                 "kept_by_gate": k in kept}

    # ---- thresholds by Youden's J on the selection set
    thr = {}
    for j, k in enumerate(BINARY):
        fpr, tpr, t = roc_curve(y_sel[:, j], p_sel[:, j])
        thr[k] = float(t[int(np.argmax(tpr - fpr))])

    # ---- T6: ceiling and floor on held-out REAL images
    t6 = {}
    for j, k in enumerate(BINARY):
        yh, ph = y_hel[:, j], p_hel[:, j] >= thr[k]
        pos, neg = yh == 1, yh == 0
        t6[k] = {
            "threshold": round(thr[k], 4),
            "tpr_ceiling": round(float(ph[pos].mean()), 4) if pos.any() else None,
            "fpr_floor": round(float(ph[neg].mean()), 4) if neg.any() else None,
            "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
            "kept": k in kept,
        }
    conf = a_hel[np.arange(len(ab_hel)), ab_hel]
    t6["age_bin"] = {
        "accuracy_ceiling": round(float((a_hel.argmax(1) == ab_hel).mean()), 4),
        "mean_prob_on_true_bin": round(float(conf.mean()), 4),
        "per_bin_recall": {str(b): round(float((a_hel.argmax(1)[ab_hel == b] == b).mean()), 4)
                           for b in (0, 1, 2)},
    }

    # ---- r_k(a): the predictor's reading of k across the groups of a, on real images
    hm = meta.iloc[held_rows].reset_index(drop=True)
    r = {}
    for a in ATTRS:
        va = hm[a].to_numpy()
        groups = sorted(np.unique(va).tolist())
        r[a] = {}
        for j, k in enumerate(BINARY):
            if k == a:
                continue
            means = {str(g): round(float(p_hel[va == g, j].mean()), 4) for g in groups}
            spread = max(means.values()) - min(means.values())
            r[a][k] = {"mean_prob_by_group": means, "max_shift": round(float(spread), 4)}
        am = {str(g): round(float(a_hel[va == g].argmax(1).mean()), 4) for g in groups}
        if a != "age_bin":
            r[a]["age_bin_mean_predicted"] = {"mean_bin_by_group": am}

    rep["auc_bootstrap_ci"] = ci
    rep["t6_calibration"] = t6
    rep["thresholds_youden_on_select"] = {k: round(v, 4) for k, v in thr.items()}
    rep["fc_correlation_correction_r_k_a"] = r
    rep["calibration_note"] = (
        "tpr_ceiling/fpr_floor are measured on held-out REAL images, so they bound what a "
        "perfect generator could score; r_k(a) is the association already present in real "
        "data and is what FC discounts before charging a model for a side effect.")
    with open(f"{args.out}/predictor_report.json", "w") as f:
        json.dump(rep, f, indent=2)

    print("%-17s %6s %9s %9s" % ("label", "thr", "TPR ceil", "FPR floor"))
    for k in BINARY:
        print("%-17s %6.3f %9s %9s %s" % (k, t6[k]["threshold"], t6[k]["tpr_ceiling"],
                                          t6[k]["fpr_floor"], "" if k in kept else "(dropped)"))
    print(f"\nage_bin accuracy ceiling {t6['age_bin']['accuracy_ceiling']}")
    print("\nlargest real-data associations (what FC must discount):")
    flat = [(a, k, v["max_shift"]) for a, d in r.items() for k, v in d.items() if isinstance(v, dict) and "max_shift" in v]
    for a, k, s in sorted(flat, key=lambda t: -t[2])[:8]:
        print(f"  do({a}) -> reading of {k}: shift {s:.3f}")
    print("\nAUC with 95% bootstrap CI  (held-out labels are report-text; radiologist_val is image truth)")
    print("%-17s %-22s %-22s" % ("label", "held-out (n=%d)" % len(y_hel), "radiologist_val (n=%d)" % len(y_rad)))
    for k in BINARY:
        c = ci[k]
        print("%-17s %.4f [%.4f,%.4f]  %.4f [%.4f,%.4f]  %s" % (
            k, c["heldout_auc"], *c["heldout_ci95"], c["radiologist_val_auc"], *c["radiologist_val_ci95"],
            "KEPT" if c["kept_by_gate"] else "dropped"))
    print(f"\nmerged into {args.out}/predictor_report.json")


if __name__ == "__main__":
    main()
