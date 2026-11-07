"""CheXpert protocol v1, amendment to section 5: re-gate on radiologist labels.

v0 gated every eval-only attribute on held-out AUC. Those held-out targets
come from the report-text labeller, so a label whose REPORTS are inconsistent
scores low however well the predictor reads the image. The label-noise audit
made that concrete, and three labels v0 dropped (edema, lung_opacity,
consolidation) score .90+ against radiologists with CI lower bounds above
.86. The v0 gate was measuring label noise, not predictor capability.

v1 gates the eval-only attributes on radiologist_val instead: AUC >= 0.85 AND
the 95% bootstrap CI lower bound >= 0.80. The CI condition is what stops the
200-image reference from admitting a label on a lucky point estimate -- it is
exactly what disqualifies pneumothorax, whose interval runs [.58,.97].

The MODELED attributes keep the v0 gate, because the amendment is scoped to
eval-only. Their radiologist_val numbers are reported anyway, and one of them
would not pass the new gate; that is stated, not buried.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_predictors import BINARY, Chex, Net, predict

MODELED = ["sex", "effusion", "cardiomegaly"]          # + age_bin, handled separately
EVAL_ONLY = ["edema", "lung_opacity", "consolidation", "atelectasis",
             "support_devices", "ap_pa", "pneumothorax"]
AUC_MIN, CI_LO_MIN = 0.85, 0.80


def boot(y, p, n=2000, seed=0):
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
    tp = np.sort(meta.loc[train_rows, "patient_id"].unique())
    perm = np.random.default_rng(args.seed + 1).permutation(tp)
    sel_pat = set(perm[:int(round(len(tp) * args.select_frac))].tolist())
    sel_rows = train_rows[meta["patient_id"].isin(sel_pat).to_numpy()[train_rows]]

    ck = torch.load(f"{args.out}/best.ckpt", map_location="cpu")
    assert ck["binary"] == BINARY
    net = Net().to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    npy = f"{args.data}/images_224.npy"
    mk = lambda rows: DataLoader(Chex(npy, meta, rows, False), batch_size=256,
                                 num_workers=args.workers, pin_memory=True)
    p_sel, a_sel, y_sel, ab_sel = predict(net, mk(sel_rows), dev)
    p_hel, a_hel, y_hel, ab_hel = predict(net, mk(held_rows), dev)
    rmeta = pd.read_csv(f"{args.data}/radiologist_val_meta.csv")
    p_rad, a_rad, y_rad, ab_rad = predict(net, DataLoader(
        Chex(f"{args.data}/radiologist_val_224.npy", rmeta, np.arange(len(rmeta)), False),
        batch_size=128, num_workers=2), dev)

    rep = json.load(open(f"{args.out}/predictor_report.json"))
    gate = {}
    for j, k in enumerate(BINARY):
        a_r, lo_r, hi_r = boot(y_rad[:, j], p_rad[:, j], seed=7)
        a_h = rep["heldout"][k]["auc"]
        scope = "eval_only" if k in EVAL_ONLY else "modeled"
        passes = (a_r is not None and a_r >= AUC_MIN and lo_r is not None and lo_r >= CI_LO_MIN)
        gate[k] = {
            "scope": scope,
            "radiologist_val_auc": a_r, "radiologist_val_ci95": [lo_r, hi_r],
            "n_pos_radiologist_val": int(y_rad[:, j].sum()),
            "heldout_report_auc": a_h,
            "v1_gate_pass": passes,
            "v0_gate_pass": a_h is not None and a_h >= 0.85,
            "kept_v1": passes if scope == "eval_only" else (a_h is not None and a_h >= 0.85),
            "gate_applied": "radiologist_val AUC>=0.85 and CI_lo>=0.80" if scope == "eval_only"
                            else "v0 held-out AUC>=0.85 (modeled: amendment is scoped to eval-only)",
        }
    # age_bin against radiologist labels
    try:
        a_age = round(float(roc_auc_score(ab_rad, a_rad, multi_class="ovr", average="macro")), 4)
    except ValueError:
        a_age = None
    gate["age_bin"] = {
        "scope": "modeled",
        "radiologist_val_auc_ovr_macro": a_age,
        "radiologist_val_balanced_acc": round(float(balanced_accuracy_score(ab_rad, a_rad.argmax(1))), 4),
        "heldout_report_auc": rep["heldout"]["age_bin"]["auc_ovr_macro"],
        "kept_v1": True,
        "gate_applied": "v0 held-out AUC>=0.85 (modeled)",
    }

    kept = [k for k in BINARY if gate[k]["kept_v1"]]
    readmitted = [k for k in EVAL_ONLY if gate[k]["kept_v1"] and k not in rep["kept_labels"]]
    revoked = [k for k in EVAL_ONLY if not gate[k]["kept_v1"] and k in rep["kept_labels"]]

    # ---- T6 on radiologist_val for every kept label (thresholds still tuned on SELECT)
    t6 = {}
    for j, k in enumerate(BINARY):
        if k not in kept:
            continue
        fpr, tpr, t = roc_curve(y_sel[:, j], p_sel[:, j])
        th = float(t[int(np.argmax(tpr - fpr))])
        out = {"threshold": round(th, 4), "threshold_tuned_on": "select set (inside train split)"}
        for name, y, p in (("radiologist_val", y_rad[:, j], p_rad[:, j]),
                           ("heldout_report_labels", y_hel[:, j], p_hel[:, j])):
            pos, neg = y == 1, y == 0
            out[name] = {
                "tpr_ceiling": round(float((p[pos] >= th).mean()), 4) if pos.any() else None,
                "fpr_floor": round(float((p[neg] >= th).mean()), 4) if neg.any() else None,
                "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
        out["readmitted_in_v1"] = k in readmitted
        t6[k] = out
    t6["age_bin"] = {
        "radiologist_val_accuracy_ceiling": round(float((a_rad.argmax(1) == ab_rad).mean()), 4),
        "heldout_accuracy_ceiling": rep["t6_calibration"]["age_bin"]["accuracy_ceiling"]}

    rep["version"] = "v1"
    rep["amendment"] = {
        "date": "2026-11-07",
        "change": "eval-only attributes are gated on radiologist_val AUC >= 0.85 with 95% "
                  "bootstrap CI lower bound >= 0.80, replacing the v0 gate on held-out "
                  "report-label AUC.",
        "reason": "Held-out targets come from the report-text labeller. A label whose reports "
                  "are inconsistent scores low no matter how well the predictor reads the "
                  "image, so the v0 gate was measuring label noise rather than predictor "
                  "capability. The label-noise audit showed edema, lung_opacity and "
                  "consolidation score .90+ against radiologists while failing on held-out.",
        "why_the_ci_condition": "radiologist_val is 200 images. A bare point estimate would "
                                "admit a label on luck; requiring the CI lower bound above "
                                "0.80 is what disqualifies pneumothorax, whose interval runs "
                                "[.58,.97].",
        "scope": "eval-only attributes only. Modeled attributes keep the v0 gate.",
        "readmitted": readmitted, "revoked": revoked,
        "modeled_that_would_fail_the_new_gate": [
            k for k in MODELED if not gate[k]["v1_gate_pass"]],
    }
    rep["v1_gate"] = gate
    rep["kept_labels_v0"] = rep["kept_labels"]
    rep["kept_labels"] = kept
    rep["dropped_by_gate"] = [k for k in BINARY if k not in kept]
    rep["t6_calibration_v1"] = t6
    with open(f"{args.out}/predictor_report.json", "w") as f:
        json.dump(rep, f, indent=2)

    print("%-17s %-9s %-24s %-10s %s" % ("label", "scope", "radiologist_val AUC [CI]", "held-out", "v1"))
    for k in BINARY:
        g = gate[k]
        print("%-17s %-9s %-24s %-10s %s" % (
            k, g["scope"],
            f"{g['radiologist_val_auc']} [{g['radiologist_val_ci95'][0]},{g['radiologist_val_ci95'][1]}]",
            g["heldout_report_auc"], "KEPT" if g["kept_v1"] else "dropped"))
    print(f"\nage_bin  radiologist_val OvR {a_age}   held-out {gate['age_bin']['heldout_report_auc']}")
    print(f"\nre-admitted: {readmitted or 'none'}")
    print(f"revoked:     {revoked or 'none'}")
    print(f"kept v1:     {kept + ['age_bin']}")
    mf = rep["amendment"]["modeled_that_would_fail_the_new_gate"]
    if mf:
        print(f"\nNOTE: modeled attribute(s) {mf} would NOT pass the new gate; they keep the v0 "
              f"gate because the amendment is scoped to eval-only.")
    print(f"\nupdated {args.out}/predictor_report.json")


if __name__ == "__main__":
    main()
