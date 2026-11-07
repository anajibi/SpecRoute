"""CheXpert label-noise audit, S5 and S6: the report.

Two measures of the same thing, deliberately kept apart.

`gap` = AUC against radiologist labels minus AUC against report labels. It is
threshold-free, so nothing about calibration can distort it. A large positive
gap means the reader sees the finding in the image but the report labeller
did not write it down.

`confident_disagree` is the share of rows where the reader is confident and
the report label says the opposite. It needs thresholds, and the fixed
0.9/0.1 pair turned out to flag 2.6-9.4% of the RADIOLOGIST set -- i.e. at
that setting the metric is mostly measuring the reader's own errors. So the
thresholds are calibrated per label on the radiologist set: `hi` at the 99th
percentile of probability among radiologist negatives, `lo` at the 1st
percentile among radiologist positives. Both numbers are reported; the
0.9/0.1 one is labelled a model-error floor, not a label-error rate.

Pneumothorax is kept in every table and flagged. With 8 positives in the
234-image reference its AUC CI runs [.50,.79] and its head never emits a
probability above .791 on any of the 234, so neither measure is estimable
for it. Flagged, not dropped.

The 668-image test set cannot serve as a second reference: its labels are
public but its images are behind Stanford AIMI registration and absent from
the Kaggle mirror, so S3 and S6.1 are reported as not runnable.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABEL_MAP = {
    "Pleural Effusion": "Effusion", "Cardiomegaly": "Cardiomegaly", "Edema": "Edema",
    "Consolidation": "Consolidation", "Lung Opacity": "Lung Opacity",
    "Atelectasis": "Atelectasis", "Pneumothorax": "Pneumothorax",
    "Lung Lesion": "Lung Lesion", "Fracture": "Fracture",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
}
NOT_AUDITABLE = {"Support Devices": "no XRV output; your own classifier only"}
MIN_REF_POS = 20   # below this the radiologist set cannot pin a threshold or an AUC


def boot(y, p, n=1000, seed=0):
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
    ap.add_argument("--out", default="/home/exouser/SpecRoute/outputs/chexpert_noise")
    ap.add_argument("--csv-dir", default="/home/exouser/able_data/chexpert/source_csv")
    args = ap.parse_args()
    O = args.out

    valid = pd.read_csv(f"{args.csv_dir}/valid.csv")
    train = pd.read_csv(f"{args.csv_dir}/train.csv")
    fr = train[train["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
    paths = json.load(open(f"{O}/s0_s1.json"))["S1"]["pathologies"]
    Pv = np.load(f"{O}/valid_xrv_probs.npy")
    Pt = np.load(f"{O}/train_xrv_probs.npy")
    assert len(Pt) == len(fr), f"{len(Pt)} probs vs {len(fr)} frontal rows"
    has_chex_train = os.path.exists(f"{O}/train_xrv_probs_chex.npy")
    Ptc = np.load(f"{O}/train_xrv_probs_chex.npy") if has_chex_train else None
    Pvc = np.load(f"{O}/valid_xrv_probs_chex.npy")

    rows = {}
    for csvcol, x in LABEL_MAP.items():
        j = paths.index(x)
        yv = pd.to_numeric(valid[csvcol], errors="coerce").to_numpy()
        kv = np.isin(yv, [0, 1])
        auc_rad = boot(yv[kv], Pv[kv, j], seed=1)
        n_ref_pos = int((yv[kv] == 1).sum())

        yt = pd.to_numeric(fr[csvcol], errors="coerce").to_numpy()
        kt = np.isin(yt, [0, 1])
        auc_rep = boot(yt[kt], Pt[kt, j], n=200, seed=2)

        # the "blank means negative" convention, costed
        yb = np.where(np.isnan(yt), 0.0, yt)
        kb = np.isin(yb, [0, 1])
        auc_blank0 = boot(yb[kb], Pt[kb, j], n=200, seed=3)

        # thresholds: calibrated on the radiologist set, each side at ~1%
        neg, pos = Pv[kv, j][yv[kv] == 0], Pv[kv, j][yv[kv] == 1]
        hi = float(np.percentile(neg, 99)) if len(neg) else 0.9
        lo = float(np.percentile(pos, 1)) if len(pos) else 0.1
        estimable = n_ref_pos >= MIN_REF_POS

        def rates(y, p, h, l):
            fp = ((y == 1) & (p < l)).sum()     # report says present, reader sure it is not
            fn = ((y == 0) & (p > h)).sum()     # report says absent, reader sure it is there
            return (round(float((fp + fn) / len(y)), 4),
                    round(float(fn / max((y == 0).sum(), 1)), 4),
                    round(float(fp / max((y == 1).sum(), 1)), 4))

        d_cal, fn_cal, fp_cal = rates(yt[kt], Pt[kt, j], hi, lo)
        d_fix, fn_fix, fp_fix = rates(yt[kt], Pt[kt, j], 0.9, 0.1)
        base_cal = rates(yv[kv], Pv[kv, j], hi, lo)[0]
        base_fix = rates(yv[kv], Pv[kv, j], 0.9, 0.1)[0]

        raw = pd.to_numeric(train[csvcol], errors="coerce").to_numpy()
        gap = None if (auc_rad[0] is None or auc_rep[0] is None) else round(auc_rad[0] - auc_rep[0], 4)

        rows[x] = {
            "csv_column": csvcol,
            "auc_vs_report_train": auc_rep[0], "auc_vs_report_ci": auc_rep[1:],
            "auc_vs_report_blank_as_negative": auc_blank0[0],
            "auc_vs_radiologist_valid": auc_rad[0], "auc_vs_radiologist_ci": auc_rad[1:],
            "auc_vs_radiologist_test": None,
            "gap": gap,
            "thresholds": {"hi": round(hi, 4), "lo": round(lo, 4)},
            "confident_disagree_calibrated": d_cal,
            "radiologist_false_flag_rate_calibrated": base_cal,
            "excess_over_radiologist_baseline": None if not estimable else round(d_cal - base_cal, 4),
            "fn_like_calibrated": fn_cal, "fp_like_calibrated": fp_cal,
            "model_error_floor_fixed_0.9_0.1": {
                "confident_disagree_train": d_fix, "radiologist_false_flag_rate": base_fix,
                "fn_like": fn_fix, "fp_like": fp_fix,
                "note": "the radiologist rate here is the reader's own error, not label error"},
            "uncertain_frac": round(float((raw == -1).mean()), 4),
            "blank_frac": round(float(np.isnan(raw).mean()), 4),
            "positive_frac": round(float((raw == 1).mean()), 4),
            "n_reference_positives": n_ref_pos,
            "estimable": estimable,
            "flag": None if estimable else
                    f"not auditable with XRV (n={n_ref_pos} positives in valid)",
        }

    # ---- verdicts
    for x, r in rows.items():
        if not r["estimable"]:
            r["verdict"] = "not auditable"
            continue
        g, d = r["gap"], r["confident_disagree_calibrated"]
        r["verdict"] = ("unreliable" if (g is not None and g > 0.15) or d > 0.15
                        else "noisy" if g is not None and g >= 0.05
                        else "clean" if g is not None and d < 0.05
                        else "noisy")

    # ---- age sanity
    age = {}
    if os.path.exists(f"{O}/train_xrv_age.npy"):
        A = np.load(f"{O}/train_xrv_age.npy")
        ya = pd.to_numeric(fr["Age"], errors="coerce").to_numpy()
        k = ~np.isnan(ya) & (ya >= 18) & (ya <= 100) & np.isfinite(A)
        b = lambda v: np.where(v < 45, 0, np.where(v > 65, 2, 1))
        cm = pd.crosstab(b(ya[k]), b(A[k])).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
        age = {"mae_years": round(float(np.abs(A[k] - ya[k]).mean()), 2),
               "bias_years": round(float((A[k] - ya[k]).mean()), 2), "n": int(k.sum()),
               "bin_confusion_rows_true_cols_pred": cm.values.tolist(),
               "bin_accuracy": round(float((b(ya[k]) == b(A[k])).mean()), 4),
               "note": "sanity only; age labels do not come from reports"}

    # ---- S6
    s6 = {}
    s6["1_valid_vs_test_agreement"] = {
        "pass": None,
        "note": "not runnable: the 668-image test set's labels are public but its images are "
                "behind Stanford AIMI registration and are not in the Kaggle mirror, which "
                "carries train and valid only. The audit therefore has ONE reference set."}
    order_gap = sorted(((r["gap"], x) for x, r in rows.items()
                        if r["estimable"] and r["gap"] is not None), reverse=True)
    order_dis = sorted(((r["confident_disagree_calibrated"], x) for x, r in rows.items()
                        if r["estimable"]), reverse=True)
    rank_g = {x: i for i, (_, x) in enumerate(order_gap)}
    rank_d = {x: i for i, (_, x) in enumerate(order_dis)}
    disagreeing = [x for x in rank_g if abs(rank_g[x] - rank_d[x]) >= 3]
    s6["2_ordering"] = {
        "by_gap": [x for _, x in order_gap], "by_confident_disagree": [x for _, x in order_dis],
        "rank_shift_ge_3": disagreeing,
        "uncertain_ge_positive": [x for x, r in rows.items()
                                  if r["uncertain_frac"] >= r["positive_frac"]],
        "pass": not disagreeing,
        "note": "the two orderings should broadly agree; a label high on one and low on the "
                "other is flagged rather than averaged away"}
    s6["3_radiologist_false_flag"] = {
        "calibrated": {x: r["radiologist_false_flag_rate_calibrated"] for x, r in rows.items()},
        "fixed_0.9_0.1": {x: r["model_error_floor_fixed_0.9_0.1"]["radiologist_false_flag_rate"]
                          for x, r in rows.items()},
        "pass": all(r["radiologist_false_flag_rate_calibrated"] <= 0.02 for r in rows.values()),
        "note": "fixed thresholds fail this for six of eight labels, which is why they are "
                "reported only as a model-error floor"}
    if has_chex_train:
        c = {}
        for csvcol, x in LABEL_MAP.items():
            j = paths.index(x)
            yt = pd.to_numeric(fr[csvcol], errors="coerce").to_numpy()
            kt = np.isin(yt, [0, 1])
            yv = pd.to_numeric(valid[csvcol], errors="coerce").to_numpy()
            kv = np.isin(yv, [0, 1])
            a_all_t, a_chex_t = rows[x]["auc_vs_report_train"], boot(yt[kt], Ptc[kt, j], n=200, seed=4)[0]
            a_all_v, a_chex_v = rows[x]["auc_vs_radiologist_valid"], boot(yv[kv], Pvc[kv, j], seed=5)[0]
            c[x] = {"report_all": a_all_t, "report_chex": a_chex_t,
                    "report_chex_higher": None if None in (a_all_t, a_chex_t) else a_chex_t > a_all_t,
                    "radiologist_all": a_all_v, "radiologist_chex": a_chex_v,
                    "radiologist_chex_not_better": None if None in (a_all_v, a_chex_v)
                                                   else a_chex_v <= a_all_v}
        # the two comparisons are available on different label subsets -- Fracture has
        # no positives in the 234-image reference, so its radiologist side is None while
        # its report side is fine. Filter each independently or the sum trips on None.
        rh = [v["report_chex_higher"] for v in c.values() if v["report_chex_higher"] is not None]
        rn = [v["radiologist_chex_not_better"] for v in c.values()
              if v["radiologist_chex_not_better"] is not None]
        s6["4_chex_control"] = {
            "per_label": c,
            "report_higher_count": int(sum(rh)), "n_report": len(rh),
            "radiologist_not_better_count": int(sum(rn)), "n_radiologist": len(rn),
            "pass": sum(rh) > len(rh) / 2,
            "note": "densenet121-res224-chex learned these very report labels. If the gap is "
                    "label noise it should beat -all against the reports and not against the "
                    "radiologists."}
    else:
        s6["4_chex_control"] = {"pass": None, "note": "chex train sweep not run"}

    rep = {"reader": "torchxrayvision densenet121-res224-all",
           "n_train_frontal": int(len(fr)), "n_reference": int(len(valid)),
           "reference": "valid.csv, 234 images, radiologist-labelled",
           "test_set": s6["1_valid_vs_test_agreement"]["note"],
           "not_auditable": NOT_AUDITABLE, "labels": rows, "age": age, "S6": s6}
    with open(f"{O}/noise_report.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {O}/noise_report.json")
    return rep


if __name__ == "__main__":
    main()
