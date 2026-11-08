"""CheXpert protocol v2, amendment to sections 2 and 5.

Two changes, one reason.

v1 moved the eval-only gate onto radiologist_val (AUC >= 0.85, 95% bootstrap
CI lower bound >= 0.80) because the v0 gate ran against report-text labels and
so measured label noise rather than predictor capability. It left the MODELED
attributes on the v0 gate and recorded, without acting on it, that cardiomegaly
scores .808 [.738,.868] against radiologists with a TPR ceiling of .515.

v2 extends the same gate to the modeled attributes. Cardiomegaly fails it. That
matters more for a modeled attribute than for an eval-only one: the modeled
attributes are what CC is computed on, and a .515 TPR ceiling means a perfect
generator asked to do(cardiomegaly=1) would still be scored correct on barely
half its outputs. The number would have been bounded by the instrument, not by
the model, and no amount of ABLE training would move it. Edema passes at
.911 [.861,.952] with a .833 ceiling, so edema replaces cardiomegaly as the
second modeled finding and the graph becomes

    sex ------------------> x
    age --> effusion -----> x
     \\ ---> edema --------> x

Second change: the two labels the XRV noise audit could not check are gated
here on this project's own classifier instead.

  * support_devices -- XRV has no output for it at all.
  * enlarged_cardiomediastinum -- XRV reads it at .839, but XRV is not the
    instrument the protocol scores with.

Cardiomegaly is not re-admitted as eval-only either: the same .85 bar applies
to eval-only attributes and it does not clear it.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_predictors import BINARY, Chex, Net, predict
import scm_exact as S

MODELED_V2 = ["sex", "age_bin", "effusion", "edema"]
EVAL_ONLY_V2_CANDIDATES = ["ap_pa", "lung_opacity", "consolidation", "cardiomegaly",
                           "support_devices", "enlarged_cardiomediastinum"]
STAY_OUT = ["atelectasis", "pneumothorax", "lung_lesion", "fracture",
            "pneumonia", "pleural_other"]
AUC_MIN, CI_LO_MIN = 0.85, 0.80

# Labels with no head in best.ckpt. The checkpoint is frozen and training is
# out of scope, so these cannot be gated on this project's classifier at all.
NO_HEAD = ["enlarged_cardiomediastinum", "lung_lesion", "fracture", "pneumonia",
           "pleural_other"]


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


def boot_ovr(y, P, n=2000, seed=0):
    """Bootstrap CI for the OvR macro AUC of the 3-way age head."""
    y, P = np.asarray(y), np.asarray(P)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == P.shape[1]:
            v.append(roc_auc_score(y[i], P[i], multi_class="ovr", average="macro"))
    pt = round(float(roc_auc_score(y, P, multi_class="ovr", average="macro")), 4)
    return pt, round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)


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
    unc = pd.read_csv(f"{args.data}/label_uncertainty.csv")
    assert (unc["path"].values == meta["path"].values).all()
    train_rows = np.flatnonzero(split == 0)
    held_rows = np.flatnonzero(split == 1)
    tp = np.sort(meta.loc[train_rows, "patient_id"].unique())
    perm = np.random.default_rng(args.seed + 1).permutation(tp)
    sel_pat = set(perm[:int(round(len(tp) * args.select_frac))].tolist())
    sel_rows = train_rows[meta["patient_id"].isin(sel_pat).to_numpy()[train_rows]]

    ck = torch.load(f"{args.out}/best.ckpt", map_location="cpu")
    assert ck["binary"] == BINARY, "checkpoint head order changed"
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
    ufrac = {k: round(float(unc[f"uncertain_{k}"].mean()), 4)
             for k in unc.columns.str.removeprefix("uncertain_") if f"uncertain_{k}" in unc}
    bfrac = {k: round(float(unc[f"blank_{k}"].mean()), 4)
             for k in unc.columns.str.removeprefix("blank_") if f"blank_{k}" in unc}

    # ---------------------------------------------------------------- the gate
    gate = {}
    for j, k in enumerate(BINARY):
        a_r, lo_r, hi_r = boot(y_rad[:, j], p_rad[:, j], seed=7)
        scope = ("modeled" if k in MODELED_V2 else
                 "eval_only_candidate" if k in EVAL_ONLY_V2_CANDIDATES else "out")
        passes = bool(a_r is not None and a_r >= AUC_MIN and lo_r is not None and lo_r >= CI_LO_MIN)
        gate[k] = {
            "scope_v2": scope,
            "radiologist_val_auc": a_r, "radiologist_val_ci95": [lo_r, hi_r],
            "n_pos_radiologist_val": int(y_rad[:, j].sum()),
            "heldout_report_auc": rep["heldout"][k]["auc"],
            "v2_gate_pass": passes,
            "kept_v2": bool(passes and scope != "out"),
            "uncertain_fraction": ufrac.get(k), "blank_fraction": bfrac.get(k),
            "gate_applied": "radiologist_val AUC>=0.85 and CI_lo>=0.80 (v2: all scopes)",
        }
    a_age, lo_age, hi_age = boot_ovr(ab_rad, a_rad, seed=7)
    age_pass = bool(a_age >= AUC_MIN and lo_age >= CI_LO_MIN)
    gate["age_bin"] = {
        "scope_v2": "modeled",
        "radiologist_val_auc": a_age, "radiologist_val_ci95": [lo_age, hi_age],
        "metric": "OvR macro AUC over the three bins",
        "radiologist_val_balanced_acc": round(float(balanced_accuracy_score(ab_rad, a_rad.argmax(1))), 4),
        "heldout_report_auc": rep["heldout"]["age_bin"]["auc_ovr_macro"],
        "v2_gate_pass": age_pass, "kept_v2": age_pass,
        "uncertain_fraction": None, "blank_fraction": None,
        "gate_applied": "radiologist_val AUC>=0.85 and CI_lo>=0.80 (v2: all scopes)",
    }
    # labels with no head: not gateable on this classifier without training one
    for k in NO_HEAD:
        gate[k] = {
            "scope_v2": "eval_only_candidate" if k in EVAL_ONLY_V2_CANDIDATES else "out",
            "radiologist_val_auc": None, "radiologist_val_ci95": [None, None],
            "n_pos_radiologist_val": None,
            "heldout_report_auc": None,
            "v2_gate_pass": False, "kept_v2": False,
            "uncertain_fraction": ufrac.get(k), "blank_fraction": bfrac.get(k),
            "gate_applied": "NOT GATEABLE: best.ckpt has no head for this label and "
                            "training one is out of scope, so there is no number to gate on. "
                            "Dropped under the stated fallback (fails the gate -> out).",
        }

    kept_modeled = [k for k in MODELED_V2 if gate[k]["kept_v2"]]
    kept_eval = [k for k in EVAL_ONLY_V2_CANDIDATES if gate.get(k, {}).get("kept_v2")]
    assert set(kept_modeled) == set(MODELED_V2), \
        f"a modeled attribute failed the v2 gate: {set(MODELED_V2) - set(kept_modeled)}"
    kept = kept_modeled + kept_eval

    # --------------------------------------------- T6 on radiologist_val, v2 set
    thr = {}
    t6 = {}
    for j, k in enumerate(BINARY):
        if k not in kept:
            continue
        fpr, tpr, t = roc_curve(y_sel[:, j], p_sel[:, j])
        th = float(t[int(np.argmax(tpr - fpr))])
        thr[k] = th
        out = {"threshold": round(th, 4),
               "threshold_tuned_on": "select set (inside train split), report labels",
               "role": "modeled" if k in MODELED_V2 else "eval_only"}
        for name, y, p in (("radiologist_val", y_rad[:, j], p_rad[:, j]),
                           ("heldout_report_labels", y_hel[:, j], p_hel[:, j])):
            pos, neg = y == 1, y == 0
            out[name] = {
                "tpr_ceiling": round(float((p[pos] >= th).mean()), 4) if pos.any() else None,
                "fpr_floor": round(float((p[neg] >= th).mean()), 4) if neg.any() else None,
                "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
        out["new_modeled_node_in_v2"] = (k == "edema")
        t6[k] = out
    t6["age_bin"] = {
        "role": "modeled",
        "radiologist_val_accuracy_ceiling": round(float((a_rad.argmax(1) == ab_rad).mean()), 4),
        "heldout_accuracy_ceiling": rep["t6_calibration"]["age_bin"]["accuracy_ceiling"]}

    # ------------------------------------------------- r_k(a) for the v2 attrs
    # For edema the association is measured on CERTAIN held-out rows only, the
    # same pool the cohort sources come from -- otherwise r_k(edema) would be
    # computed across a group boundary that 1,107 folded labels sit on the wrong
    # side of.
    hm = meta.iloc[held_rows].reset_index(drop=True)
    hu = unc.iloc[held_rows].reset_index(drop=True)
    r = {}
    for a in MODELED_V2:
        va = hm[a].to_numpy()
        sub = np.ones(len(hm), bool)
        if a in S.CERTAIN_REQUIRED:
            sub = hu[f"uncertain_{a}"].to_numpy() == 0
        r[a] = {"measured_on": ("certain-label held-out rows only"
                               if a in S.CERTAIN_REQUIRED else "all held-out rows"),
                "n": int(sub.sum())}
        for j, k in enumerate(BINARY):
            if k == a or k not in kept:
                continue
            means = {str(g): round(float(p_hel[sub & (va == g), j].mean()), 4)
                     for g in sorted(np.unique(va[sub]).tolist())}
            r[a][k] = {"mean_prob_by_group": means,
                       "max_shift": round(float(max(means.values()) - min(means.values())), 4)}
        if a != "age_bin":
            r[a]["age_bin_mean_predicted"] = {
                "mean_bin_by_group": {str(g): round(float(a_hel[sub & (va == g)].argmax(1).mean()), 4)
                                      for g in sorted(np.unique(va[sub]).tolist())}}

    # ------------------------------------------------------------------ record
    rep["version"] = "v2"
    rep["amendment_v2"] = {
        "date": "2026-11-08",
        "change": "the radiologist_val gate (AUC >= 0.85, 95% bootstrap CI lower bound >= 0.80) "
                  "now applies to MODELED attributes as well as eval-only ones. Cardiomegaly "
                  "fails it and is replaced by edema as the second modeled finding; the graph "
                  "becomes sex->x, age->effusion->x, age->edema->x.",
        "reason": "v1 scoped the radiologist gate to eval-only attributes and recorded, without "
                  "acting on it, that cardiomegaly scores .808 [.738,.868] against radiologists "
                  "with a TPR ceiling of .515 at its Youden threshold. For a MODELED attribute "
                  "that ceiling is not a reporting caveat, it is a cap on the headline result: "
                  "CC on do(cardiomegaly) would have been bounded by what the predictor can see "
                  "in a real radiograph, not by what ABLE generates, and no training would move "
                  "it. Edema passes at .911 [.861,.952] with a .833 ceiling.",
        "second_change": "support_devices and enlarged_cardiomediastinum, the two labels the XRV "
                         "noise audit could not check (no XRV output; XRV is not the scoring "
                         "instrument), are gated here on this project's own classifier under the "
                         "same rule.",
        "cardiomegaly_disposition": "out of the protocol entirely -- it fails the gate as a "
                                    "modeled attribute and fails the same .85 bar as an "
                                    "eval-only one. Its column stays in the frozen CSV.",
        "row_set_note": "chexpert_meta.csv, split.npy and images_224.npy are UNCHANGED from v0. "
                        "Promoting edema under the section-3 filter would have dropped its "
                        "10,105 uncertain rows and invalidated every row index in the memmap, "
                        "which cannot be rebuilt. The certainty policy for a modeled finding is "
                        "applied downstream instead: edema's mechanism is fitted only on certain "
                        "rows (8,998 train rows excluded) and no cohort source carries an "
                        "uncertain edema label (1,107 held-out rows excluded from the pool).",
        "stay_out": STAY_OUT,
    }
    rep["v2_gate"] = gate
    # idempotent: a second run must not overwrite the v1 record with the v2 one
    rep["kept_labels_v1"] = rep.get("kept_labels_v1") or rep["kept_labels"]
    rep["kept_labels"] = kept
    rep["modeled_v2"] = MODELED_V2
    rep["eval_only_v2"] = kept_eval
    rep["dropped_by_gate_v2"] = [k for k in list(BINARY) + NO_HEAD if k not in kept]
    rep["t6_calibration_v2"] = t6
    rep["thresholds_youden_on_select_v2"] = {k: round(v, 4) for k, v in thr.items()}
    rep["fc_correlation_correction_r_k_a_v2"] = r
    rep["uncertain_fraction_frozen_rowset"] = ufrac
    rep["blank_fraction_frozen_rowset"] = bfrac
    with open(f"{args.out}/predictor_report.json", "w") as f:
        json.dump(rep, f, indent=2)

    # --------------------------------------------------------------- the table
    order = MODELED_V2 + [k for k in EVAL_ONLY_V2_CANDIDATES]
    hdr = ("%-28s %-11s %-24s %-19s %-19s %7s %7s %s"
           % ("attribute", "role", "radiologist AUC [95% CI]", "T6 rad TPR/FPR",
              "T6 held TPR/FPR", "unc%", "blank%", "v2"))
    print(hdr); print("-" * len(hdr))
    for k in order:
        g = gate[k]
        a = g["radiologist_val_auc"]
        ci = g["radiologist_val_ci95"]
        auc_s = f"{a:.4f} [{ci[0]:.3f},{ci[1]:.3f}]" if a is not None else "no head -- not gateable"
        if k in t6:
            rr, hh = t6[k].get("radiologist_val", {}), t6[k].get("heldout_report_labels", {})
            t6r = (f"{rr['tpr_ceiling']:.3f}/{rr['fpr_floor']:.3f}"
                   if rr.get("tpr_ceiling") is not None else "-")
            t6h = (f"{hh['tpr_ceiling']:.3f}/{hh['fpr_floor']:.3f}"
                   if hh.get("tpr_ceiling") is not None else "-")
        elif k == "age_bin":
            t6r = f"{t6['age_bin']['radiologist_val_accuracy_ceiling']:.3f} acc"
            t6h = f"{t6['age_bin']['heldout_accuracy_ceiling']:.3f} acc"
        else:
            t6r = t6h = "-"
        u = g["uncertain_fraction"]; b = g["blank_fraction"]
        print("%-28s %-11s %-24s %-19s %-19s %7s %7s %s" % (
            k, g["scope_v2"].replace("_candidate", ""), auc_s, t6r, t6h,
            "-" if u is None else f"{u*100:.2f}", "-" if b is None else f"{b*100:.2f}",
            "KEPT" if g["kept_v2"] else "OUT"))
    print(f"\nmodeled:    {MODELED_V2}")
    print(f"eval-only:  {kept_eval}")
    print(f"out:        {rep['dropped_by_gate_v2']}")
    print(f"\nupdated {args.out}/predictor_report.json")


if __name__ == "__main__":
    main()
