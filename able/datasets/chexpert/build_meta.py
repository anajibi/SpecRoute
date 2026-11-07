"""CheXpert protocol v0, section 3: filtering -> chexpert_meta.csv.

Reads train.csv straight out of the Kaggle archive (nothing is extracted; the
archive is ~11 GB and the disk cannot hold a JPEG extraction beside the image
memmap). Every filter step logs the row count before and after, which is what
section 7's prevalence and label-policy tests are checked against.

Filters, applied in the order the protocol fixes them:

  1. frontal views only
  2. Sex in {Female, Male}; Age present and 18 <= age <= 100
  3. drop rows carrying -1 (uncertain) on Pleural Effusion or Cardiomegaly;
     blank -> 0 on those two
  4. eval-only labels: -1 -> 0, blank -> 0
  5. one image per study, the first frontal view listed for that study
  6. write the metadata CSV

Step 3 drops uncertainty rather than modelling it: the two modeled findings
have to be binary for the SCM's logistic mechanisms to be identified, and an
uncertain label is not a third value of the finding, it is missing information
about a binary one. Step 4 folds uncertainty into the negative class instead,
because the eval-only labels are only ever read through a predictor whose AUC
is measured against exactly this convention.
"""
import argparse, hashlib, json, logging, os, sys, zipfile
import numpy as np
import pandas as pd

MODELED_FINDINGS = {"effusion": "Pleural Effusion", "cardiomegaly": "Cardiomegaly"}
EVAL_ONLY_FINDINGS = {
    "support_devices": "Support Devices",
    "edema": "Edema",
    "lung_opacity": "Lung Opacity",
    "consolidation": "Consolidation",
    "atelectasis": "Atelectasis",
    "pneumothorax": "Pneumothorax",
}
OUT_COLS = [
    "path", "patient_id", "study_id", "sex", "age_years", "age_bin",
    "effusion", "cardiomegaly", "support_devices", "ap_pa", "edema",
    "lung_opacity", "consolidation", "atelectasis", "pneumothorax",
]


def age_bin(a):
    """<45 -> 0, 45-65 inclusive -> 1, >65 -> 2."""
    return np.where(a < 45, 0, np.where(a > 65, 2, 1)).astype(np.int8)


def parse_path(paths):
    """CheXpert-v1.0-small/train/patient00065/study2/view1_frontal.jpg -> (65, 2)."""
    parts = paths.str.split("/", expand=True)
    # fullmatch yields True/False/NaN -- NaN is "this row has no such component",
    # so it has to become False. .notna() here would count False as a match.
    def flags(pattern):
        return parts.apply(lambda c: c.str.fullmatch(pattern).fillna(False).astype(bool)).values
    pat = flags(r"patient(\d+)")
    sty = flags(r"study(\d+)")
    if not (pat.sum(axis=1) == 1).all() or not (sty.sum(axis=1) == 1).all():
        bad = paths[(pat.sum(axis=1) != 1) | (sty.sum(axis=1) != 1)].head(3).tolist()
        raise ValueError(f"paths without exactly one patientN and one studyN component: {bad}")
    pcol = pat.argmax(axis=1)
    scol = sty.argmax(axis=1)
    arr = parts.values
    pid = np.array([int(arr[i, pcol[i]][7:]) for i in range(len(paths))], dtype=np.int32)
    sid = np.array([int(arr[i, scol[i]][5:]) for i in range(len(paths))], dtype=np.int16)
    return pid, sid


def read_csv(args):
    if args.zip:
        with zipfile.ZipFile(args.zip) as z:
            names = [n for n in z.namelist() if os.path.basename(n) == os.path.basename(args.csv)]
            if len(names) != 1:
                raise SystemExit(f"{os.path.basename(args.csv)}: {len(names)} matches in archive: {names[:5]}")
            logging.info("reading %s out of %s", names[0], args.zip)
            with z.open(names[0]) as f:
                return pd.read_csv(f), names[0]
    return pd.read_csv(args.csv), args.csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None, help="Kaggle archive; CSV is read from inside it")
    ap.add_argument("--csv", default="train.csv", help="CSV path, or its basename when --zip is given")
    ap.add_argument("--out", required=True, help="destination chexpert_meta.csv")
    ap.add_argument("--counts-out", default=None, help="filter counts JSON (default: <out>.counts.json)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df, src = read_csv(args)
    counts = {"source": src, "rows_in": len(df)}
    logging.info("step 0  rows in CSV                      %8d", len(df))

    need = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"]
    need += list(MODELED_FINDINGS.values()) + list(EVAL_ONLY_FINDINGS.values())
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"columns absent from {src}: {missing}")

    # 1. frontal only
    counts["lateral_dropped"] = int((df["Frontal/Lateral"] != "Frontal").sum())
    df = df[df["Frontal/Lateral"] == "Frontal"].copy()
    logging.info("step 1  frontal only                     %8d  (-%d lateral)", len(df), counts["lateral_dropped"])
    counts["after_frontal"] = len(df)

    # 2. sex and age
    before = len(df)
    sex_ok = df["Sex"].isin(["Female", "Male"])
    counts["sex_unknown_dropped"] = int((~sex_ok).sum())
    df = df[sex_ok]
    age = pd.to_numeric(df["Age"], errors="coerce")
    age_ok = age.notna() & (age >= 18) & (age <= 100)
    counts["age_missing_or_out_of_range_dropped"] = int((~age_ok).sum())
    df = df[age_ok].copy()
    df["age_years"] = pd.to_numeric(df["Age"]).astype(np.int16)
    logging.info("step 2  sex and age 18-100               %8d  (-%d sex, -%d age)",
                 len(df), counts["sex_unknown_dropped"], counts["age_missing_or_out_of_range_dropped"])
    counts["after_sex_age"] = len(df)
    assert len(df) <= before

    # 3. modeled findings: -1 drops the row, blank is a negative
    for key, col in MODELED_FINDINGS.items():
        v = pd.to_numeric(df[col], errors="coerce")
        counts[f"{key}_uncertain_dropped"] = int((v == -1).sum())
        counts[f"{key}_blank_to_zero"] = int(v.isna().sum())
        df = df[v != -1].copy()
    for key, col in MODELED_FINDINGS.items():
        df[key] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int8)
    logging.info("step 3  modeled findings certain         %8d  (-%d effusion -1, -%d cardiomegaly -1)",
                 len(df), counts["effusion_uncertain_dropped"], counts["cardiomegaly_uncertain_dropped"])
    counts["after_modeled_certain"] = len(df)

    # 4. eval-only labels: uncertain and blank both become negatives
    for key, col in EVAL_ONLY_FINDINGS.items():
        v = pd.to_numeric(df[col], errors="coerce")
        counts[f"{key}_uncertain_to_zero"] = int((v == -1).sum())
        counts[f"{key}_blank_to_zero"] = int(v.isna().sum())
        df[key] = v.replace(-1, 0).fillna(0).astype(np.int8)
    # ap_pa is eval-only too, so a blank view position also becomes 0 (= AP).
    # That is the protocol's rule, but it is a real conflation and the count is
    # logged so the predictor report can be read with it in mind.
    ap = df["AP/PA"]
    counts["ap_pa_blank_to_zero"] = int(ap.isna().sum())
    counts["ap_pa_other_to_zero"] = int((~ap.isna() & ~ap.isin(["AP", "PA"])).sum())
    df["ap_pa"] = (ap == "PA").astype(np.int8)
    logging.info("step 4  eval-only labels folded          %8d  (ap_pa: %d blank, %d neither AP nor PA -> 0)",
                 len(df), counts["ap_pa_blank_to_zero"], counts["ap_pa_other_to_zero"])

    # 5. one image per study, first frontal listed
    df["patient_id"], df["study_id"] = parse_path(df["Path"])
    before = len(df)
    df = df.drop_duplicates(subset=["patient_id", "study_id"], keep="first").copy()
    counts["extra_views_per_study_dropped"] = before - len(df)
    logging.info("step 5  one image per study              %8d  (-%d extra frontal views)",
                 len(df), counts["extra_views_per_study_dropped"])
    counts["after_one_per_study"] = len(df)

    # 6. write
    df["sex"] = (df["Sex"] == "Male").astype(np.int8)        # F=0, M=1
    df["age_bin"] = age_bin(df["age_years"].values)
    df["path"] = df["Path"]
    out = df[OUT_COLS].reset_index(drop=True)

    if out.isna().any().any():
        raise SystemExit(f"NaN survived filtering in {out.columns[out.isna().any()].tolist()}")
    for c in OUT_COLS:
        if c in ("path",):
            continue
        if (pd.to_numeric(out[c]) == -1).any():
            raise SystemExit(f"a -1 survived filtering in column {c}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    counts["rows_out"] = len(out)
    counts["patients"] = int(out["patient_id"].nunique())
    counts["sha256"] = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    counts["prevalence"] = {c: round(float(out[c].mean()), 6) for c in OUT_COLS
                            if c not in ("path", "patient_id", "study_id", "age_years", "age_bin")}
    counts["age_bin_counts"] = {str(k): int(v) for k, v in out["age_bin"].value_counts().sort_index().items()}
    counts["finding_rate_by_age_bin"] = {
        k: {str(b): round(float(g[k].mean()), 6) for b, g in out.groupby("age_bin")}
        for k in MODELED_FINDINGS
    }

    cpath = args.counts_out or args.out + ".counts.json"
    with open(cpath, "w") as f:
        json.dump(counts, f, indent=2)
    logging.info("step 6  wrote %s  (%d rows, %d patients)", args.out, len(out), counts["patients"])
    logging.info("        sha256 %s", counts["sha256"])
    logging.info("        counts -> %s", cpath)
    for k in MODELED_FINDINGS:
        rates = counts["finding_rate_by_age_bin"][k]
        logging.info("        %-13s by age bin: %s", k, "  ".join(f"{b}:{r:.4f}" for b, r in rates.items()))


if __name__ == "__main__":
    sys.exit(main())
