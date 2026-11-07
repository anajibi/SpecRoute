"""Augment noise_report.json with the comparable gap, and render noise_report.md.

The plan defines gap = AUC(radiologist) - AUC(report). Computed literally, the
report side uses only rows carrying an explicit 0 or 1 -- which is 20-60% of
the training set, and exactly the rows where the report NAMED the finding,
i.e. the unambiguous ones. The radiologist side labels all 234 images,
subtle cases included. So the two sides are not measuring the same task and
the literal gap comes out negative for most labels.

Recomputing the report side under the convention the protocol actually uses
for eval-only labels -- blank counts as negative -- puts both sides on all
rows and flips the picture. Both numbers are reported; neither is hidden, and
the literal one is kept first so the plan's metric is still readable.
"""
import json
import numpy as np

O = "/home/exouser/SpecRoute/outputs/chexpert_noise"
r = json.load(open(f"{O}/noise_report.json"))

for x, v in r["labels"].items():
    a_b, a_r = v["auc_vs_report_blank_as_negative"], v["auc_vs_radiologist_valid"]
    v["gap_blank_as_negative"] = None if None in (a_b, a_r) else round(a_r - a_b, 4)
    g, d = v["gap_blank_as_negative"], v["confident_disagree_calibrated"]
    v["verdict_blank_as_negative"] = (
        "not auditable" if not v["estimable"] else
        "unreliable" if (g is not None and g > 0.15) or d > 0.15 else
        "noisy" if g is not None and g >= 0.05 else
        "clean" if g is not None and d < 0.05 else "noisy")

r["gap_definitions"] = {
    "gap": "AUC(radiologist, all 234) - AUC(report, rows with an explicit 0 or 1). The report "
           "side here is restricted to rows where the labeller named the finding, which are "
           "the unambiguous ones, so this comparison favours the report side.",
    "gap_blank_as_negative": "AUC(radiologist, all 234) - AUC(report, ALL frontal rows with "
                             "blank treated as negative). Both sides then score every image, "
                             "and this is the convention the protocol uses for eval-only "
                             "labels. This is the comparable number.",
}
r["independence_caveat"] = (
    "densenet121-res224-all was trained on NIH + PadChest + CheXpert + MIMIC + RSNA. Its "
    "CheXpert component was trained on these very report labels, so the reader is only "
    "partly independent of the thing being audited, and AUC-against-report is biased upward. "
    "That bias pushes the literal gap negative and is the likeliest reason it comes out so.")
json.dump(r, open(f"{O}/noise_report.json", "w"), indent=2)

L = r["labels"]
S6 = r["S6"]
out = []
w = out.append
w("# CheXpert label-noise audit\n")
w(f"Reader: **{r['reader']}** (no training). Reference: **{r['reference']}**.  ")
w(f"Training sweep: **{r['n_train_frontal']:,}** frontal images.\n")
w("## Headline table\n")
w("| label | AUC vs report (explicit rows) | AUC vs report (blank→0) | AUC vs radiologist | gap (plan's defn) | **gap (comparable)** | confident-disagree | radiologist false-flag | excess | uncertain % | blank % | verdict |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for x, v in L.items():
    f = lambda z: "—" if z is None else f"{z:.4f}" if isinstance(z, float) else str(z)
    pc = lambda z: "—" if z is None else f"{100*z:.2f}%"
    w(f"| {x} | {f(v['auc_vs_report_train'])} | {f(v['auc_vs_report_blank_as_negative'])} | "
      f"{f(v['auc_vs_radiologist_valid'])} | {f(v['gap'])} | **{f(v['gap_blank_as_negative'])}** | "
      f"{pc(v['confident_disagree_calibrated'])} | {pc(v['radiologist_false_flag_rate_calibrated'])} | "
      f"{pc(v['excess_over_radiologist_baseline'])} | {pc(v['uncertain_frac'])} | "
      f"{pc(v['blank_frac'])} | {v['verdict_blank_as_negative']}"
      f"{' — ' + v['flag'] if v['flag'] else ''} |")
w("\n`Support Devices`: " + r["not_auditable"]["Support Devices"] + ".\n")
w("## Model-error floor (fixed 0.9 / 0.1 thresholds)\n")
w("Reported because the plan asks for it, and labelled a floor rather than a label-error rate: "
  "at these thresholds the metric flags this share of the RADIOLOGIST set, where by construction "
  "there is no report-label error to find.\n")
w("| label | train confident-disagree | radiologist false-flag (= reader error) |")
w("|---|---|---|")
for x, v in L.items():
    f = v["model_error_floor_fixed_0.9_0.1"]
    w(f"| {x} | {100*f['confident_disagree_train']:.2f}% | {100*f['radiologist_false_flag_rate']:.2f}% |")
w("\nCalibrated per-label thresholds (hi = 99th percentile of probability among radiologist "
  "negatives, lo = 1st percentile among radiologist positives) hold the false-flag rate at "
  "≤1.73%, which is what makes the train-side number interpretable.\n")
w("## S6 self-consistency\n")
for k, v in S6.items():
    mark = {True: "PASS", False: "**FAIL**", None: "N/A"}[v.get("pass")]
    w(f"- **{k}** — {mark}. {v.get('note','')}")
w("")
a = r["age"]
w("## Age (sanity only; age labels do not come from reports)\n")
w(f"MAE **{a['mae_years']} years**, bias {a['bias_years']}, bin accuracy {a['bin_accuracy']} "
  f"over {a['n']:,} images. Bin confusion (rows = CSV truth, cols = predicted):\n")
w("| | <45 | 45–65 | >65 |")
w("|---|---|---|---|")
for i, nm in enumerate(["<45", "45–65", ">65"]):
    w(f"| **{nm}** | " + " | ".join(str(c) for c in a["bin_confusion_rows_true_cols_pred"][i]) + " |")
w("")
open(f"{O}/noise_report.md", "w").write("\n".join(out) + "\n")
print(f"wrote {O}/noise_report.md")
