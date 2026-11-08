# CheXpert · data protocol v2 — FROZEN 2026-11-08

Supersedes v1 (2026-11-07) and v0 (2026-11-07). No training was run for this
version. `best.ckpt` is the v0 checkpoint, untouched.

## What changed

**1. The radiologist_val gate now applies to modeled attributes.**
v1 moved the eval-only gate onto radiologist-labelled images (AUC ≥ 0.85, 95%
bootstrap CI lower bound ≥ 0.80) and left the modeled attributes on the v0
gate, recording but not acting on cardiomegaly's .808 [.738,.868]. v2 applies
one rule to every attribute. Cardiomegaly fails.

For an eval-only attribute a weak gate is a reporting caveat. For a modeled one
it is a cap on the headline number. CC is computed on the modeled attributes,
and cardiomegaly's TPR ceiling on real radiologist-labelled images is **.515**
at its Youden threshold: a perfect generator asked to `do(cardiomegaly=1)`
would still be scored wrong on nearly half its outputs. The bound is the
instrument, not the model, and no amount of ABLE training moves it.

**Edema replaces cardiomegaly** — .9111 [.861,.952], TPR ceiling .833.

```
  sex -----------------------> x
  age --> effusion ----------> x
   \ ---> edema -------------> x
```

Cardiomegaly leaves the protocol entirely: it fails the gate as a modeled
attribute and fails the same .85 bar as an eval-only one. Its column stays in
the frozen CSV, unused.

**2. The two labels the XRV noise audit could not check were gated here on this
project's own classifier.**

| label | result |
|---|---|
| `support_devices` | **.8483 [.793,.897] → OUT.** Fails both conditions. XRV has no output for it, so the audit could not reach it; this classifier can, and the answer is no. |
| `enlarged_cardiomediastinum` | **OUT — no number exists.** `best.ckpt` has 10 binary heads and this is not one of them. Gating it on this project's classifier means training a head, which is out of scope for this version. Dropped under the stated fallback. XRV reads it at .839 on 109 positives, but XRV is not the scoring instrument, which is the whole reason step 2 was asked for. |

**3. The row set did not change, and could not have.**
Promoting edema to modeled means applying §3 step 3 (uncertain `-1` **drops the
row**) instead of step 4 (uncertain `-1` → 0). That would drop edema's 10,105
uncertain rows — and `images_224.npy` is aligned to the v0 row set by index,
cannot be rebuilt (the 11.5 GB archive is deleted to make room for it), and the
frozen predictor's splits are defined on those same indices. Refiltering would
have invalidated the memmap, the split and the checkpoint at once.

So the certainty policy for a modeled finding is enforced **downstream**:

- `build_meta.py` now also writes `label_uncertainty.csv`, carrying the `-1` and
  blank flags that step 4's fold destroys, row-for-row aligned with the frozen
  CSV. It does not touch `chexpert_meta.csv`, whose bytes are unchanged.
- edema's mechanism is fitted only on certain rows — **8,998 train rows
  excluded**. Fitting `P(edema=1|age)` on a column where 5.88% of rows say
  "absent" because the report said "uncertain" would bias every conditional
  downward and drag every `do(age)` propagation with it.
- **no cohort source carries an uncertain edema label** — 1,107 held-out rows
  excluded from the source pool. This restricts eligibility, not the data, and
  it is not attribute selection: which attributes are modeled was decided on
  radiologist_val, never on the cohort.

§7 test 2 asserts all three, and would have passed vacuously without the
sidecar — the folded column contains no `-1` to find.

## The kept set

7 attributes, down from 8 in v1.

| attribute | role | radiologist AUC [95% CI] | T6 rad TPR/FPR | T6 held TPR/FPR | unc % | blank % |
|---|---|---|---|---|---|---|
| `sex` | modeled | .9945 [.986, 1.000] | .972 / .021 | .969 / .026 | — | — |
| `age_bin` | modeled | .9523 [.930, .972] † | .815 acc | .794 acc | — | — |
| `effusion` | modeled | .9049 [.858, .944] | .656 / .118 | .813 / .233 | 0.00 | 43.63 |
| `edema` | modeled ‡ | .9111 [.861, .952] | .833 / .133 | .788 / .287 | 5.88 | 59.71 |
| `ap_pa` | eval-only | 1.0000 [1.000, 1.000] | 1.000 / .006 | .963 / .003 | — | — |
| `lung_opacity` | eval-only | .9027 [.856, .943] | .733 / .131 | .829 / .510 | 2.03 | 45.94 |
| `consolidation` | eval-only | .9126 [.866, .950] | .844 / .161 | .687 / .371 | 12.10 | 70.72 |

† OvR macro over the three bins. ‡ new modeled node in v2.

Thresholds are Youden's J on the **select** set (inside the train split, report
labels) — never on held-out, never on radiologist_val, never on the cohort. So
the radiologist TPR/FPR columns are a genuinely out-of-sample reading of a
report-calibrated threshold.

**Out**, with the number that put them there:

| label | radiologist AUC [CI] | why |
|---|---|---|
| `cardiomegaly` | .8080 [.738, .868] | fails as modeled **and** as eval-only; TPR ceiling .515 |
| `support_devices` | .8483 [.793, .897] | fails both conditions |
| `atelectasis` | .7612 [.690, .823] | fails |
| `pneumothorax` | .8110 [.601, .972] | 6 positives in radiologist_val; CI floor |
| `enlarged_cardiomediastinum` | no head | not gateable without training |
| `lung_lesion`, `fracture`, `pneumonia`, `pleural_other` | no head | stay out |

## Fitted mechanism

`P(sex) = [.4145, .5855]`, `P(age_bin) = [.1902, .3942, .4155]`

| age_bin | P(effusion=1) | P(edema=1) |
|---|---|---|
| <45 | .3121 | .1921 |
| 45–65 | .4222 | .2674 |
| >65 | .4796 | .3316 |

Both monotone increasing (§7 test 4). Effusion's conditionals are **unchanged
from v0/v1** — its fit is unmasked and the row set did not move.

## No edge between the findings

φ(effusion, edema) = **+0.122** on certain-label rows, against **+0.041** for
v0's effusion/cardiomegaly pair. Both findings are driven by an unmodeled
common cause — volume overload in cardiac failure — so an arrow between them
would make every `do(effusion)` counterfactual move edema for the wrong reason
and the FC score would measure a mechanism that does not exist. The residual
dependence goes to `r_k(a)` instead, **which therefore matters about three
times as much in v2 as it did in v0.**

## r_k(a) — what FC must discount

Largest shifts in the predictor's mean reading across the groups of `a`, on real
held-out images (`a = edema` measured on certain-label rows only, the same pool
the cohort draws from):

| a | k | shift | means |
|---|---|---|---|
| `edema` | `ap_pa` | **.1742** | .2286 → .0544 |
| `effusion` | `lung_opacity` | .1640 | .4037 → .5677 |
| `effusion` | `consolidation` | .1598 | .2909 → .4507 |
| `effusion` | `edema` | .1546 | .3592 → .5138 |
| `age_bin` | `edema` | .1545 | .3278 → .4823 |
| `age_bin` | `effusion` | .1530 | .3592 → .5122 |
| `edema` | `effusion` | .1375 | .4163 → .5538 |

**Flagged:** `r_ap_pa(edema) = .1742` is the largest correction in the protocol
and it is a confound, not a finding. Patients with pulmonary edema are imaged
AP at the bedside because they are too sick to stand for a PA film. So
`do(edema=1)` will drag the `ap_pa` reading downward in any generator that has
learned real radiographs, and FC will charge that to the model unless .1742 is
discounted. `ap_pa` is the eval-only attribute with a perfect AUC, so it is
exactly the one where an uncorrected FC drop would look most like real damage.

## Frozen artifacts

**Byte-identical to v0 and v1** — this is the proof v2 changed only the graph
and the gate:

| artifact | sha256 |
|---|---|
| `chexpert_meta.csv` | `025b913e…f5e34e28` |
| `split.npy` | `60d82455…36e35cd2` |
| `images_224.npy` | `6f134ad1…cad068e5` |
| `radiologist_val_224.npy` | `bedbb253…2da3c781` |
| `best.ckpt` | `75585762…eda66ce7` |

**New or changed in v2:**

| artifact | note |
|---|---|
| `label_uncertainty.csv` | new; 171,996 rows × 13 labels, `-1`/blank flags · `28647ae2…0ea0ab4e` |
| `scm_params.json` | refitted: edema replaces cardiomegaly, masked fit · `57083776…60a46081` |
| `eval_cohort.json` | rebuilt: `do_edema` replaces `do_cardiomegaly`, 24 cells of (sex, age_bin, effusion, edema) · `f73990a3…a6f5ad36` |
| `predictor_report.json` | `v2_gate`, `t6_calibration_v2`, `fc_correlation_correction_r_k_a_v2`, `amendment_v2` |
| `tests_report.json` | 7 passed, 0 failed |

Cohort: 512 held-out sources × 4 interventions = 2,048 counterfactuals, 21–22
per cell over all 24 cells (smallest cell had 127 eligible images). `do(age)`
moved a finding on **107 of 512** sources.

## §7 — 7 passed, 0 failed

1. **split disjointness** — 55,266 train / 6,141 held-out patients, zero
   overlap; 200 radiologist_val patients disjoint from both; zero cohort
   sources outside held-out.
2. **label policy** — no `-1`, no NaN; and the v2 certainty check: 10,105
   uncertain edema rows, 8,998 excluded from the SCM fit, **0 in the cohort**.
3. **prevalence** — every label within ±5 points of `train.csv`'s own rates.
4. **graph sanity** — effusion and edema both monotone across the three age
   bins, each measured on the rows its own mechanism is fitted on.
5. **predictor gate** — every modeled attribute clears the v2 gate; every kept
   eval-only attribute clears it; and the modeled set in the report, in
   `scm_exact.ORDER` and in this protocol agree, so the gate cannot be read
   against a graph the cohort was not built from.
6. **image integrity** — 171,996 rows, 0 constant, contact sheet inspected.
7. **determinism** — `chexpert_meta.csv`, `label_uncertainty.csv`, `split.npy`,
   `scm_params.json` and `eval_cohort.json` all rebuild to identical sha256.
   `images_224.npy` is checked for integrity, not determinism: rebuilding it
   needs the deleted archive.

## Carried over from the v1 noise audit

Unchanged and still relevant: the audit's reader (`densenet121-res224-all`) was
trained on NIH + PadChest + **CheXpert** + MIMIC + RSNA, so it is only partly
independent of these report labels and AUC-against-report is biased upward.
`pneumothorax` was not auditable (8 positives). Two S6 checks failed; see
`noise_report.md`.
