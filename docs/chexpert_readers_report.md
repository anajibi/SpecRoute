# Three readers vs radiologists — CheXpert, 2026-11-08

**Measurement only. The frozen v2 modeled and eval-only sets are unchanged.** `best.ckpt`
was read, never written (sha256 `75585762…eda66ce7`, verified against `SHA256SUMS`).

All numbers are AUC against the radiologist labels in `valid.csv`, on the same 200 frontal
images of `radiologist_val`, with 2,000-resample percentile bootstrap CIs, seed 7, identical
procedure for every reader. Reference gate: **AUC ≥ 0.85 and CI lower bound ≥ 0.80**.

## The table

| attribute | XRV `-all` | CheXzero 0-shot | ours (`best.ckpt`, 10-head) | ours (11-head) | rad. pos |
|---|---|---|---|---|---|
| `cardiomegaly` | .8063 [.741, .863] | **.8544** [.797, .904] | .8080 [.738, .868] | .8155 [.748, .875] | 66/200 |
| `enlarged_cardiomediastinum` | .8292 [.767, .883] | **.9160** [.876, .950] | — no head — | **.5533** [.470, .633] | 105/200 |
| `support_devices` | — no output — | .7855 [.718, .845] | .8483 [.793, .897] | .8270 [.767, .882] | 99/200 |

**Not one of the three clears the gate on our own classifier.** CheXzero clears it on two.

### The readers, and how independent each one is

| reader | independence |
|---|---|
| XRV `densenet121-res224-all` | **Partly compromised.** Its training mix includes CheXpert, so it learned these very report labels. No `Support Devices` output at all. |
| **CheXzero**, zero-shot, 10-checkpoint ensemble, prompt pair `("{}", "no {}")` | **The most independent reader here.** Trained on MIMIC-CXR report *text*, not on CheXpert and not on any label column. |
| ours, `best.ckpt` | the frozen v2 predictor, 10 binary heads + age |
| ours, 11-head | same architecture, recipe, split, seed, plus an 11th head. Trunk disturbance vs `best.ckpt`: mean \|Δ\| **.0097**, max .0442 — the ten shared heads barely moved, so the new head's number is comparable. |

**CheXzero pipeline gate.** CheXzero's native preprocessing letterboxes the whole radiograph
onto 320×320 and resizes to 224; `radiologist_val_224.npy` is resize-short-side-224 plus a
224 centre crop, and the source JPEGs went with the deleted archive. So CheXzero was scored
on our crops — which has the compensating virtue that it and our classifier see **identical
pixels**. Before trusting the result, all five CheXpert competition pathologies were scored
the same way: **Atelectasis .7845, Cardiomegaly .8544, Consolidation .8839, Edema .9096,
Pleural Effusion .9034 — mean .8672, min .7845.** That is CheXzero's published range, so the
deviation did not break the pipeline.

## `support_devices` — the errors are false negatives

At the protocol's Youden-on-select threshold (.4386), on 200 images:

| | |
|---|---|
| true pos | 64 |
| **false neg (device present, predicted absent)** | **35** |
| false pos | 17 |
| true neg | 84 |
| errors | 52 → **67% are false negatives** |
| TPR / FPR | .6465 / .1683 |

At 0.5: 51 errors, **39 false negatives (76%)**. **The head is under-sensitive, not
over-eager** — it misses hardware that is there.

The likely cause, stated as inference rather than measurement: 39.9% of the frozen row set
has a *blank* `Support Devices` cell, which the protocol folds to 0. A radiology report
mentions a line or a tube only when it is clinically worth mentioning, so a large share of
those silent negatives do have hardware, and the head is trained to call them absent.
CheXzero, which never saw a CheXpert label, reads the label *worse* than we do (.7855 vs
.8483) — so this is not simply our label noise; support devices appear to be a genuinely
harder visual target than their 49.5% prevalence suggests.

(`report_vs_radiologist_disagree = 0` on this set is vacuous — `radiologist_val`'s label
column *is* `valid.csv`'s radiologist label. It confirms the join, nothing more.)

## `enlarged_cardiomediastinum` — the label is a different concept, not a noisy one

Our new head reads it at **.5533 [.470, .633]** — the CI contains 0.5, i.e. chance — on 105
positives, so this is not a small-sample artefact. Against the *report* label on held-out it
scores .70, so the head did learn something; that something just is not what radiologists
mean.

**The prevalence mismatch is 11×:**

| label | report label (blank→0), 171,996 rows | radiologist, 200 images | ratio |
|---|---|---|---|
| `enlarged_cardiomediastinum` | **4.77%** | **52.50%** | **11.0×** |
| `cardiomegaly` | 12.61% | 33.00% | 2.6× |
| `support_devices` | 56.74% | 49.50% | 0.87× |

A head trained to fire on 4.8% of images cannot rank a concept present in 52.5% of them. The
CheXpert labeller writes `Enlarged Cardiomediastinum` only when a report explicitly says the
mediastinum is widened; the radiologists re-reading the images flag any widened silhouette.
Two different variables sharing a column name.

**And the image evidence is there**: CheXzero reads it **.9160 [.876, .950]** zero-shot, its
best score of all seven labels tested. So `enlarged_cardiomediastinum` is *more* legible than
`cardiomegaly` (.8544) to a reader that never saw a CheXpert label. Our classifier's .55 is a
label-definition failure, not an image-content one.

## One structural finding, flagged not acted on

On `radiologist_val`, **cardiomegaly is a strict subset of enlarged cardiomediastinum**:

```
EC=1 & CM=1 : 66      EC=1 & CM=0 : 39      EC=0 & CM=1 : 0      phi = +0.668
```

Every one of the 66 cardiomegaly positives is also an EC positive, with 39 more besides. The
radiologists treated EC as the permissive reading of the same silhouette judgement, and
cardiomegaly as the strict one requiring a cardiothoracic-ratio call. Both independent readers
score the permissive version higher than the strict one — CheXzero .916 vs .854, XRV .829 vs
.806.

That is relevant to why v2 dropped cardiomegaly (.808, TPR ceiling .515): the difficulty may
be the **threshold call on a graded feature**, not the feature's visibility. It bears on
whether a cardiomegaly-like attribute could re-enter the graph under a different label
definition. No change made — this is measurement only.

## Artifacts

`outputs/chexpert_readers/` — `readers_table.json`, `readers_report.md`,
`chexzero_radval.json`, `chexzero_radval_probs.npy`, `head11_radval_ci.json`,
`support_devices_errors.json`; and `outputs/chexpert_noise/xrv_ci_cardio_ec.json`.
The 11-head checkpoint is at `outputs/attr_predictors_chexpert_head11/best.ckpt`, separate
from the frozen one.
