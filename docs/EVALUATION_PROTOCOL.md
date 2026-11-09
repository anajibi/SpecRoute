# Counterfactual Evaluation Protocol

Dataset-independent. Working sets: **CelebA-HQ 256px**, **Causal3DIdent 128px**.
Rationale, measurements and failure cases: `EVALUATION_RATIONALE.md`.

Supersedes `cc_from_error` in `able/adapters/base.py`. FC and CF1 follow
`able/shared/metrics.py`.

---

## 1. Instrument admission

No predictor scores a counterfactual unless admitted. Measured on **reconstructions**, on a set
never used for the predictor's training or threshold selection.

```
binary / categorical    balanced_accuracy >= 0.80
                        F1                >= 0.55
                        CI_lower_95       >= 0.75     (bootstrap, >= 1000 resamples)
continuous              R^2               >= 0.90
                        MAE               <= 0.15 * target_sd
```

Accuracy alone is not admissible evidence. Neither is balanced accuracy alone.

Each instrument is labelled **independent** or **entangled** for the model under test, and that
label is reported.

---

## 2. CC — did the edit land

One definition, both attribute kinds, in error space:

```
            e_achieved - e_floor
CC = 1 -  -------------------------
            e_inaction - e_floor
```

| term | binary / categorical | continuous |
|---|---|---|
| `e_achieved` | `1 - hit_rate` | `mean dist(pred_cf, target)` |
| `e_floor` | `1 - readback_accuracy` on an unedited reconstruction | `mean dist(pred_recon, factual)` |
| `e_inaction` | `1 - freebie_rate` | `mean dist(factual, target)` |

```
dist:  categorical  1[a != b]
       continuous   |a - b|, angular-wrapped where cyclic: d <- min(d, span - d)
                    multi-dim factors averaged over dims
```

Equivalent categorical form: `CC = (hit - inaction) / (ceiling - inaction)`.

| value | meaning |
|---|---|
| 1 | everything the instrument can detect |
| 0 | no better than leaving the image alone |
| < 0 | worse than not editing |
| > 1 | clip to 1 **and flag** — the ceiling was mis-measured |
| `e_inaction <= e_floor` | **undefined, must raise** |

---

## 3. FC — did everything else survive

```
FC_reported = measured_preservation / floor_preservation
```

`floor_preservation` is measured under a null intervention `do(A = its current value)`.

- Descendants are **excluded** from FC and scored separately against the SCM's propagated
  target. A descendant that did not move is the bug.
- The FC set is always named: `fc_obs`, `fc_unobs`.
- Low-prevalence binaries report a plain agreement rate, not a normalised one.

---

## 4. CF1

```
hm(a, b) = 2ab / (a + b),  a and b clipped to [0, 1] first
CF1      = mean over interventions of hm(CC, FC)
```

Mean of per-intervention harmonic means, **not** the harmonic mean of pooled rates. One CF1 per
FC set, each named. No single number is promoted to "the" CF1.

---

## 5. Cohorts

A cohort is a file. It carries, per source and attribute: the source row, the factual vector,
the post-intervention vector **after SCM propagation**, which attributes moved, the support
check, and both value codings where they differ.

### 5.1 Interventions

A counterfactual is a flip, then abduct → act → predict through the SCM.

| kind | flip |
|---|---|
| binary | `v -> 1 - v` |
| categorical (K > 2) | `v -> (v + 1 + U{0, K-2}) mod K`, seeded |
| continuous, bounded symmetric | `v -> -v` |
| continuous, angular | `v -> v + pi` |

An attribute counts as **moved** above `1e-4` (the SCM's abduct→propagate round-trip floor).

### 5.2 Support

Requested only if the post-intervention state is attested:

```
CelebA   count(post-intervention joint cell, full dataset) >= 10
C3DI     post-intervention value inside the class-conditional [min, max] of all rendered rows
```

Checked **after** propagation. There is **no out-of-distribution track**.

### 5.3 Main cohorts

**CelebA-HQ** — all 3,000 images of partition 2. `cohorts/celebahq/graph{A,B}_{attr}.json`.

| graph | intervention | admitted | propagated | excluded |
|---|---|---|---|---|
| A | do(Male) | 3,000 | 2,249 | 0 |
| A | do(Young) | 3,000 | 54 | 0 |
| A | do(Beard) | 1,121 | 0 | 1,879 |
| A | do(Bald) | 1,121 | 0 | 1,879 |
| B | do(Eyeglasses) | 3,000 | 0 | 0 |
| B | do(Smiling) | 3,000 | 0 | 0 |

**14,242 admitted.**

**Causal3DIdent** — stratified 5,000 of 25,200 test rows, balanced over 7 classes
(715/715/714×5). `cohorts/causal3dident/do_{attr}.npz`. Built with the **spline** SCM.
`row` indexes the full 25,200-row test set. Support referenced against all 277,200 rendered rows.

| intervention | admitted | moves | excluded |
|---|---|---|---|
| do(class) | 4,996 | class, pos_obj, rot_obj | 4 |
| do(pos_spl) | 5,000 | pos_spl, pos_obj | 0 |
| do(pos_obj) | 4,991 | pos_obj (sink) | 9 |
| do(rot_obj) | 4,987 | rot_obj (sink) | 13 |

**19,974 admitted.** `hue_obj`, `hue_spl`, `hue_bg` are the `fc_unobs` pool.

### 5.4 Composition cohort

Separate, never averaged into the main cohort. Stratified over the factual joint cell (CelebA)
or class (C3DI). Only main-cohort-admitted interventions are eligible.

608 sources across 10 blocks (56–64 each), 12,160 generations at 10 cycles. Stores two
endpoints; cycle *k* applies `target` on odd *k*, `factual` on even *k*.

---

## 6. SCM availability

For every model under evaluation, its trained SCM must load and perform abduction. Verified by:

```
counterfactual(x, {}) == x      # a null intervention reproduces the factual values
```

Where several SCMs are fitted for a dataset, **the highest held-out fit is canonical**. C3DI:
spline (test NLL 5.2910) over gaussian (5.8832).

An SCM checkpoint must record its own architecture. The cohort header records which SCM built
it; a mismatch at evaluation time invalidates the targets.

Loader and verification: `able/shared/scm_registry.py`.

---

## 7. Auxiliary metrics

**None of these enter CC, FC or CF1.** Reported beside them, never combined.

| § | metric | computed against |
|---|---|---|
| 7.1 | LPIPS, L1/PSNR, SSIM | the source's own **reconstruction** |
| 7.2 | FID | the real test distribution, at fixed and reported n |
| 7.3 | composition drift at cycles 1 and 10 | the source's reconstruction after an even cycle count |

§7.1 values are expected non-zero: an edit that changes nothing scores perfectly here and 0 on
CC. Read them against the null-intervention floor.

Identity preservation is deliberately not measured.

---

## 8. Reporting contract

`CC`, `fc_obs`, `fc_unobs`, `cf1` are reported **per attribute, unaggregated**, in addition to
any mean.

Every cell carries:

```
attribute, kind, intervention, guidance
CC, fc_obs, fc_unobs, cf1                # raw, per attribute
e_achieved, e_floor, e_inaction          # so CC is recomputable
hit / real-flip / freebie / wrong-move   # categorical
CC split by direction                    # binary
edit-size distribution                   # continuous
instrument: id, independent|entangled, admitted-with (bal / F1 / R^2, CI)
auxiliary: LPIPS, L1, PSNR, SSIM, FID(n), composition drift
n, bootstrap CI on CC
```

A cell missing `e_floor` or the instrument record is not reportable.

CC and FC are reported **as a function of guidance**, never at a single operating point.
