# Counterfactual Evaluation Protocol

**Status: draft, 2026-11-09.** Dataset-independent. Working sets are **CelebA-HQ (256px)** and
**Causal3DIdent (128px)**.

Supersedes the CC definition in `able/adapters/base.py` (`cc_from_error`). FC and CF1 carry over
from `able/shared/metrics.py`, with the FC floor made explicit (§3).

Four layers, not interchangeable:

1. **Instrument admission** — which predictors may score anything (§1)
2. **CC / FC** — per-intervention components (§2, §3)
3. **CF1** — combination (§4)
4. **Reporting contract** — what every cell must carry (§6)

---

## 0. Why the old CC was replaced

The previous definition was

```
CC = 1 − error / trivial_error ,   trivial_error = error of the best constant predictor
```

That is a skill score (the shape of R², Nash–Sutcliffe, Brier Skill Score). It is correct for a
regression target, where "predict the mean" is a real competing strategy. It is **wrong for a
flip task**: on a flip task the targets are the mirror of the factual labels, so `trivial_error`
is just **label prevalence in the cohort you happened to draw**. The same model on the same
images scores differently when the cohort is reshuffled.

Prevalence stress — identical model behaviour throughout (hit = .50, inaction = 0, ceiling = 1):

| cohort | CC old | CC new |
|---|---|---|
| balanced 8/8 | +0.000 | +0.500 |
| imbalanced 11/5 | −0.600 | +0.500 |
| very imbalanced 15/1 | **−7.000** | +0.500 |

A constant label is not a counterfactual generator. Scoring image synthesis against a label
prior is a category error, and on an imbalanced attribute the constant predictor cannot be
beaten by construction however good the generator is.

*(Empirically: on the archived CheXpert run, an intervention producing 9 correct flips of 16
with zero wrong directions scored CC = −0.400, because 11 of 16 sources carried the majority
label. That run is archived, but the arithmetic is general and applies to every binary CelebA
attribute with a skewed base rate — which is most of them: mean base rate .2308, minimum .0042.)*

There was also an internal inconsistency: FC already normalised to a **measured floor**
(`normalise_to_floor`), CC did not. The two halves of CF1 were anchored to different references.

---

## 1. Instrument admission gate

**A predictor may not score a counterfactual unless it is admitted.** §2 credits a model against
the instrument's ceiling, and crediting against a blind instrument is how the metric is gamed
(§5.2).

### 1.1 Thresholds

Measured on a held-out set never used for the predictor's training or threshold selection:

```
binary / categorical    balanced_accuracy ≥ 0.80
                        F1                ≥ 0.55
                        CI_lower_95       ≥ 0.75      (bootstrap, ≥ 1000 resamples)
continuous              R²                ≥ 0.90
                        MAE ≤ 0.15 · target_sd       (error small against the factor's spread)
```

`F1 ≥ 0.55` is not new — it is the rule already applied in `reader_keep.json`, which cut 8 of
the 40 CelebA readers on the partition-0 selection slice. These thresholds formalise the
existing practice rather than inventing one.

**Accuracy alone is forbidden as an admission criterion**, and so is balanced accuracy alone.
The CelebA `Blurry` reader has **balanced accuracy .879 but F1 .033** — base rate 0.4%,
precision .017, recall 1.000. It calls almost everything blurry, and balanced accuracy rewards
it. Across the 40 readers, **17 have F1 < .70** while only 5 fall below balanced .80.

### 1.2 Measurement surface

Floors and admission are measured **on reconstructions**, not on real images, because
reconstructions are the distribution the protocol actually judges.

This matters more than it sounds. On CelebA the real → reconstruction drop in balanced accuracy
is **−5.8 pp on average**, and far worse for individual attributes:

| attribute | bal (real) | bal (recon) | Δ | F1 (real) | F1 (recon) |
|---|---|---|---|---|---|
| Rosy_Cheeks | .8911 | .5983 | **−.2929** | .648 | .240 |
| Wearing_Lipstick | .9387 | .7004 | **−.2383** | .951 | .375 |
| Big_Nose | .8234 | .6751 | −.1482 | .752 | .792 |
| 5_o_Clock_Shadow | .9296 | .8005 | −.1290 | .759 | .778 |

Twelve of 38 readers fall below balanced .80 on reconstructions while passing on real images.
A reader qualified on real data can be unqualified on exactly the images it is asked to judge.

**Current standing of the working sets** (CelebA keep-set of 32 = 4 observed + 28 unobserved):

| set | real (tuned) | recon (tuned) |
|---|---|---|
| CelebA keep (32) | bal .9018, F1 .7808 | bal .8369, F1 .6937 |
| CelebA cut (8) | bal .8521, F1 .5343 | bal .7707, F1 .4404 |

| C3DI factor | kind | test |
|---|---|---|
| class | categorical | acc .99996 |
| pos_spl | continuous | R² .9994, MAE .0099 (1.7% of sd) |
| pos_obj | continuous (3-dim) | R² .9990, MAE .0113 |
| rot_obj | continuous (3-dim) | R² .9939, MAE .0186 |
| hue_bg | continuous | R² .9984, MAE .0089 |
| hue_spl | continuous | R² .9817, MAE .0420 |
| **hue_obj** | continuous | **R² .9470**, MAE .0738 (14.2% of sd) |

All seven C3DI factors pass. `hue_obj` is the weakest and the only one near the MAE bound.

C3DI reconstruction floors have **not yet been measured** — that is the outstanding gap before
§1.2 can be enforced on that dataset.

### 1.3 Independence

Each admitted instrument is labelled **independent** or **entangled** for the model under test.
A predictor trained on the same labels the generator conditioned on is entangled, and its
agreement with that generator is weaker evidence. This is recorded, not gated.

---

## 2. CC — did the edit land

One definition for every attribute kind, in **error space** (lower is better):

```
            e_achieved − e_floor
CC = 1 −  ─────────────────────────
            e_inaction − e_floor
```

| term | meaning | binary / categorical | continuous |
|---|---|---|---|
| `e_achieved` | error of the counterfactual reading vs the post-intervention target | `1 − hit_rate` | `mean dist(pred_cf, target)` |
| `e_floor` | the instrument's own error on an **unedited reconstruction** | `1 − readback_accuracy` | `mean dist(pred_recon, factual)` |
| `e_inaction` | error from not editing at all | `1 − freebie_rate` | `mean dist(factual, target)` — **the size of the requested edit** |

`dist` is the adapter's existing distance:

```
binary / categorical : 1[a ≠ b]
continuous           : |a − b| ,  angular-wrapped where cyclic:
                       d ← min(d, span − d)        (names matching rot*, hue*)
                       multi-dim factors averaged over dims
```

### 2.1 Interpretation

- **CC = 1** — the model did everything the instrument can detect. Not "perfect image";
  "indistinguishable from perfect at this instrument's resolution".
- **CC = 0** — no better than leaving the image alone.
- **CC < 0** — the edit made the attribute *worse than not editing*. A real failure, rare,
  reported rather than clipped.
- **CC > 1** — read-back better than the ceiling. Clip to 1 **and raise a flag**: the ceiling
  was mis-measured, or the cohort differs from the one it came from.
- **`e_inaction ≤ e_floor`** — the requested edit is no larger than the instrument's noise.
  CC is **undefined and must raise**. There is no achievable signal and therefore no honest
  number. This is a computability condition, not a tuning parameter: cohorts are constructed so
  the requested edit is comfortably above the floor.

### 2.2 Equivalence to the hit-rate form

For categorical attributes this is algebraically identical to

```
CC = (hit − inaction) / (ceiling − inaction)
```

(verified to 1e-9). Either may be implemented; they are the same number.

### 2.3 Behaviour

| scenario | hit | inaction | ceiling | CC old | **CC new** |
|---|---|---|---|---|---|
| perfect gen, perfect reader | 1.00 | .00 | 1.00 | +1.000 | **+1.000** |
| perfect gen, reader ceiling .83 | .83 | .00 | .83 | +0.660 | **+1.000** |
| does nothing | .00 | .00 | 1.00 | −1.000 | **0.000** |
| does nothing, reader misreads 12% | .12 | .12 | .88 | −0.760 | **0.000** |
| worse than doing nothing | .05 | .12 | .88 | −0.900 | **−0.092** |

Continuous, using real C3DI instrument errors as the floor:

| factor | floor | edit size | achieved | CC |
|---|---|---|---|---|
| pos_spl | .0099 | .577 | .0099 | **+1.000** |
| pos_spl | .0099 | .577 | .289 | +0.508 |
| pos_spl | .0099 | .577 | .577 | **0.000** |
| pos_spl | .0099 | .577 | .800 | −0.393 |
| hue_obj | .0738 | .520 | .260 | +0.583 |

`hue_obj` "halfway" scores above `pos_spl` "halfway" because its instrument is weaker and that
is correctly discounted. The model is not penalised for the reader.

---

## 3. FC — did everything else survive

```
FC_reported = measured_preservation / floor_preservation
```

`floor_preservation` is measured under a **null intervention** — `do(A = its current value)` —
because inversion, guidance and reader noise all cost something before any edit is requested.
The difference form `(measured − floor)/(1 − floor)` was tried and discarded: it goes negative
whenever a model lands below the floor, which happens.

**Descendants are excluded from FC, never counted as failures.** `descendants(X)` walks the
graph edges; a descendant that did *not* move is the bug. They are scored separately against
the SCM's propagated target. For C3DI the edges are `class → rot_obj`, `class → pos_obj`,
`pos_spl → pos_obj`; for CelebA graph A, `Male/Young → Beard/Bald`.

**The FC set is always named in output** (`fc_obs`, `fc_unobs`). Two datasets once silently
diverged here — both reporting `hm(CC, FC)` while feeding different FC sets — and nothing in any
single table revealed it. C3DI's three hue factors are a ready-made `fc_unobs` set: conditioned
on by nothing, so anything that moves there is collateral damage by construction.

For binary attributes with low prevalence, preservation is reported as a plain **agreement
rate**, not normalised against a constant predictor: on a .42%-prevalence attribute such as
`Blurry`, constant-predictor normalisation makes a reader that always answers "no" look perfect.

---

## 4. CF1 — combination

```
hm(a, b) = 2ab / (a + b) ,   a, b clipped to [0, 1] first
CF1      = mean over interventions of hm(CC, FC)
```

**CF1 is the mean of per-intervention harmonic means, not the harmonic mean of pooled rates.**
They differ by Jensen's inequality (measured: 0.7717 vs 0.7756 on one real cell), and the pooled
form lets a frequent intervention dominate a rare one.

The clip before `hm` is load-bearing: CC goes negative on genuine failures, and a negative input
makes a harmonic mean meaningless rather than merely bad.

No single number is promoted to "the" CF1. Each FC set gets its own, named.

---

## 5. Known limits

### 5.1 A weak instrument flatters the model

The dual of the old problem: ceiling-normalisation is generous when little is achievable.

| reader ceiling | model hit | CC |
|---|---|---|
| .95 | .50 | +0.526 |
| .60 | .50 | **+0.833** |
| .30 | .25 | **+0.833** |

Handled **upstream by §1**, not by the formula. Ceiling-normalisation is only meaningful once
the instrument is known to see most of the signal. An un-admitted predictor must not produce
a CC.

### 5.2 Single-reader adversarial edits are not detectable by CC

CC asks only "did the reading move to target". It cannot distinguish a real edit from one that
exploits the reader's decision surface. Observed on a diffusion model at high classifier-free
guidance: an image degraded past recognition scored **CC +0.768** against one reader.

**This protocol assumes one admitted instrument per attribute.** The consequence is explicit:
with a single reader, CC alone cannot catch this failure mode. The defences are

1. **FC** — an adversarial edit that wrecks the image collapses preservation, so
   `hm(CC, FC) → 0`. In the observed case FC on the control attribute went 1.000 → 0.000.
2. **The guidance sweep** — CC rising while FC falls across increasing guidance is the
   signature. Report CC and FC *as a function of guidance*, never at a single operating point.
3. **Visual inspection of the reported operating point**, recorded as a figure.

FC does not catch an adversarial edit that happens to preserve everything else. That residual
risk is accepted and documented rather than mitigated.

---

## 6. The common ruler, and the reporting contract

CC is dimensionless, anchored at 0 = inaction and 1 = instrument ceiling, for **both** attribute
kinds — units cancel (error / error). This is what makes averaging across attributes, and across
datasets at different resolutions (CelebA 256, C3DI 128), defensible. The previous definition
averaged a mode-rate baseline against a mean-absolute-deviation baseline, which were never
comparable.

Two conditions preserve it:

1. **Floors measured identically** — all on reconstructions, same cohort, same thresholds
   (§1.2). A floor from real images for one attribute and reconstructions for another bends the
   ruler back.
2. **Edit magnitude is not captured.** "Move hue by 0.1" and "move hue by 3.0" both score 1.0 if
   achieved. CC asks *did you do what was asked*, so a cohort requesting only small edits scores
   high. **Report the edit-size distribution beside CC.** For binaries, report CC **split by
   direction**: a diffusion prior makes the move toward the majority class far easier, and
   pooling hides it entirely. (Measured on a skewed binary: 60% success removing the attribute
   against 18% creating it, on a label that was 74% negative in training.)

Every reported cell carries:

```
attribute, kind, intervention, guidance
CC, FC (per named FC set), CF1
e_achieved, e_floor, e_inaction          # so CC is recomputable
hit / real-flip / freebie / wrong-move   # categorical only
CC split by direction                    # binary only
edit-size distribution                   # continuous only
instrument used, independent|entangled, admitted-with (bal / F1 / R², CI)
n, and bootstrap CI on CC
```

A cell missing `e_floor` or the instrument record is not reportable.
