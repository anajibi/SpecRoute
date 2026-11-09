# Counterfactual Evaluation Protocol

**Status: draft, 2026-11-09.** Dataset-independent. Working sets are **CelebA-HQ (256px)** and
**Causal3DIdent (128px)**.

Supersedes the CC definition in `able/adapters/base.py` (`cc_from_error`). FC and CF1 carry over
from `able/shared/metrics.py`, with the FC floor made explicit (§3).

Five layers, not interchangeable:

0. **Test set, cohort and intervention semantics** — what is scored at all (§7)
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

---

## 7. Test set, cohort and intervention semantics

### 7.1 Test set

**CelebA-HQ: all 3,000 images of partition 2.** The dataset is 30,000, partitioned
24,000 / 3,000 / 3,000 (train / val / test) — confirmed from `celebahq_256_attrs.npz`.

No sub-sampling. Earlier evaluations used three different n's and they are not
interchangeable:

| set | n | what it was |
|---|---|---|
| test | 3,000 | partition 2, real images |
| selection | 2,400 | a slice of **partition 0** — the *training* partition — used to tune thresholds and apply the F1 keep rule |
| reconstruction eval | 2,048 | the images actually reconstructed and re-read |

`selection_metrics.json` is therefore **not a held-out measurement** and must not be quoted as
one; it reads about 1 pp optimistic against the test set.

**Outstanding work:** §1.2 requires floors measured on reconstructions, and only 2,048 of the
3,000 have been reconstructed. The remaining 952 must be reconstructed before the gate can be
enforced on the full test set.

### 7.2 Intervention semantics

A counterfactual is a **flip**. For each source image and each conditioned attribute, the
attribute is flipped and the SCM is used to abduct, act and predict, so every descendant takes
its post-intervention value and the model is conditioned on the whole post-intervention vector.

| kind | flip |
|---|---|
| binary | `v → 1 − v` |
| categorical (K > 2) | `v → (v + 1 + U{0, K−2}) mod K`, seeded |
| continuous, bounded symmetric | **`v → −v`** — reflection through the range centre |
| continuous, angular | **`v → v + π`** (wrapped) |

Reflection is the continuous analogue of a binary flip: deterministic, maximal, symmetric and
reproducible, and it puts the edit size comfortably above the instrument floor by construction
(§2.1). C3DI's `pos_spl`, `pos_obj` and `rot_obj` are all bounded to [−1, 1]; `rot_obj` and the
hue factors are angular and take the `+π` form.

### 7.3 Support restriction — no out-of-distribution track

**Interventions are restricted to attested combinations.** (Decision of 2026-09-24, retained.)
An intervention is requested only if the resulting joint cell of the conditioned attributes is
attested in the data:

```
count(post-intervention cell, full dataset) ≥ 10
```

The threshold applies to the **full joint cell**, not to a pairwise margin. Measured support in
CelebA-HQ graph A (Male, Young → Beard, Bald):

| combination | whole dataset | test partition |
|---|---|---|
| Beard & Male | 5,662 | 555 |
| **Beard & Female** | **10** | **0** |
| Bald & Male | 711 | 65 |
| **Bald & Female** | **1** | **0** |
| Bald & Young | 138 | 11 |

**Six of the sixteen joint cells are empty in the test partition, and every one is a female
with beard or baldness.** At the ≥ 10 rule:

| intervention | kept | excluded |
|---|---|---|
| do(Male) | 2,416 / 3,000 | 19.5% |
| do(Young) | 3,000 / 3,000 | 0% |
| do(Beard) | 1,121 / 3,000 | 62.6% |
| do(Bald) | 1,121 / 3,000 | 62.6% |
| **total** | **7,658 / 12,000** | **36.2%** |

Thresholds of ≥ 1 and ≥ 50 exclude 2.3% and 39.2% respectively; ≥ 1 admits the single bald
female and the ten bearded females as "support", which contradicts the policy's own stated
intent.

**There is no OOD track.** Off-distribution generalisation — arguably the point of a causal
model — therefore goes unmeasured by this protocol, and that is a deliberate, recorded cost.
A consequence worth stating: because support is checked *after* propagation, an intervention on
a **child** attribute is what exits support (`do(Beard=1)` leaves `Male` fixed), while an
intervention on a **parent** propagates and stays in support. The exclusions above are almost
entirely child interventions.

### 7.4 Cohort

**Every admitted intervention on every test image.** No stratified sub-sampling: 3,000 sources
× 4 conditioned attributes, minus the support exclusions, = **7,658 counterfactuals per
guidance value** for CelebA graph A, plus 3,000 reconstructions and 3,000 DDIM inversions.

This removes cohort construction as a source of disagreement entirely, and makes the prevalence
of each cell in the score equal to its prevalence in the test set.

### 7.5 Cohort artifacts

A cohort is a **file**, not a convention, so an evaluator never infers anything. Per dataset and
attribute it carries: the source row, the factual vector, the **post-intervention vector after
SCM propagation**, which attributes moved, the support count or range check, and — where the
model's coding differs from the logical one — both codings.

**CelebA-HQ** (`cohorts/celebahq/graph{A,B}_{attr}.json`), 3,000 test images:

| graph | intervention | admitted | propagated | excluded (out of support) |
|---|---|---|---|---|
| A | do(Male) | 3,000 | **2,249** | 0 |
| A | do(Young) | 3,000 | 54 | 0 |
| A | do(Beard) | 1,121 | 0 | **1,879** |
| A | do(Bald) | 1,121 | 0 | **1,879** |
| B | do(Eyeglasses) | 3,000 | 0 | 0 |
| B | do(Smiling) | 3,000 | 0 | 0 |

**14,242 admitted interventions.** Graph B has no edges, so there the flip *is* the
counterfactual — same code, different `moved` sets.

**Causal3DIdent** (`cohorts/causal3dident/do_{attr}.npz` + `cohort_summary.json`), 25,200 test
rows across 7 classes. Written as npz because 100,800 interventions as JSON is ~40 MB of
repeated floats.

| intervention | admitted | moves | excluded |
|---|---|---|---|
| do(class) | 21,016 / 25,200 | class, pos_obj, rot_obj | 4,184 |
| do(pos_spl) | 23,456 / 25,200 | pos_spl, pos_obj | 1,744 |
| do(pos_obj) | 25,177 / 25,200 | pos_obj (sink) | 23 |
| do(rot_obj) | 25,132 / 25,200 | rot_obj (sink) | 68 |

**94,781 admitted interventions.** The propagation is visible in the `moves` column and matches
the graph exactly: `class` is a parent of both `pos_obj` and `rot_obj`; `pos_spl` is a parent of
`pos_obj` only; the two sinks move nothing else.

**C3DI has essentially no out-of-support region.** Every class spans the full [−1, 1] on every
conditioned dimension, so reflection lands outside a class's observed range for only 0.032% of
values. The exclusions above come almost entirely from `do(class)`, where the SCM moves
`pos_obj`/`rot_obj` into a value the *new* class does not occupy. Contrast CelebA, where 6 of 16
joint cells are empty and 36.2% of interventions are dropped. **The support rule is the same;
the datasets differ.** A rendered dataset with full-support marginals does not exercise it; a
dataset of photographs does. Any conclusion about off-distribution behaviour drawn from C3DI
alone would not transfer.

C3DI's three rendered-but-unconditioned factors — `hue_obj`, `hue_spl`, `hue_bg` — are the
`fc_unobs` pool and are carried in the cohort so FC is scorable without re-reading the data.

---

## 8. SCM availability

**For every model under evaluation, the SCM it was trained against must be loadable and must
perform abduction.** Without it `do(X)` cannot be propagated to descendants, and what is scored
is not the counterfactual — it is the flip with the graph ignored. On CelebA graph A that
difference is 2,249 of 3,000 interventions; on C3DI `do(class)` it is all 25,200.

Each SCM is verified before use by the property that actually matters:

```
null intervention must reproduce the factual values:  counterfactual(x, {}) == x
```

A loader that returns a model with silently wrong weights otherwise looks like success.

Current state (`able/shared/scm_registry.py`, run it to re-check):

| SCM | status | loaded with |
|---|---|---|
| celebahq graph A | loaded, **consistent** | attrs Male, Young, Beard, Bald |
| celebahq graph B | loaded, **consistent** | attrs Eyeglasses, Smiling |
| c3di gaussian | loaded, **consistent** | hidden 32, gaussian |
| c3di spline | loaded, **consistent** | hidden 64, spline |

Two of these could not be loaded at all before this protocol was written, and the failures are
the kind that stay hidden until someone tries:

- `causal3dident_scm.pt` was trained when `ContextEncoder`'s hidden width defaulted to **32**.
  The default later became **64**, and the width is not recorded in the checkpoint, so
  `load_state_dict` failed on a shape mismatch. The weights were never wrong — **the loader
  drifted away from them.**
- Neither C3DI checkpoint records its mechanism (`gaussian` vs `spline`).
- `celebahq_b_scm.pt` did not exist despite being referenced by
  `causal_graph_celebahq_b.yaml`. Fitted 2026-11-09 (no edges, so two marginals).

`scm_registry.py` infers hidden width and mechanism from the state_dict rather than guessing, so
a checkpoint saved under any past default still loads. **Requirement going forward: an SCM
checkpoint records its own architecture**, and the cohort header records which SCM produced it
(`cohort_summary.json: scm.loaded_with`). An evaluation run must use the same SCM the cohort was
built with; a mismatch invalidates the targets.
