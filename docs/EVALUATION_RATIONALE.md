# Evaluation protocol — rationale and measurements

Why `EVALUATION_PROTOCOL.md` says what it says. Every number here was measured, not assumed.

---

## 1. Why CC was redefined

The old definition `CC = 1 - error / trivial_error`, with `trivial_error` the error of the best
constant predictor, is a skill score — right for a regression target where "predict the mean" is
a real competitor, wrong for a flip task. On a flip task the targets are the mirror of the
factual labels, so the baseline is **label prevalence in the cohort you drew**.

Identical model behaviour (hit = .50, inaction = 0, ceiling = 1):

| cohort | CC old | CC new |
|---|---|---|
| balanced 8/8 | +0.000 | +0.500 |
| imbalanced 11/5 | −0.600 | +0.500 |
| very imbalanced 15/1 | **−7.000** | +0.500 |

Measured on the archived CheXpert run: an intervention producing **9 correct flips of 16 with
zero wrong directions** scored **CC = −0.400**, because 11 of 16 sources carried the majority
label. CelebA's mean base rate is .2308 and its minimum .0042, so this applies to most of its
attributes.

There was also an internal inconsistency: FC already normalised to a measured floor
(`normalise_to_floor`), CC did not. The two halves of CF1 were anchored differently.

### Behaviour of the new form

| scenario | hit | inaction | ceiling | CC old | CC new |
|---|---|---|---|---|---|
| perfect gen, perfect reader | 1.00 | .00 | 1.00 | +1.000 | +1.000 |
| perfect gen, reader ceiling .83 | .83 | .00 | .83 | +0.660 | **+1.000** |
| does nothing | .00 | .00 | 1.00 | −1.000 | **0.000** |
| does nothing, reader misreads 12% | .12 | .12 | .88 | −0.760 | **0.000** |
| worse than doing nothing | .05 | .12 | .88 | −0.900 | −0.092 |

Continuous, using real C3DI instrument errors as the floor:

| factor | floor | edit size | achieved | CC |
|---|---|---|---|---|
| pos_spl | .0099 | .577 | .0099 | +1.000 |
| pos_spl | .0099 | .577 | .289 | +0.508 |
| pos_spl | .0099 | .577 | .577 | 0.000 |
| pos_spl | .0099 | .577 | .800 | −0.393 |
| hue_obj | .0738 | .520 | .260 | +0.583 |

`hue_obj` halfway scores above `pos_spl` halfway because its instrument is weaker and that is
correctly discounted.

### Known limit: a weak instrument flatters the model

| reader ceiling | model hit | CC |
|---|---|---|
| .95 | .50 | +0.526 |
| .60 | .50 | +0.833 |
| .30 | .25 | +0.833 |

Handled upstream by the admission gate, not by the formula.

---

## 2. Why admission is measured on reconstructions

Real → reconstruction drop in balanced accuracy on CelebA: **−5.8 pp mean**, and far worse
individually.

| attribute | bal real | bal recon | Δ | F1 real | F1 recon |
|---|---|---|---|---|---|
| Rosy_Cheeks | .8911 | .5983 | **−.2929** | .648 | .240 |
| Wearing_Lipstick | .9387 | .7004 | **−.2383** | .951 | .375 |
| Big_Nose | .8234 | .6751 | −.1482 | .752 | .792 |
| 5_o_Clock_Shadow | .9296 | .8005 | −.1290 | .759 | .778 |

**12 of 38 readers fall below balanced .80 on reconstructions while passing on real images.**

### Why F1 and not accuracy

CelebA's `Blurry` reader: **balanced accuracy .879, F1 .033**, base rate 0.4%, precision .017,
recall 1.000. It calls almost everything blurry. **17 of 40 readers have F1 < .70** while only 5
fall below balanced .80.

`F1 >= 0.55` is the rule `reader_keep.json` already applied, which cut 8 of the 40.

### Current standing

| set | real (tuned) | recon (tuned) |
|---|---|---|
| CelebA keep (32) | bal .9018, F1 .7808 | bal .8369, F1 .6937 |
| CelebA cut (8) | bal .8521, F1 .5343 | bal .7707, F1 .4404 |

C3DI: all seven factors pass. Weakest is `hue_obj` (R² .9470, MAE 14.2% of sd); the rest are
R² .98–.9994.

---

## 3. Three different n's were in use

| set | n | what |
|---|---|---|
| test | 3,000 | partition 2, real images |
| selection | 2,400 | a slice of **partition 0 — the training partition** |
| reconstruction eval | 2,048 | images actually reconstructed and re-read |

`selection_metrics.json` is **not** a held-out measurement and reads ~1 pp optimistic. The
protocol uses all 3,000; the remaining **952 still need reconstructing** before the floor gate
can be enforced on the full test set.

---

## 4. Support: the two datasets behave oppositely

**CelebA** — 6 of 16 joint cells are empty in the test partition, all female + beard/bald.

| combination | whole dataset | test |
|---|---|---|
| Beard & Male | 5,662 | 555 |
| **Beard & Female** | **10** | **0** |
| Bald & Male | 711 | 65 |
| **Bald & Female** | **1** | **0** |

Threshold choice is load-bearing:

| attested threshold | interventions kept | excluded |
|---|---|---|
| ≥ 1 | 11,723 / 12,000 | 2.3% |
| **≥ 10** | **7,658 / 12,000** | **36.2%** |
| ≥ 50 | 7,296 / 12,000 | 39.2% |

≥ 1 would admit the single bald female and ten bearded females as support, contradicting the
2026-09-24 policy's own intent.

**C3DI** — essentially no out-of-support region. Every class spans the full [−1, 1] on every
conditioned dimension; reflection escapes a class's observed range for **0.032%** of values.

Same rule, opposite behaviour. A rendered dataset with full-support marginals does not exercise
it; a dataset of photographs does. **No off-distribution conclusion from C3DI alone transfers.**

Because support is checked *after* propagation, it is interventions on **child** attributes that
exit support — `do(Beard=1)` leaves `Male` fixed. Parent interventions propagate and stay in
support. That is why `do(Young)` loses nothing and `do(Beard)` loses 62.6%.

---

## 5. SCM drift — two were unloadable

| SCM | was | now |
|---|---|---|
| celebahq graph A | fine | consistent |
| celebahq graph B | **did not exist** despite being referenced by its own graph yaml | fitted 2026-11-09, consistent |
| c3di gaussian | **would not load** | consistent, hidden 32 |
| c3di spline | **would not load** | consistent, hidden 64 |

`causal3dident_scm.pt` was trained when `ContextEncoder`'s hidden width defaulted to **32**. The
default later became **64** and the width is not recorded in the checkpoint, so
`load_state_dict` failed on a shape mismatch. **The weights were never wrong — the loader
drifted away from them.** Neither C3DI checkpoint records its mechanism either. Both are now
inferred from the state_dict.

Why it matters quantitatively: without the SCM, `do(X)` is the flip with the graph ignored —
wrong for **2,249 of 3,000** CelebA graph-A interventions and **all 25,200** C3DI `do(class)`.

### Gaussian vs spline

Measured on all 25,200 held-out rows, reproducing each checkpoint's stored `test_nll`:

| mechanism | test NLL | per dim |
|---|---|---|
| gaussian | 5.8832 | 0.7354 |
| **spline** | **5.2910** | **0.6614** |

Not cosmetic — rebuilding the cohort with spline collapsed the exclusions:

| intervention | excluded, gaussian | excluded, spline |
|---|---|---|
| do(class) | 837 | **4** |
| do(pos_spl) | 339 | **0** |

The gaussian was propagating descendants to values the post-intervention class does not occupy.
The worse fit showed up directly as spurious out-of-support exclusions.

### Why the move threshold is 1e-4

A 1e-6 threshold reported `rot_obj` — a sink with no children — as moving `pos_obj` on 3 of
5,000 sources. The deviation is **identical under a null intervention** (2.9e-06 for `pos_obj`
under spline, 1.2e-07 for `pos_spl`, 3.0e-07 under gaussian), so it is abduct→propagate
round-trip noise in float32, not leakage.

---

## 6. Why auxiliary metrics stay out of CF1

CC asks only "did the reading move to target". It cannot distinguish a real edit from one that
exploits the reader's decision surface. Observed on a diffusion model at guidance 12: an image
degraded past recognition scored **CC +0.768** against one reader, while three readers reported
the same pixels as **9/11, 3/11 and 0/11**.

With a single reader per attribute — the stated operating assumption — CC alone cannot catch
this. The defences are FC (preservation collapsed 1.000 → 0.000 in that case), the guidance
sweep (CC rising while FC falls is the signature), and recorded visual inspection.

Mixing perceptual or distributional measures into CF1 would make the headline numbers
incomparable with everything measured before, which is why §7 of the protocol keeps them out.

---

## 7. Budgets

T = 100, one guidance value, CFG doubling the per-step cost of counterfactuals only.

| | sources | CFs | evals | at batch 16 |
|---|---|---|---|---|
| CelebA graph A model | 3,000 | 8,242 | 2.25M | ~7.8 h |
| CelebA graph B model | 3,000 | 6,000 | 1.80M | ~6.2 h |
| C3DI model | 5,000 | 19,974 | 5.00M | ~4.6 h |
| composition (608 sources, 10 cycles) | — | 12,160 | 2.43M | ~8.4 h |

Across the five archived CelebA models (3 × graph A + 2 × graph B): **10.3M evals ≈ 36 h**.

Levers: T = 50 halves everything; each extra guidance value adds the CF term again; dropping
per-source reconstruction saves 13% but costs `e_floor`, which the protocol requires.
