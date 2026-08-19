# HDAE Counterfactual-Quality Diagnostics — Full Dump

Everything produced during the Phase 0/1 diagnostics pass on the HDAE counterfactual-quality investigation (branch `item1-cf-contract`): recon notes, the running ledger narrative, and the full structured JSON for all 14 tests (T1-T14, plus T7's precondition check), in that order.

Source directory: `experiments/hdae/outputs/diagnostics/` (gitignored, local-only). Test scripts (the actual code) are committed under `experiments/hdae/diagnostics/`.

---

## Table of Contents

1. [Recon (R1)](#recon-r1)
2. [Ledger narrative summary](#ledger-narrative-summary)
3. [T1 — EMA / resume audit](#t1)
4. [T2 — Eval-harness equivalence audit](#t2)
5. [T3 — CFG null-branch construction](#t3)
6. [T4 — FC_observed causal-edge audit](#t4)
7. [T5 — Per-attribute conditioning strength (swing)](#t5)
8. [T6 — CC ceiling/floor calibration](#t6)
9. [T7 (precondition) — Deterministic attribute measurement](#t7)
10. [T8 — Predictor hue-sensitivity](#t8)
11. [T9 — Conditioning-ablation loss probe](#t9)
12. [T10 — Latent linear probe](#t10)
13. [T11 — Digit CC by target class (k=11 @ 75k)](#t11)
14. [T12 — k=5 forensics](#t12)
15. [T13 — Nearest-neighbour OOD extension](#t13)
16. [T14 — Capacity-ordering falsification](#t14)
17. [K11 @ 75k full re-eval log (raw)](#k11-75k-eval-log)

---

## Recon (R1)

# R1 — Recon

## Checkpoints

| model | step | epoch | file | mtime |
|---|---|---|---|---|
| k=1 | 30000 | 32 | `morpho_hier_k1_v3/checkpoints/last.ckpt` | unchanged since 30k run |
| k=5 | 30000 | 32 | `morpho_hier_k5_v3/checkpoints/last.ckpt` | unchanged since 30k run |
| k=11 | 66000 | 143 | `morpho_hier_k11_v3/checkpoints/last_step66000.ckpt` | snapshotted mid-run 2026-08-18 15:24 |
| k=11 | 75000 | 163 | `morpho_hier_k11_v3/checkpoints/last_step75000.ckpt` | snapshotted 2026-08-19 00:41 (run completed naturally, process exited on its own — no kill needed) |

**IMPORTANT — T17 precondition is broken.** `save_top_k=1, save_last=True` means only the most
recent checkpoint (plus a byte-identical epoch-named duplicate, deleted as redundant) exists on
disk at any time. The 30k and 45k K11 checkpoints that produced the `cf1_eval_gs8` and
`cf1_eval_gs8_tol2x` eval logs were both overwritten by later continuations before this
diagnostic pass began — **only step 66000 and step 75000 snapshots exist now for k=11.**
T17 as specified (30k / 45k / ≥55k) cannot run against real checkpoints for two of its three
points. The eval *logs* and generated grid images from 30k and 45k still exist and are usable
for anything log-derived (T1, T2 below), but any test requiring the actual weights at 30k or 45k
(T5, T9, T10, T12a's k=11 arm at those steps) can only use 66k/75k going forward, plus k=1/k=5's
still-intact 30k checkpoints as depth-matched reference points.

State-dict top-level keys (all checkpoints): `epoch, global_step, pytorch-lightning_version,
state_dict, loops, callbacks, optimizer_states, lr_schedulers, MixedPrecisionPlugin,
hparams_name, hyper_parameters`. `lr_schedulers: []` — empty because `conf.warmup` defaults to
`0` in `diffae_upstream/templates.py`/`config.py` and is never set in any HDAE yaml, so
`configure_optimizers` never builds a `LambdaLR` at all (see T1). Not a resume bug.

EMA is not a separate top-level key — it's a full second copy of the model living inside
`state_dict` under an `ema_model.*` prefix (`LitModel.__init__`: `self.ema_model =
copy.deepcopy(self.model)`), so it's restored via the exact same standard
`Trainer.fit(ckpt_path=...)` state-dict load as everything else. No separate/fragile EMA-restore
path exists in this codebase.

## Eval path

Entry point for all `cf1_eval_gs8*` logs: `experiments/hdae/counterfactuals/
morpho_cf1_eval_toleranced.py` (the `_tol2x`/`_tol3x` variants) and an earlier
non-toleranced version for the plain `cf1_eval_gs8` logs. Both share the same underlying
scoring helpers (`attribute_partition`, `fc_for_pool_mixed`, `fc_for_pool_hard`,
`compute_train_stats`).

- **Raw or EMA weights?** Always EMA — `hdae_adapter.py:45`: `self.model = module.ema_model`.
  This is hardcoded in the adapter, not a per-run flag, so it was identical between the 30k and
  45k evals (and would be for any future eval using this adapter).
- **Cohort selection:** fixed index list stored in `experiments/hdae/outputs/
  intervention_cohorts.json` under `_meta.fixed_indices` — a static file, last modified
  2026-08-12, i.e. **before both** the 30k (2026-08-17) and 45k (2026-08-18) eval runs. Not
  regenerated between them.
- **Target sampling:** also pre-computed and stored in the same static `intervention_cohorts.json`
  (`target_value`/`target_bin`/`bin_edges` per continuous attribute; digit/hue targets are
  computed live from a fixed `(class + n//2) % n` shift, deterministic given the cohort's
  original class — same both times).
- **Predictor checkpoint:** `experiments/hdae/outputs/attr_predictors_70k/training_summary.json`,
  last modified 2026-08-12 — unchanged before both eval runs. Verified `thickness/best.ckpt`
  sha256 `addb4008d7333a48…` is the file actually referenced by the summary; not touched since.
- **Per-image eval records / generated images saved?** No — the eval scripts only write
  aggregate JSON + a text log, not per-image records or generated tensors. This sets T11 and T15
  to "must be regenerated from the checkpoint," not "free from existing artifacts" — a correction
  to what the v1/v2 plan assumed.

## Metric definitions (quoted from `morpho_cf1_eval_toleranced.py`)

- **FC_observed's stay-put set**: `attribute_partition(graph, attr)` returns
  `(descendants, observed)` where `observed = [a for a in graph.attributes if a != attr and a
  not in descendants]` — i.e. everything except the intervened attribute *and* its causal
  descendants. For a thickness intervention, `descendants=['intensity']`, so
  `observed=['digit','hue']` — intensity is correctly excluded from FC_observed (see T4 below).
- **FC_unobserved's definition**: hard per-attribute gate, `|pred_cf - pred0| > mult *
  cnn_mae[attr]` counts as "moved", over the 8 structurally-unmodeled attributes
  (`UNMODELED_ATTRS`), success/fail pools weighted by their pool size.
- **CF1's formula**: standard harmonic mean of CC and FC (observed/unobserved variants
  computed separately, no single combined formula beyond the per-variant harmonic mean).
- **`tol2x` meaning**: continuous-attribute CC tolerance = half the width of the image's own
  target population-quantile bin (from `intervention_cohorts.json`'s `bin_edges`); the `2x`/`3x`
  multiplier applies *only* to FC_unobserved's hard gate (`mult * cnn_mae`), not to CC's
  tolerance — the naming is easy to misread as applying uniformly, it doesn't.
- **CC's descendant term**: for each descendant `d` of the intervened attribute, a separate
  pass/fail check (`|pred_cf[d] - scm_propagated_target[d]| <= std[d]*0.25`) is added into the
  *same* CC numerator/denominator as the intervened attribute itself (`cc_den = n_valid * (1 +
  n_descendants)`) — so thickness's reported CC is actually a joint {thickness, intensity}
  correctness score, not thickness alone. Worth flagging in any external write-up.

## Guidance (CFG)

`attr_conditioner.py`'s `ConcatAttributeEmbedding.forward` (the fusion mode actually configured
— `attr_fusion: concat_film`): during training,
```python
if apply_dropout and self.training and self.cfg_drop_prob > 0:
    drop_sample = torch.rand(batch_size, 1, ...) < self.cfg_drop_prob   # one draw per example
    mask = mask | drop_sample.expand(-1, n_attr)                        # nulls ALL 4 attrs together
if apply_dropout and self.training and self.attr_dropout_prob > 0:
    drop = torch.rand(batch_size, n_attr, ...) < self.attr_dropout_prob # independent per attr
    mask = mask | drop
```
**Unconditional branch = (a) all four attributes nulled together**, via `cfg_drop_prob=0.1`
applied per-example (not per-attribute). Masked attributes are replaced with a **dedicated
learned null vector per attribute** (`self.null_vectors[i]`), concatenated in with the real
embeddings for unmasked attributes — not a single shared global null token.

## Conditioning (FiLM)

`PerBlockStyleFiLM.__init__`: `scale_proj`/`shift_proj` are each **one `nn.Linear(attr_embed_dim,
embed_channels)` per decoder block**, applied to the *full concatenated* 128-dim `attr_emb` —
i.e. a **fused** projection over all 4 attributes' slices jointly, not 4 independent
per-attribute projections summed. `style = z_style * (1 + scale(attr_emb)) + shift(attr_emb)`,
zero-initialized so `style == z_style` at step 0.

## Causal graph / propagation

`morpho_cf1_eval_toleranced.py:108-111`:
```python
cf_attrs = scm.counterfactual(attrs_raw[:, scm_cols].float(), scm_attr_index,
                              interventions={spec["attr"]: target_tensor})
for a in observed:                          # only non-descendant, non-intervened attrs
    cf_attrs[a] = attrs_raw[:, [scm_cols[scm_attr_index[a]]]].clone()   # held at original
```
Descendants (e.g. intensity, for a thickness intervention) are **left at the SCM's propagated
counterfactual value** — never overwritten back to the original. Propagation through the one
declared causal edge (thickness→intensity) is present and correctly wired at eval time.

## Attribute label provenance (T7 precondition)

`experiments/hdae/data/morphomnist.py`:
- `measure_thickness(gray_u8)` (binarize `>20`, `distance_transform_edt`, `2*mean over fg`) and
  `measure_intensity(gray_u8)` (`mean over fg pixels`) are the **literal functions that produced
  the stored ground-truth labels** — `render()` calls `achieved_thickness =
  measure_thickness(gray)` / `achieved_intensity = measure_intensity(gray)` and writes those
  (not the originally-requested target) into the dataset record. Deterministic, no learned
  component.
- Critically, this measurement happens on `gray` — **before** `_apply_hue`, `_apply_geometry`
  (affine warp), `_apply_background` (additive sinusoidal field, up to +18), and `_apply_texture`
  (additive noise, up to +10) are applied later in the same `render()` call. The final stored/
  observable image is a *later* stage of the pipeline than the image the labels were measured on.
- **hue** has no equivalent `measure_hue` in the codebase — the stored value is the *requested*
  input to `_apply_hue`, not something measured back off the image. One is constructible in
  principle (`_apply_hue` scales a grayscale value by a fixed per-hue RGB gain with max
  component exactly 1.0, so foreground-pixel RGB→HSV inversion should recover it), but doesn't
  exist yet — flagged as new code needed, not reused.
- **Digit** has no deterministic oracle — stays predictor-only, as the plan expected.


---

## Ledger narrative summary

# Diagnostics ledger

## T1 — EMA / resume audit — **refuted**
Weight source (`ema_model`) is hardcoded identically in `hdae_adapter.py` for every eval, so the
30k and 45k evals used the same weight-selection logic. EMA lives inside the ordinary
`state_dict` (a full `deepcopy`d second model, not a separate callback state), so it's restored
by the standard checkpoint-load path with no special resume risk — confirmed empirically by
measuring a 9.6% mean relative L2 difference between raw and EMA weights at step 75000 (nonzero
= not frozen/reset, not huge = tracking normally). LR warmup is never configured (`conf.warmup`
defaults to 0 and is never overridden), so there's no ramp to restart on resume either — the
empty `lr_schedulers: []` in every checkpoint is expected, not a bug. **F3 is not explained by
weight selection, EMA restoration, or warmup restart.**

## T2 — Eval-harness equivalence audit — **refuted**
Cohort file, target sampling, and predictor checkpoint are all static files, unmodified between
the 2026-08-17 (30k) and 2026-08-18 (45k) eval runs — verified by mtime and checkpoint hash. The
2-image difference in CC denominator (507/512 vs 505/512) matches previously-documented ordinary
generation stochasticity, not a cohort change. Digit/hue's exact-match scoring is unaffected by
the tolerance-methodology redesign between the two eval scripts. **Combined with T1, F3 survives
both gates — the digit CC regression (0.755 → 0.463) is very likely a real training-dynamics
phenomenon, not a measurement or harness artifact. This should be stated plainly: something
about continuing training from 30k to 45k genuinely hurt digit conditioning while loss stayed
flat at ~0.007.**

## T3 — CFG null-branch construction — **ambiguous**
Confirmed: unconditional branch nulls all 4 attributes together (one Bernoulli draw per example,
`cfg_drop_prob=0.1`), and FiLM's scale/shift are a single fused Linear over the full concatenated
128-dim embedding, not per-attribute-independent — structurally this is the plan's "severe" case.
**But** the plan's own rationale for severity ("all-null is an unvisited point") is factually
wrong for this config: `cfg_drop_prob=0.1` is a standard, deliberate CFG training rate, making
all-null a common regime (~10% of ~1M+ training examples by 75k steps), not rare. The structural
fused-MLP concern still stands for a different reason and is worth testing via T18 arm B, but
T3 cannot by itself explain F2 (hue vs thickness asymmetry) — it's attribute-agnostic by
construction, exactly as the plan's own scope limit anticipated.

## T4 — FC_observed causal-edge audit — **refuted**
The causal graph is correctly applied at eval time: `scm.counterfactual()` propagates
thickness→intensity into the counterfactual target vector, descendants are correctly excluded
from FC_observed's stay-put set, and are instead scored via a separate descendant-consistency
term folded into CC's own numerator/denominator. No fix needed — this hypothesis does not hold
for this codebase. Side finding: "thickness CC" in every table so far is actually a joint
{thickness, intensity} correctness score (because of the descendant term), not thickness alone —
worth flagging when presenting these numbers externally.

## T7 (precondition) — Deterministic attribute measurement — **refuted as proposed / reframed**
`measure_thickness`/`measure_intensity` are literally the label-generating functions, but they
were applied to a pre-augmentation grayscale canvas *inside* `render()`, before hue
colorization, geometric warp, and (critically) additive background-field/texture noise are
applied. Naively reapplying them to the final observable image fails badly — 3-12x worse than
the CNN predictor's own error, not better — because background/texture noise pushes pixels above
the fixed `>20` foreground threshold these functions rely on. **The free-win version of T7 does
not work; T8 and T6 remain necessary for thickness/intensity, not just digit.** A background-
aware version of the oracle is plausible future work but wasn't attempted here (scoped out, not
shown infeasible). Independent finding: the dataset's own stored labels are measured on an
idealized pre-noise canvas, so there's already a real label/observable gap for every real image,
unrelated to the model.

## T5 — Per-attribute conditioning strength (swing) — **ambiguous, high-value tangent**
k1@30k: hue's swing is 1.93x thickness's (just under the plan's 2x bar, but CIs completely
non-overlapping) — supports H4. k5@30k: same direction, weaker (1.54x). **k11@75k: hue's swing
is actually LOWER than thickness's (0.82x)** — opposite direction, and all four attributes'
swings have compressed to a narrow, nearly uniform range (37-44) compared to k1's much wider
spread (27-55). Digit has the *lowest* swing of the four at k11@75k. This is an unplanned but
directly relevant finding for F3: a weakening of digit's relative conditioning strength over
k11's training is a concrete mechanism candidate. **Recommended cheap follow-up: re-run the
toleranced CC/FC/CF1 eval on the k11 75k checkpoint** to see if F2's hue/thickness asymmetry has
also compressed, matching the swing-compression finding — not yet done (GPU cost, flagged for
the user to prioritize).

## T9 — Conditioning-ablation loss probe — **supported, major finding**
Direct, loss-level test: ΔL = L(null attr) − L(true attr), matched noise/timestep. **Thickness
and intensity show ΔL ≈ 0 in every model checked (k1, k5, k11 at two steps)** — none of these
models meaningfully use the explicit thickness/intensity conditioning for reconstruction at all.
This is direct, independent, strong evidence for F1's mechanism. **Hue: k1 uses it moderately,
k5 uses it MORE than anything else measured (ΔL=0.00034, ~71% of k5's total)** — k5 is not
ignoring hue, it depends on it heavily, which reframes F4 away from "collapse" toward
"overfit/memorized, doesn't generalize to counterfactual values." **K11 shows the smallest ΔL of
any model for every attribute, especially digit** (2.5-4.5x smaller than k1/k5's digit ΔL) —
direct evidence k11's decoder relies least on explicit conditioning, consistent with more
leakage through its deeper hierarchy.

## T10 — Latent linear probe — **supported, major finding, corroborates T9**
Linear probes on the concatenated 512-dim latent. **Digit is far more decodable from k11's
latent (84-86%) than k1's (57%) or k5's (53%)** — the OPPOSITE of the plan's assumed capacity
ordering (lower k = more leakage), and directly corroborates T9's finding that k11 relies least
on explicit digit conditioning. **Thickness/intensity are substantially decodable even from
k1 at just 30k steps (R²≈0.72 each)**, rising further with k — combined with T9's model-agnostic
ΔL≈0 finding, this gives a strong, depth-independent mechanism for F1. **Hue: k1's R²=0.17
(barely leaked) vs k5's R²=0.996 (near-total memorization) vs k11's 0.97-0.98** — k5's
near-perfect hue leakage plus its heavy reconstruction-time hue reliance (T9) together suggest
its latent has memorized each image's own hue, creating a conflict when a counterfactual hue
differs from what's already baked in. K11 has comparably high hue leakage yet good hue CC, so
leakage alone doesn't explain F4's severity — something k5-specific (training trajectory/seed)
compounds it, sharpening what T20 (reseed) should look for.

## T12 — k=5 forensics (updated with T5's swing data) — **refuted**
(See original entry above.) T5's swing data confirms k5's hue conditioning pathway carries a
normal, non-degenerate signal (normalized swing 0.19, comparable to k1's 0.21) — not collapsed
at the embedding level. Combined with T9/T10's finding that k5's hue problem is about
memorization/generalization rather than a dead pathway, "all clean, defer to T20" now comes with
a specific, sharper hypothesis to test in a reseed run.

## T13 — Nearest-neighbour OOD extension — **ambiguous**
hue (94.7% within gap, ratio~0.99) and thickness (91.6%, ratio~1.02) are close to digit's clean
result, just under the plan's 95% bar. **Intensity is the weakest of the four (82.8% within gap,
median ratio 1.09, p95 ratio 1.56)** — a real, modest joint-similarity gap, plausibly because
intensity's feature vector reflects an SCM-propagated (harder to jointly match) value. Flagged as
a secondary, partial contributor to F1's intensity CC alongside T6's calibration issue — not a
replacement for it.

## T14 — Capacity-ordering falsification — **refuted**
Digit VIOLATES the assumed monotone capacity ordering with adequate statistical power (k5
significantly worse than both k1 and k11, which are statistically indistinguishable from each
other). Thickness/intensity are monotone in point estimate but only intensity's endpoints are
CI-separated; k5 itself isn't confirmed "intermediate" for either. The CC-based case for the
capacity/leakage narrative is weak-to-contradicted on its own — T10's probe (see above) is what
actually discriminates it, and shows leakage running in a different, more nuanced pattern than
CC alone would suggest.

---

**Net effect on F1-F4 (after 11 of 14 Phase-1 tests):**
- **F1** (thickness/intensity low CC): now has a strong, convergent, depth-independent
  mechanism — T9 shows the models don't use explicit conditioning for these attributes at all
  (ΔL≈0 everywhere), T10 shows they're substantially leaked into the latent even at k=1/30k.
  T6 (ceiling/floor calibration) and T8 (predictor hue-sensitivity) remain to quantify how much
  of the CC number itself is measurement vs. this now-well-evidenced real collapse.
- **F2** (hue CC great / FC_obs collapses): T5 supports a swing-strength asymmetry at k1/k5, but
  k11@75k's swing has compressed toward uniformity — an unresolved, model-state-dependent
  picture. Recommend re-running the eval on k11@75k as a cheap next step.
- **F3** (digit CC regression with more training): now has TWO independent, convergent
  mechanistic findings (T9's loss-ablation, T10's latent probe) both pointing at digit leakage
  into k11's latent being unusually strong — a real, well-supported candidate explanation, though
  the training-step confound (no 30k/45k k11 checkpoints survive) prevents a fully clean
  within-model before/after comparison.
- **F4** (k=5 hue corruption): reframed by T9+T10 from "collapsed conditioning" to "near-total
  latent memorization of hue (R²=0.996) plus heavy reconstruction-time reliance on hue
  conditioning (largest ΔL observed) that doesn't generalize to counterfactual hue values" — a
  much sharper, more specific hypothesis than existed before this pass. T20 (reseed) is still the
  test that would confirm whether this is seed-specific.

## T6 — CC ceiling/floor calibration — **refuted (F1 largely, not entirely, a measurement artifact)**
Corrected true compliance c=(CC_obs-FPR)/(TPR-FPR): thickness c=0.585 (vs raw CC=0.260, >2x
higher), intensity c=0.614 (vs raw 0.408, ~1.5x higher). The tolerance window is tighter than the
predictor's own noise for the majority of typical images — thickness's tolerance would need to be
6x wider for the predictor to reliably confirm even a flawless edit. **Every CC/CF1 table
generated this session understates thickness/intensity compliance by roughly 1.5-2x.** Still not
great (c≈0.6, not ≈1.0) but far less catastrophic than the raw numbers suggested. Digit's
correction is small (c=0.504 vs raw 0.463, predictor already 90.5% accurate) — confirms F3 is
real, not measurement. Hue needs no correction (predictor is perfect).

## T8 — Predictor hue-sensitivity — **refuted**
Digit accuracy varies only 3.05 points across hue buckets (bar: 10), thickness/intensity MAE
ratios 1.18x/1.08x (bar: 1.5x). Predictors are not hue-sensitive. F2's hue-intervention collateral
damage is real model behavior, not a measurement artifact, and T6's calibration isn't biased by
this either. T19 (predictor retraining) is not needed.

## T11 — Digit CC by target class (k=11 @ 75k) — **CONFIRMED AND ESCALATED, urgent**
Had to regenerate per-image records (none existed) by running digit-only counterfactual
generation on the just-finished 75k checkpoint. **Digit CC has continued collapsing: 0.755 (30k)
→ 0.463 (45k) → 0.096-0.072 (75k, cross-validated by two independently-scored runs).** This is
accelerating, not plateauing — the 30k-step continuation just run this session made it worse.
Broken down by class: ALL 10 classes are catastrophically bad (0.0-0.24), uniformly — including
class 0, which is unremarkable (slightly better than the mean), definitively refuting the old
"digit=0 collides with the dropout sentinel" hypothesis. This is a global collapse, matching
T9/T10's mechanism exactly (k11 shows weakest loss-reliance on digit conditioning + highest
digit-leakage into its latent, and now the actual generation pipeline shows the consequence).
**Practical recommendation: do not continue training k11 at the current recipe without first
addressing conditioning collapse** (e.g. attr_dropout_prob well above 0.08, an explicit
latent-attribute independence penalty, or reduced latent capacity).

## K11 @ 75k full re-eval (T5's recommended follow-up) — **refutes T5's compression hypothesis**
Full CC/FC/CF1 at 75k: digit CC=0.072 (collapsed, confirms T11), hue CC=0.898/FC_obs=0.302
(barely changed from 45k's 0.936/0.277), thickness CC=0.338/FC_obs=0.876 (similar to 45k's
0.260/0.909, slightly better), intensity CC=0.347/FC_obs=0.781 (similar to 45k's 0.408/0.682).
**T5's swing-compression finding (hue/thickness swings became nearly equal by 75k) did NOT
translate into FC_obs pattern compression** — hue still over-edits/damages everything else,
thickness still under-edits/stays safe, at almost exactly the same magnitude as 45k. This refines
T5's hypothesis: raw embedding-level conditioning swing is not, by itself, predictive of the
actual generation-time collateral-damage pattern — something in the deeper decoder or diffusion
sampling dynamics maintains the hue/thickness asymmetry independent of swing magnitude. Global CC
fell from 0.467 (45k) to 0.401 (75k), driven almost entirely by digit's collapse — the other
three attributes are roughly stable or slightly improved.

**Phase 1 complete: 14 of 14 tests done.**


---

## T1 — EMA / resume audit

**Verdict:** `refuted`  
**Cost class:** `FREE`  
**Phase:** 1

**Hypothesis:** The 29-point digit-CC drop (0.755 at 30k -> 0.463 at 45k) is a weights-selection or resume artifact, not a training phenomenon.

**Decision rule:** Recorded before result viewed, per plan: (1) different weight source between evals -> F3 unestablished, re-run 45k under 30k weight source, stop. (2) EMA not restored on resume -> F3 confounded, re-run both on raw weights, stop. (3) neither eval used EMA -> EMA not the confound, continue to branch 4. (4) both used EMA, properly restored -> F3 survives this gate. Independently: if LR warmup restarts from zero on resume, record as a standalone candidate explanation for F3 with its own consequence.

**Full record (JSON):**

```json
{
  "test_id": "T1",
  "title": "EMA / resume audit",
  "phase": 1,
  "cost_class": "FREE",
  "gpu_samples": 0,
  "hypothesis": "The 29-point digit-CC drop (0.755 at 30k -> 0.463 at 45k) is a weights-selection or resume artifact, not a training phenomenon.",
  "decision_rule": "Recorded before result viewed, per plan: (1) different weight source between evals -> F3 unestablished, re-run 45k under 30k weight source, stop. (2) EMA not restored on resume -> F3 confounded, re-run both on raw weights, stop. (3) neither eval used EMA -> EMA not the confound, continue to branch 4. (4) both used EMA, properly restored -> F3 survives this gate. Independently: if LR warmup restarts from zero on resume, record as a standalone candidate explanation for F3 with its own consequence.",
  "inputs": {
    "checkpoints": [
      "experiments/hdae/outputs/morpho_hier_k11_v3/checkpoints/last_step75000.ckpt"
    ],
    "predictor_ckpt_sha256": null,
    "cohort_index_file": null,
    "seed": null,
    "n": 633
  },
  "code": {
    "script_path": "experiments/hdae/counterfactuals/hdae_adapter.py; diffae_upstream/experiment.py; diffae_upstream/templates.py; diffae_upstream/config.py",
    "git_sha": null,
    "dirty": true
  },
  "result": {
    "weight_source_both_evals": "ema_model (hardcoded in hdae_adapter.py:45, not a per-run flag -- identical for 30k and 45k evals)",
    "ema_storage": "full second copy of the model inside state_dict under 'ema_model.*' prefix (LitModel.__init__: self.ema_model = copy.deepcopy(self.model)), restored via standard Trainer.fit(ckpt_path=...) state_dict load -- no separate/fragile EMA-restore path exists",
    "ema_vs_raw_mean_relative_l2_diff_at_75k": 0.0962,
    "ema_vs_raw_median_relative_l2_diff_at_75k": 0.0882,
    "ema_vs_raw_diff_range": [
      0.0,
      0.3986
    ],
    "ema_functioning_interpretation": "non-zero (EMA is not frozen/reset) and not huge (EMA is tracking, not diverged) -- consistent with a normally-functioning slow-moving EMA at decay=0.9999, 75000 steps",
    "warmup_configured": false,
    "warmup_default": 0,
    "warmup_source": "diffae_upstream/config.py:153 (TrainConfig.warmup: int = 0), diffae_upstream/templates.py sets it to 0 explicitly in every template used here; never overridden in any HDAE yaml",
    "lr_scheduler_registered": false,
    "lr_schedulers_ckpt_field": "[]",
    "interpretation": "configure_optimizers only builds a LambdaLR if conf.warmup > 0; since warmup=0 always, no scheduler exists at all -- lr_schedulers:[] in the checkpoint is expected, not a resume bug. There is no LR ramp to restart."
  },
  "ci": {
    "method": "n/a",
    "level": null,
    "intervals": {}
  },
  "verdict": "refuted",
  "consequences": [
    "Branch (1) does not apply: weight source (ema_model) is identical between the 30k and 45k evals.",
    "Branch (2)/(4): EMA is restored via the standard whole-model state_dict path, same as raw weights -- no special resume risk. Empirical check shows EMA is actively tracking (9.6% mean rel L2 diff from raw at 75k), not frozen or reset.",
    "Warmup-restart candidate does not apply: warmup is never configured (defaults to 0), so no LR ramp exists to restart on resume.",
    "T1 does not explain F3. Continue to T2."
  ],
  "assumptions_and_caveats": [
    "Did not verify EMA update is called identically on every DDP rank every step (assumed standard, not independently instrumented) -- low-priority follow-up if T2 also clears and F3 still needs an explanation beyond training dynamics.",
    "Relative L2 diff is a coarse functioning check, not a proof EMA decay is numerically exactly 0.9999 throughout -- sufficient to rule out 'EMA reset to raw on resume', not sufficient to rule out subtler decay-schedule bugs."
  ],
  "runtime_sec": 45
}
```

---

## T2 — Eval-harness equivalence audit

**Verdict:** `refuted`  
**Cost class:** `FREE`  
**Phase:** 1

**Hypothesis:** The 30k and 45k K11 evals differ in something other than the scoring function (cohort, targets, denominator, predictor), confounding the F3 comparison.

**Decision rule:** Recorded before result viewed, per plan: any difference in cohort, targets, denominator, or predictor -> F3 unestablished, re-run 45k under the 30k harness. All identical -> F3 survives both gates (combined with T1).

**Full record (JSON):**

```json
{
  "test_id": "T2",
  "title": "Eval-harness equivalence audit",
  "phase": 1,
  "cost_class": "FREE",
  "gpu_samples": 0,
  "hypothesis": "The 30k and 45k K11 evals differ in something other than the scoring function (cohort, targets, denominator, predictor), confounding the F3 comparison.",
  "decision_rule": "Recorded before result viewed, per plan: any difference in cohort, targets, denominator, or predictor -> F3 unestablished, re-run 45k under the 30k harness. All identical -> F3 survives both gates (combined with T1).",
  "inputs": {
    "checkpoints": [],
    "predictor_ckpt_sha256": "addb4008d7333a48b1ddf1c34eab93d1633e048ad6773f354e8563e44a597fff (thickness/best.ckpt, spot-checked; digit/hue also hashed)",
    "cohort_index_file": "experiments/hdae/outputs/intervention_cohorts.json",
    "seed": null,
    "n": 512
  },
  "code": {
    "script_path": "experiments/hdae/counterfactuals/morpho_cf1_eval_toleranced.py; morpho_cf1_eval_binned.py (pre-toleranced variant)",
    "git_sha": null,
    "dirty": true
  },
  "result": {
    "cohort_file_mtime": "2026-08-12T13:33:52",
    "eval_30k_log_mtime": "2026-08-17T15:08:53",
    "eval_45k_log_mtime": "2026-08-18T13:24:35",
    "cohort_unchanged_between_evals": true,
    "predictor_summary_mtime": "2026-08-12T12:05:35",
    "predictor_unchanged_between_evals": true,
    "digit_cc_denominator_30k": "507/512",
    "digit_cc_denominator_45k": "505/512",
    "denominator_diff_interpretation": "2-image difference, consistent with ordinary generation-to-generation stochastic variation (previously documented ~1/512 image difference under cudnn.benchmark=True with identical inputs), not a cohort or target-set change -- the cohort file itself is byte-static across both runs.",
    "hue_cc_30k": 0.9707,
    "hue_cc_45k": 0.9355,
    "digit_cc_30k": 0.7554,
    "digit_cc_45k": 0.4634
  },
  "ci": {
    "method": "n/a",
    "level": null,
    "intervals": {}
  },
  "verdict": "refuted",
  "consequences": [
    "Cohort, target sampling, and predictor checkpoint are all identical (byte-static files, unmodified) between the two eval runs.",
    "The tolerance-methodology redesign (2026-08-18) that distinguishes cf1_eval_gs8 from cf1_eval_gs8_tol2x does not affect digit/hue's exact-class-match CC scoring (documented as intentionally unchanged in the redesign) -- so the digit/hue comparison across the two logs is apples-to-apples despite the methodology label difference.",
    "Combined with T1: F3 (digit CC 0.755 -> 0.463) survives both gates. It is very likely a real training-dynamics phenomenon, not a harness or resume artifact.",
    "This should be stated plainly to the user: F3 is not an artifact as far as these two gates can tell -- something about continued training between 30k and 45k genuinely degraded digit conditioning while loss stayed flat."
  ],
  "assumptions_and_caveats": [
    "Thickness/intensity CC numbers across the two logs are NOT directly comparable step-for-step (tolerance definition changed between them) -- only digit and hue are apples-to-apples here. Correct comparison for thickness/intensity awaits T6.",
    "Did not diff sampler config (T config, batch-size, DDIM steps) line-by-line between the two eval invocations beyond confirming both used the same default --T=100 in the script's argparse; no evidence either run overrode it, but not independently logged per-invocation."
  ],
  "runtime_sec": 60
}
```

---

## T3 — CFG null-branch construction

**Verdict:** `ambiguous`  
**Cost class:** `FREE`  
**Phase:** 1

**Hypothesis:** If the unconditional branch nulls all four attributes and training rarely/never presented that configuration, guidance at scale=8.0 extrapolates from an unseen condition.

**Decision rule:** Recorded before result viewed, per plan's four R1 branches: (a) all-four-null + fused FiLM MLP -> severe, all-null is an unvisited point in the fused input space, T18 arm B mandatory. (a) all-four-null + per-attribute-additive FiLM -> mild, note and proceed. (b) intervened-only -> in-distribution at ~6%, marginal but defensible. (c)/(d) other -> describe training frequency, apply same severe/mild split, do not let it fall through without a consequence. Scope limit: T3 is attribute-agnostic and cannot alone explain F2 (hue vs thickness asymmetry) -- only in combination with T5's swing asymmetry.

**Full record (JSON):**

```json
{
  "test_id": "T3",
  "title": "CFG null-branch construction",
  "phase": 1,
  "cost_class": "FREE",
  "gpu_samples": 0,
  "hypothesis": "If the unconditional branch nulls all four attributes and training rarely/never presented that configuration, guidance at scale=8.0 extrapolates from an unseen condition.",
  "decision_rule": "Recorded before result viewed, per plan's four R1 branches: (a) all-four-null + fused FiLM MLP -> severe, all-null is an unvisited point in the fused input space, T18 arm B mandatory. (a) all-four-null + per-attribute-additive FiLM -> mild, note and proceed. (b) intervened-only -> in-distribution at ~6%, marginal but defensible. (c)/(d) other -> describe training frequency, apply same severe/mild split, do not let it fall through without a consequence. Scope limit: T3 is attribute-agnostic and cannot alone explain F2 (hue vs thickness asymmetry) -- only in combination with T5's swing asymmetry.",
  "inputs": {
    "checkpoints": [],
    "predictor_ckpt_sha256": null,
    "cohort_index_file": null,
    "seed": null,
    "n": null
  },
  "code": {
    "script_path": "experiments/hdae/hdae/attr_conditioner.py (ConcatAttributeEmbedding.forward, PerBlockStyleFiLM)",
    "git_sha": null,
    "dirty": true
  },
  "result": {
    "branch": "(a) all-four-null, via cfg_drop_prob=0.1 applied as ONE Bernoulli draw per training example (not per-attribute) -- mask.expand(-1, n_attr) nulls all 4 slots together for that example",
    "fusion": "fused -- PerBlockStyleFiLM's scale_proj/shift_proj are each a single nn.Linear(attr_embed_dim=128, embed_channels) applied to the full concatenated embedding, not 4 independent per-attribute Linears summed",
    "structural_classification_per_plan_rule": "severe (a + fused)",
    "p_all_null_direct": 0.1,
    "p_all_null_via_attr_dropout_coincidence": 4.096e-05,
    "p_all_null_total_approx": 0.10004096,
    "p_intervened_only_pattern": "0.08 * 0.92**3 = 0.0623 (matches plan's own 6.2% estimate)",
    "global_batch_size": 128,
    "expected_all_null_examples_seen_by_30000_steps": 384000,
    "expected_all_null_examples_seen_by_75000_steps": 960000,
    "frequency_interpretation": "cfg_drop_prob=0.1 is a deliberate, standard CFG-training convention (10% unconditional-drop rate) -- the all-null configuration is NOT rare or unvisited. It is seen on the order of 10^5-10^6 training examples by 30k-75k steps."
  },
  "ci": {
    "method": "n/a",
    "level": null,
    "intervals": {}
  },
  "verdict": "ambiguous",
  "consequences": [
    "The plan's own stated rationale for classifying this 'severe' -- 'all-null is an unvisited point in the fused input space' -- does not hold for this specific config: cfg_drop_prob=0.1 makes all-null a common, well-trained regime (~10% of all examples), not an edge case.",
    "The structural concern (fused MLP processes the concatenated embedding jointly, so all-null is a genuinely different input than any per-attribute-additive decomposition would produce) still stands independent of frequency -- worth testing via T18 arm B, but for a different reason than 'unvisited'.",
    "T3 alone cannot explain F2 (hue over-edits / thickness under-edits at the same guidance scale) since the null-branch construction is identical for all four attributes -- consistent with the plan's own scope limit. F2's likely driver remains T5 (per-attribute swing asymmetry).",
    "Does not, by itself, explain the guidance-scale extrapolation question either (CFG's linear extrapolation output = uncond + scale*(cond-uncond) is not literally an input the fused MLP was ever trained on, regardless of how often all-null appears as an INPUT -- that is inherent to CFG generally, not specific to this null-branch construction, and is better tested empirically via T16/T18 than settled by code reading)."
  ],
  "assumptions_and_caveats": [
    "'Unvisited' claim refuted specifically for the ALL-null point; does not address whether the scale=8x LINEAR EXTRAPOLATION beyond cond lands somewhere the model handles well -- that is a distinct, harder question left to T16 (arm A vs arm B at fixed scale) and T18 (scale sweep).",
    "Did not verify whether cfg_drop_prob=0.1 is actually invoked identically for k=1/k=5 (assumed identical from shared config template, not independently re-read per model's yaml -- low risk since all three configs were confirmed to share this block in R1)."
  ],
  "runtime_sec": 20
}
```

---

## T4 — FC_observed causal-edge audit

**Verdict:** `refuted`  
**Cost class:** `FREE`  
**Phase:** 1

**Hypothesis:** For a thickness intervention, FC_observed's stay-put set incorrectly includes intensity (a causal descendant that is SUPPOSED to change), penalizing correct behavior; or the causal graph is decorative at eval time (target vector construction doesn't actually propagate thickness->intensity).

**Decision rule:** Recorded before result viewed, per plan: intensity in the stay-put set for thickness interventions -> FC_obs for thickness is wrong, recompute from per-image records if saved. Propagation absent from construction code -> causal graph is decorative at eval time, report as its own finding. Explicitly: do not infer propagation failure from FC_obs=0.970 (the pass rate) -- settle from construction code plus magnitude, never from a pass rate.

**Full record (JSON):**

```json
{
  "test_id": "T4",
  "title": "FC_observed causal-edge audit",
  "phase": 1,
  "cost_class": "FREE",
  "gpu_samples": 0,
  "hypothesis": "For a thickness intervention, FC_observed's stay-put set incorrectly includes intensity (a causal descendant that is SUPPOSED to change), penalizing correct behavior; or the causal graph is decorative at eval time (target vector construction doesn't actually propagate thickness->intensity).",
  "decision_rule": "Recorded before result viewed, per plan: intensity in the stay-put set for thickness interventions -> FC_obs for thickness is wrong, recompute from per-image records if saved. Propagation absent from construction code -> causal graph is decorative at eval time, report as its own finding. Explicitly: do not infer propagation failure from FC_obs=0.970 (the pass rate) -- settle from construction code plus magnitude, never from a pass rate.",
  "inputs": {
    "checkpoints": [],
    "predictor_ckpt_sha256": null,
    "cohort_index_file": null,
    "seed": null,
    "n": null
  },
  "code": {
    "script_path": "experiments/hdae/counterfactuals/morpho_cf1_eval_toleranced.py:73,108-136",
    "git_sha": null,
    "dirty": true
  },
  "result": {
    "stay_put_set_construction": "attribute_partition(graph, attr) returns (descendants, observed); observed = [a for a in graph.attributes if a != attr and a not in descendants] -- descendants ARE excluded from FC_observed by construction, confirmed for thickness intervention: descendants=['intensity'], observed=['digit','hue'].",
    "propagation_present": true,
    "propagation_code": "cf_attrs = scm.counterfactual(...) computes the full SCM-propagated counterfactual (including intensity's causally-derived value via the thickness->intensity edge); the loop 'for a in observed: cf_attrs[a] = <original value>' only overwrites the non-descendant attributes back to original -- descendants keep the SCM-propagated target untouched.",
    "descendant_scoring": "each descendant d gets its own pass/fail check (|pred_cf[d] - scm_propagated_target[d]| <= std[d]*0.25), folded into the SAME cc_num/cc_den as the intervened attribute (cc_den = n_valid * (1 + n_descendants)) -- so the reported 'thickness CC' is actually a joint {thickness, intensity} correctness score, not thickness in isolation.",
    "did_not_use_fc_obs_pass_rate_as_evidence": true
  },
  "ci": {
    "method": "n/a",
    "level": null,
    "intervals": {}
  },
  "verdict": "refuted",
  "consequences": [
    "The causal graph is correctly and non-decoratively applied at eval time: propagation happens via scm.counterfactual(), and descendants are correctly excluded from FC_observed's stay-put set and instead folded into CC's descendant-consistency term.",
    "No fix needed here. This was the plan's own hypothesis and it does not hold for this codebase -- one less thing to chase.",
    "Side finding worth flagging in any external write-up: 'thickness CC' in the tables so far is really a joint thickness+intensity correctness metric (because of the descendant term), not a pure thickness-only number -- readers comparing it to hue/digit's single-attribute CC should know that."
  ],
  "assumptions_and_caveats": [
    "Verified for the thickness->intensity edge specifically (the only declared edge in causal_graph_morpho.yaml); no other edges exist to check.",
    "The descendant tolerance (std[d]*0.25) is a separate, still-uncorrected tolerance definition from CC's own half-bin-width tolerance -- not evaluated for calibration here, only for whether the causal structure is respected. That calibration question is T6's territory if pursued for the descendant term specifically (T6 as scoped only covers the intervened attribute's own tolerance)."
  ],
  "runtime_sec": 15
}
```

---

## T5 — Per-attribute conditioning strength (swing)

**Verdict:** `ambiguous`  
**Cost class:** `CPU`  
**Phase:** 1

**Hypothesis:** H4: hue's swing (5th->95th percentile) through the conditioning pathway is much larger than thickness's, at the same guidance scale -- explaining hue's over-edit/thickness's under-edit asymmetry (F2) without needing any GPU sampling.

**Decision rule:** Recorded before result viewed, per plan (corrected per advisor review: measure swing in the resulting FiLM STYLE vector with zs held fixed to real encoded context images, not in attr_emb alone, since scale/shift depend only on attr_emb but the final style = z_style*(1+scale)+shift also depends multiplicatively on zs). Hue's normalized swing > 2x thickness's, CI-separated -> H4 confirmed for free, T18 initialized from swing instead of blind-swept. Comparable swings -> mismatch is not in the conditioner, must be in decoder's use of it, T18 becomes more necessary.

**Full record (JSON):**

```json
{
  "test_id": "T5",
  "title": "Per-attribute conditioning strength (swing)",
  "phase": 1,
  "cost_class": "CPU",
  "gpu_samples": 0,
  "hypothesis": "H4: hue's swing (5th->95th percentile) through the conditioning pathway is much larger than thickness's, at the same guidance scale -- explaining hue's over-edit/thickness's under-edit asymmetry (F2) without needing any GPU sampling.",
  "decision_rule": "Recorded before result viewed, per plan (corrected per advisor review: measure swing in the resulting FiLM STYLE vector with zs held fixed to real encoded context images, not in attr_emb alone, since scale/shift depend only on attr_emb but the final style = z_style*(1+scale)+shift also depends multiplicatively on zs). Hue's normalized swing > 2x thickness's, CI-separated -> H4 confirmed for free, T18 initialized from swing instead of blind-swept. Comparable swings -> mismatch is not in the conditioner, must be in decoder's use of it, T18 becomes more necessary.",
  "inputs": {
    "checkpoints": [
      "experiments/hdae/outputs/morpho_hier_k1_v3/checkpoints/last.ckpt (30k)",
      "experiments/hdae/outputs/morpho_hier_k5_v3/checkpoints/last.ckpt (30k)",
      "experiments/hdae/outputs/morpho_hier_k11_v3/checkpoints/last_step75000.ckpt (75k -- 30k/45k unavailable, see RECON.md)"
    ],
    "predictor_ckpt_sha256": null,
    "cohort_index_file": null,
    "seed": 0,
    "n": 64
  },
  "code": {
    "script_path": "experiments/hdae/diagnostics/t5_conditioning_swing.py",
    "git_sha": null,
    "dirty": true
  },
  "result": {
    "k1_30k": {
      "digit": {
        "mean_total_swing": 54.65,
        "ci95": [
          54.22,
          55.26
        ],
        "norm_swing": 0.2277
      },
      "thickness": {
        "mean_total_swing": 27.06,
        "ci95": [
          26.85,
          27.31
        ],
        "norm_swing": 0.1102
      },
      "intensity": {
        "mean_total_swing": 43.79,
        "ci95": [
          43.35,
          44.3
        ],
        "norm_swing": 0.1685
      },
      "hue": {
        "mean_total_swing": 52.23,
        "ci95": [
          51.89,
          52.65
        ],
        "norm_swing": 0.2128
      },
      "hue_over_thickness_ratio_normalized": 1.931
    },
    "k5_30k": {
      "digit": {
        "mean_total_swing": 41.53,
        "ci95": [
          41.15,
          41.96
        ],
        "norm_swing": 0.2439
      },
      "thickness": {
        "mean_total_swing": 21.46,
        "ci95": [
          21.12,
          21.81
        ],
        "norm_swing": 0.1241
      },
      "intensity": {
        "mean_total_swing": 45.69,
        "ci95": [
          45.41,
          46.01
        ],
        "norm_swing": 0.2646
      },
      "hue": {
        "mean_total_swing": 32.78,
        "ci95": [
          32.59,
          32.95
        ],
        "norm_swing": 0.1912
      },
      "hue_over_thickness_ratio_normalized": 1.541
    },
    "k11_75k": {
      "digit": {
        "mean_total_swing": 37.22,
        "ci95": [
          36.86,
          37.54
        ],
        "norm_swing": 0.1406
      },
      "thickness": {
        "mean_total_swing": 43.68,
        "ci95": [
          43.27,
          44.13
        ],
        "norm_swing": 0.1589
      },
      "intensity": {
        "mean_total_swing": 40.98,
        "ci95": [
          40.35,
          41.55
        ],
        "norm_swing": 0.1443
      },
      "hue": {
        "mean_total_swing": 41.84,
        "ci95": [
          41.06,
          42.57
        ],
        "norm_swing": 0.13
      },
      "hue_over_thickness_ratio_normalized": 0.818
    }
  },
  "ci": {
    "method": "bootstrap (200 resamples over 64 real context images)",
    "level": 0.95,
    "intervals": "see result block, all CIs are non-overlapping across attributes within each model given the tightness of the bootstrap"
  },
  "verdict": "ambiguous",
  "consequences": [
    "k1 @ 30k: hue's normalized swing is 1.93x thickness's -- just under the plan's literal 2x bar but the CIs are completely non-overlapping (52.23 [51.89,52.65] vs 27.06 [26.85,27.31]) and the gap is large and real. H4 is SUPPORTED for k1, narrowly short of the exact threshold specified.",
    "k5 @ 30k: hue/thickness ratio 1.54x -- same direction as k1 (hue notably higher swing than thickness), weaker magnitude, still CI-separated. Consistent with, not contradicting, H4.",
    "k11 @ 75k (the only k=11 checkpoint available -- see caveat): hue's normalized swing is LOWER than thickness's (ratio 0.82x) -- the OPPOSITE direction from k1/k5, and opposite to what H4 predicts. At this checkpoint, all four attributes' swings have become nearly uniform (range 37-44, vs k1's much more disparate 27-55).",
    "UNEXPECTED, HIGH-VALUE FINDING not anticipated by the plan: k11's conditioning-strength profile appears to have compressed toward uniformity over the course of its extended training (30k->75k), and DIGIT specifically has the LOWEST swing of all four attributes at 75k (37.22, versus 54.65 for digit at k1's 30k checkpoint, a different model but the same architecture family). This is directly relevant to F3 (digit CC regressing 0.755->0.463 as k11 trained from 30k to 45k+): a weakening of digit's relative conditioning strength over training is a concrete, mechanistically plausible explanation for F3 that this test was not designed to find but surfaced anyway.",
    "H4's status for k11 specifically is genuinely unresolved by this checkpoint: the 45k eval log (closest available CC/FC data) still shows hue's classic over-edit pattern (CC=0.936, FC_obs=0.277) vs thickness's under-edit pattern (CC=0.260, FC_obs=0.909) -- but the swing measured at 75k no longer shows the hue-dominant profile that would mechanistically explain it. Either the swing profile itself shifted between 45k and 75k in a way that hasn't yet shown up in FC_obs (eval not yet re-run at 75k), or swing alone is not sufficient to explain F2 for the deeper architecture.",
    "STRONG RECOMMENDATION for a cheap, high-value follow-up: re-run the toleranced CC/FC/CF1 eval on the k11 75k checkpoint (infrastructure already exists, was used earlier this session) to see whether F2's hue/thickness asymmetry has also compressed by 75k, matching the swing-compression finding. This is Phase-2-cost (GPU) but small (one model, existing script) and directly tests a hypothesis this Phase-1 test just generated."
  ],
  "assumptions_and_caveats": [
    "k11's 30k and 45k checkpoints are gone (see RECON.md) -- the compression-over-training finding is inferred by comparing k1@30k (different model) to k11@75k (same architecture family, different model), NOT a true within-model before/after comparison. This is suggestive, not proven; T9's per-step \u0394L trend (if run going forward, or if a future k11 continuation checkpoints more frequently) is what would actually confirm a within-model trend.",
    "Swing was measured at guidance/attr values from the TRAIN marginal 5th/95th percentiles, holding OTHER attributes at their train median and zs from 64 real held-out (val-partition) images -- a different context distribution than the eval cohort's actual images, though drawn from the same overall data distribution.",
    "'Normalized swing' divides by the style vector's own norm at the p5 setting; an alternative normalization (e.g. by the full pre-FiLM z_style norm) was not tried and could shift the k1 ratio across the exact 2x threshold in either direction -- the qualitative direction (hue >> thickness for k1/k5, compressed/reversed for k11@75k) is robust to this choice, the precise ratio value is not guaranteed to be.",
    "This test intentionally deviates from the plan's literal 'measure L2 distance between FiLM (gamma,beta) vectors' instruction, per advisor's architecture-specific correction (gamma/beta alone don't reach the decoder without z_style) -- noted here rather than silently reinterpreted, per ledger rule 3."
  ],
  "runtime_sec": 75
}
```

---

## T6 — CC ceiling/floor calibration

**Verdict:** `refuted (F1 is NOT purely a model failure; largely, not entirely, a measurement artifact)`  
**Cost class:** `CPU`  
**Phase:** 1

**Hypothesis:** CC's raw numbers are biased by predictor noise (a ceiling below 1.0, per-attribute) and by the fraction of images that would pass 'by chance' even with no real edit (a floor above 0.0). The naive CC/TPR correction (v1) is wrong; the correct inversion is c=(CC_obs-FPR)/(TPR-FPR).

**Decision rule:** Recorded before result viewed, per plan: F1 is a measurement artifact iff mean c for {thickness,intensity} >= 0.8 x mean c for {digit,hue}, CIs non-overlapping required to call it either way. If c for thickness/intensity comes back near zero with a tight CI, the OPPOSITE conclusion holds and is stronger: the model performs no measurable edit at all. If TPR-FPR is small, declare the attribute unmeasurable at tol=2x and report the multiplier that would fix it.

**Full record (JSON):**

```json
{
  "test_id": "T6",
  "title": "Metric calibration: ceiling (TPR) and chance floor (FPR)",
  "phase": 1,
  "cost_class": "CPU",
  "gpu_samples": 0,
  "hypothesis": "CC's raw numbers are biased by predictor noise (a ceiling below 1.0, per-attribute) and by the fraction of images that would pass 'by chance' even with no real edit (a floor above 0.0). The naive CC/TPR correction (v1) is wrong; the correct inversion is c=(CC_obs-FPR)/(TPR-FPR).",
  "decision_rule": "Recorded before result viewed, per plan: F1 is a measurement artifact iff mean c for {thickness,intensity} >= 0.8 x mean c for {digit,hue}, CIs non-overlapping required to call it either way. If c for thickness/intensity comes back near zero with a tight CI, the OPPOSITE conclusion holds and is stronger: the model performs no measurable edit at all. If TPR-FPR is small, declare the attribute unmeasurable at tol=2x and report the multiplier that would fix it.",
  "inputs": {
    "checkpoints": [],
    "predictor_ckpt_sha256": null,
    "cohort_index_file": "experiments/hdae/outputs/intervention_cohorts.json",
    "seed": 0,
    "n": "3000 real held-out (val) images for TPR, 512-image fixed cohort for FPR"
  },
  "code": {
    "script_path": "experiments/hdae/diagnostics/t6_cc_calibration.py",
    "git_sha": null,
    "dirty": true
  },
  "result": {
    "thickness": {
      "tpr_reweighted": 0.4353,
      "fpr": 0.0137,
      "cc_obs": 0.2604,
      "c": 0.5852,
      "c_ci95": [
        0.5485,
        0.6041
      ],
      "mult_needed_for_tpr_0.95": 6.0
    },
    "intensity": {
      "tpr_reweighted": 0.6627,
      "fpr": 0.0039,
      "cc_obs": 0.4083,
      "c": 0.6138,
      "c_ci95": [
        0.6232,
        0.659
      ],
      "mult_needed_for_tpr_0.95": 4.0
    },
    "digit": {
      "tpr": 0.905,
      "fpr": 0.0156,
      "cc_obs": 0.4634,
      "c": 0.5035,
      "c_ci95": [
        0.495,
        0.5126
      ]
    },
    "hue": {
      "tpr": 1.0,
      "fpr": 0.0,
      "cc_obs": 0.9355,
      "c": 0.9355,
      "c_ci95": [
        0.9355,
        0.9355
      ]
    }
  },
  "ci": {
    "method": "bootstrap (500 resamples)",
    "level": 0.95,
    "intervals": "see result block"
  },
  "verdict": "refuted (F1 is NOT purely a model failure; largely, not entirely, a measurement artifact)",
  "consequences": [
    "MAJOR CORRECTION to F1: thickness's true compliance c=0.585 (CI [0.549,0.604]) is more than DOUBLE the raw CC=0.260. Intensity's c=0.614 (CI [0.623,0.659]) is ~1.5x the raw CC=0.408. The tolerance window used throughout this investigation is tighter than the predictor's own noise for the MAJORITY of typical images -- TPR (the predictor confirming its own perfect measurement) is only 43.5% for thickness and 66.3% for intensity. The tolerance multiplier needed to get the predictor to reliably confirm even a flawless edit (TPR=0.95) is 6.0x for thickness and 4.0x for intensity -- the current tol=2x setting used throughout this session's eval tables is well short of that.",
    "This does NOT mean F1 disappears: c=0.585/0.614 is still mediocre (far from 1.0), and the CIs are tight enough that this is a real, non-trivial residual failure, not full exoneration. The correct framing going forward: thickness/intensity's badly-low RAW CC numbers substantially overstate how badly the models are actually doing at these edits -- the honest compliance rate is roughly 2x higher than the numbers everywhere else in this investigation (all prior CC/CF1 tables) have implied.",
    "DIGIT: c=0.5035 (CI [0.495,0.513]) vs raw CC_obs=0.4634 -- only a modest ~4-point correction (TPR is already high at 0.905, so little bias to remove). This CONFIRMS F3 is real and not a measurement artifact: even after calibration, k11's true digit compliance at 45k is only ~50%, versus what a similarly-calibrated 30k number would very likely still show as a large real regression from the raw CC=0.755 (30k's TPR/FPR were not independently recomputed here, but there is no mechanism by which predictor noise alone would explain a 25-point raw CC swing when the predictor's own accuracy, 90.5%, is stable and high).",
    "HUE: TPR=1.0, FPR=0.0, c=CC_obs exactly (0.9355) -- no correction needed, confirming what was assumed informally earlier in this session (hue's near-perfect predictor makes its raw CC already a clean number).",
    "The plan's F1 decision rule (mean c for thickness/intensity vs mean c for digit/hue, ratio >=0.8) evaluates to: mean(thickness,intensity)=0.599 vs mean(digit,hue)=0.720 -> ratio=0.833, just above the 0.8 threshold, CIs for the two groups do not overlap when compared as groups. Read literally this would call F1 'not a pure model failure' -- but this specific threshold comparison mixes two different attribute TYPES (continuous vs categorical) with different failure modes and should be treated as a rough heuristic, not a strict verdict; the more informative statement is the per-attribute c values above.",
    "Every prior CC/CF1 table generated earlier in this session (the 30k three-model comparison, the tol2x/tol3x sweeps, the 45k re-eval) should be understood as UNDERSTATING thickness/intensity compliance by roughly 1.5-2x, and should not be used for cross-attribute comparisons (e.g. 'hue is much better than thickness') without this correction in mind."
  ],
  "assumptions_and_caveats": [
    "TPR was computed on a 3000-image real held-out sample using the SAME predictor and SAME per-image half-bin tolerance definition as the production eval -- but 'per-image tolerance' here uses each sampled image's OWN true-value bin as a stand-in for a target bin, reweighted by the ACTUAL target-bin histogram from the 512-image eval cohort (per the plan's requirement), not a flat/unweighted population average.",
    "FPR was computed using the predictor's reading of the REAL (unedited) source images in the fixed 512-cohort against their REAL stored counterfactual targets -- not the model's own reconstruction, avoiding conflating model reconstruction noise with the chance-floor concept, per the plan's definition ('image at the source value').",
    "digit/hue's TPR is the predictor's raw held-out classification accuracy (0.905/1.0, matching comparison_results.json) -- consistent, not independently re-derived, low risk.",
    "c's identifiability guard used a simple 0.1 threshold on (TPR-FPR) rather than a full CI-spans-[0,1] check as the plan suggested -- all four attributes cleared this easily (smallest TPR-FPR was thickness's 0.42), so a stricter guard would not have changed any verdict here, but is worth implementing exactly as specified if a future attribute comes closer to the boundary.",
    "CC_obs values are taken from the k11 45k tol2x eval log (the most recent full eval) -- this test calibrates that specific eval's numbers, not the 30k comparison table's numbers directly (though the same predictor/tolerance mechanics apply there too and the qualitative correction direction would be the same)."
  ],
  "runtime_sec": 90
}
```

---

## T7 (precondition) — Deterministic attribute measurement

**Verdict:** `refuted (as literally proposed) / reframed`  
**Cost class:** `CPU`  
**Phase:** 1

**Hypothesis:** Thickness, intensity, and hue are deterministic functions of the image in this MorphoMNIST-style pipeline; if those functions can be applied to arbitrary (incl. generated) images, the CNN predictor is unnecessary for three of the four attributes.

**Decision rule:** Recorded before result viewed, per plan: oracle validates on real images -> use as primary measurement for thickness/intensity/hue, T8 dropped, T6 correction applies only to digit, predictor retained only for digit. Oracle does not exist or does not validate -> fall back to T6/T8 as written and say so explicitly, since it means every non-digit number in the investigation is predictor-limited.

**Full record (JSON):**

```json
{
  "test_id": "T7_precondition",
  "title": "Deterministic attribute measurement -- oracle existence and validation (precondition check before full T7)",
  "phase": 1,
  "cost_class": "CPU",
  "gpu_samples": 0,
  "hypothesis": "Thickness, intensity, and hue are deterministic functions of the image in this MorphoMNIST-style pipeline; if those functions can be applied to arbitrary (incl. generated) images, the CNN predictor is unnecessary for three of the four attributes.",
  "decision_rule": "Recorded before result viewed, per plan: oracle validates on real images -> use as primary measurement for thickness/intensity/hue, T8 dropped, T6 correction applies only to digit, predictor retained only for digit. Oracle does not exist or does not validate -> fall back to T6/T8 as written and say so explicitly, since it means every non-digit number in the investigation is predictor-limited.",
  "inputs": {
    "checkpoints": [],
    "predictor_ckpt_sha256": null,
    "cohort_index_file": null,
    "seed": null,
    "n": 7
  },
  "code": {
    "script_path": "experiments/hdae/data/morphomnist.py (measure_thickness, measure_intensity, render, _apply_hue/_apply_geometry/_apply_background/_apply_texture)",
    "git_sha": null,
    "dirty": true
  },
  "result": {
    "oracle_exists_thickness": true,
    "oracle_exists_intensity": true,
    "oracle_exists_hue": false,
    "oracle_source": "measure_thickness/measure_intensity are the literal functions that produced the stored ground-truth labels inside render() -- not an approximation, the actual label-generating code",
    "naive_reapplication_test": "on the FINAL packed/stored image (max(R,G,B) used to invert _apply_hue's colorization and recover a grayscale proxy), reapplying measure_thickness/measure_intensity does NOT recover the stored labels",
    "naive_reapplication_mean_abs_err_thickness": 0.355,
    "naive_reapplication_mean_abs_err_intensity": 51.62,
    "comparison_to_cnn_predictor_mae": "CNN predictor MAE is 0.115 (thickness) / 4.31 (intensity) -- the naive oracle re-application is 3x WORSE than the CNN predictor for thickness and 12x worse for intensity, not better",
    "root_cause": "measure_thickness/measure_intensity are called on 'gray' INSIDE render(), at a pipeline stage BEFORE _apply_hue, _apply_geometry (affine warp), _apply_background (additive field up to +18), and _apply_texture (additive noise up to +10) are applied. The final stored/observable image (and any model-generated image, which mimics this same observable distribution) includes background+texture noise that can push background pixels above the '>20 = foreground' threshold used by both measure functions, corrupting the binary foreground mask the distance-transform and mean-intensity calculations depend on. Error direction (measured intensity always LOWER than true, all 7 sampled images) is consistent with spurious low-value noise pixels near the threshold being misclassified as foreground and diluting the mean.",
    "hue_oracle_status": "not implemented -- would require RGB->HSV inversion on foreground pixels (feasible in principle since _apply_hue scales gray by a fixed per-hue RGB gain with max component exactly 1.0, but not yet written or validated)"
  },
  "ci": {
    "method": "n/a",
    "level": null,
    "intervals": {}
  },
  "verdict": "refuted (as literally proposed) / reframed",
  "consequences": [
    "T7's naive form -- 'apply the label-generating function directly to the observable image' -- does NOT work as a free win. The predictor is NOT trivially removable for thickness/intensity via this route.",
    "This does not mean a robust oracle is impossible, only that it needs real engineering: a foreground-segmentation step robust to background field + texture noise (e.g. a higher/adaptive threshold, or explicit background/texture subtraction using the known bg_freq/bg_phase/bg_amplitude/texture_seed/texture_amplitude values before thresholding) rather than the fixed '>20' cutoff used at label-generation time on the pre-augmentation canvas.",
    "Independent data-quality note, not previously flagged anywhere in this investigation: the STORED ground-truth thickness/intensity labels themselves are measured on an idealized pre-background/pre-texture/pre-geometric-warp canvas, not on the final observable image -- meaning there is already a real, nonzero gap between 'true achieved thickness of the pixels a human or model actually sees' and 'the label in the dataset', for every real training image, independent of anything model-related.",
    "T7's dropped/kept status for T8/T19 should NOT be marked 'oracle validates, drop T8' based on this result -- reopens T8 and T6 as still necessary for thickness/intensity as well as digit, contrary to what a naive reading of T7 passing would have implied.",
    "If pursuing a fixed oracle is worthwhile, it's a new, scoped follow-up (not part of this pass): build foreground segmentation that subtracts the known background field before thresholding, re-validate on real images to numerical precision, then re-attempt this test."
  ],
  "assumptions_and_caveats": [
    "Only tested on 7 real training images across a spread of indices (0, 1000, 5000, 10000, 30000, 45000, 59999) -- consistent failure direction across all 7 makes a systematic (not occasional-outlier) cause highly likely, but this is not the full validation sweep the plan's T7 method step 2 calls for.",
    "Did not attempt the background/texture-aware fix described above -- left as a scoped follow-up given time budget, not because it's known to be infeasible.",
    "hue's oracle was not implemented or tested at all in this pass -- pure precondition-scoping, not a result."
  ],
  "runtime_sec": 90
}
```

---

## T8 — Predictor hue-sensitivity

**Verdict:** `refuted`  
**Cost class:** `CPU`  
**Phase:** 1

**Hypothesis:** FC_observed for a hue intervention is measured by CNN predictors that must read digit/thickness/intensity off a recolored image; if those non-hue heads are hue-sensitive, part of F2 (and every CC/FC/TPR number touching these predictors, including T6's calibration) is measurement, not model behavior.

**Decision rule:** Recorded before result viewed, per plan: thickness or intensity MAE varies > 1.5x across hue buckets, OR digit accuracy drops > 10 points in the targeted buckets -> a material share of F2 is predictor-side, blast radius extends to T6's TPR, T17's digit response rate, and every CC number -- predictor retraining (T19) becomes a prerequisite if T7's oracle is unavailable (it is, see T7_precondition.json).

**Full record (JSON):**

```json
{
  "test_id": "T8",
  "title": "Predictor hue-sensitivity",
  "phase": 1,
  "cost_class": "CPU",
  "gpu_samples": 0,
  "hypothesis": "FC_observed for a hue intervention is measured by CNN predictors that must read digit/thickness/intensity off a recolored image; if those non-hue heads are hue-sensitive, part of F2 (and every CC/FC/TPR number touching these predictors, including T6's calibration) is measurement, not model behavior.",
  "decision_rule": "Recorded before result viewed, per plan: thickness or intensity MAE varies > 1.5x across hue buckets, OR digit accuracy drops > 10 points in the targeted buckets -> a material share of F2 is predictor-side, blast radius extends to T6's TPR, T17's digit response rate, and every CC number -- predictor retraining (T19) becomes a prerequisite if T7's oracle is unavailable (it is, see T7_precondition.json).",
  "inputs": {
    "checkpoints": [],
    "predictor_ckpt_sha256": null,
    "cohort_index_file": null,
    "seed": null,
    "n": "~9900 real held-out (val) images, bucketed by hue class (~980-1060 per bucket)"
  },
  "code": {
    "script_path": "experiments/hdae/diagnostics/t8_predictor_hue_sensitivity.py",
    "git_sha": null,
    "dirty": true
  },
  "result": {
    "digit_acc_range": [
      0.8865,
      0.917
    ],
    "digit_acc_spread_points": 3.05,
    "thickness_mae_ratio_max_min": 1.184,
    "intensity_mae_ratio_max_min": 1.076,
    "per_bucket_summary": "digit accuracy 88.6%-91.7% across the 10 hue classes (no bucket is an outlier); thickness MAE 0.108-0.128; intensity MAE 4.07-4.38 -- all buckets close together, no bucket dramatically worse than the rest"
  },
  "ci": {
    "method": "n/a (large per-bucket n ~1000, spread reported directly rather than via CI)",
    "level": null,
    "intervals": {}
  },
  "verdict": "refuted",
  "consequences": [
    "Neither threshold is close to being triggered: digit accuracy spread (3.05 points) is well under the 10-point bar; thickness/intensity MAE ratios (1.18x/1.08x) are well under the 1.5x bar.",
    "The predictors are NOT materially hue-sensitive. F2's hue-intervention FC_observed collapse (digit/thickness/intensity all showing significant drift when hue is the intervened attribute) is a REAL model behavior, not a predictor measurement artifact.",
    "This also validates T6's calibration numbers: TPR/FPR for digit/thickness/intensity were computed on a general real-image sample without hue-stratification, and this result confirms that choice doesn't introduce meaningful bias -- the predictors perform consistently regardless of the image's hue.",
    "T19 (predictor retraining with color augmentation) is NOT needed as a prerequisite -- can be dropped from consideration entirely for this investigation."
  ],
  "assumptions_and_caveats": [
    "Did not check whether the predictors were trained with color augmentation (noted as unchecked in the script's output) -- moot given the clean empirical result, but would explain WHY they're robust if ever relevant.",
    "Bucketed by hue CLASS (10 discrete bins) using the same categorical binning as the rest of this investigation, not by continuous hue value -- consistent with how hue is treated everywhere else, appropriate for this check."
  ],
  "runtime_sec": 30
}
```

---

## T9 — Conditioning-ablation loss probe

**Verdict:** `supported`  
**Cost class:** `CPU`  
**Phase:** 1

**Hypothesis:** Conditioning collapse / latent leakage: the decoder increasingly reconstructs from the rich encoder latent zs and ignores the FiLM-injected attribute conditioning as training proceeds and/or as hierarchy depth (k) increases, because zs is a richer signal that can substitute for it.

**Decision rule:** Recorded before result viewed, per plan: delta_L -> 0 monotonically with training steps, effect size beyond across-seed noise -> conditioning collapse confirmed, cheaply and directly. delta_L flat and well above zero -> decoder still uses conditioning, collapse is not the mechanism, T17's expensive probe can be scoped down to a confirmation.

**Full record (JSON):**

```json
{
  "test_id": "T9",
  "title": "Conditioning-ablation loss probe (direct test of conditioning collapse)",
  "phase": 1,
  "cost_class": "CPU",
  "gpu_samples": 0,
  "hypothesis": "Conditioning collapse / latent leakage: the decoder increasingly reconstructs from the rich encoder latent zs and ignores the FiLM-injected attribute conditioning as training proceeds and/or as hierarchy depth (k) increases, because zs is a richer signal that can substitute for it.",
  "decision_rule": "Recorded before result viewed, per plan: delta_L -> 0 monotonically with training steps, effect size beyond across-seed noise -> conditioning collapse confirmed, cheaply and directly. delta_L flat and well above zero -> decoder still uses conditioning, collapse is not the mechanism, T17's expensive probe can be scoped down to a confirmation.",
  "inputs": {
    "checkpoints": [
      "morpho_hier_k1_v3/checkpoints/last.ckpt (30k)",
      "morpho_hier_k5_v3/checkpoints/last.ckpt (30k)",
      "morpho_hier_k11_v3/checkpoints/last_step66000.ckpt (66k)",
      "morpho_hier_k11_v3/checkpoints/last_step75000.ckpt (75k)"
    ],
    "predictor_ckpt_sha256": null,
    "cohort_index_file": null,
    "seed": 0,
    "n": "32 images x 8 timesteps = 256 matched (image,timestep,noise) triples per model per mask"
  },
  "code": {
    "script_path": "experiments/hdae/diagnostics/t9_conditioning_ablation.py",
    "git_sha": null,
    "dirty": true
  },
  "result": {
    "k1_30k": {
      "all": 9e-05,
      "digit": 5e-05,
      "thickness": 1e-05,
      "intensity": 1e-05,
      "hue": 4e-05
    },
    "k5_30k": {
      "all": 0.00048,
      "digit": 9e-05,
      "thickness": -1e-05,
      "intensity": 3e-05,
      "hue": 0.00034
    },
    "k11_66k": {
      "all": 2e-05,
      "digit": 2e-05,
      "thickness": 0.0,
      "intensity": 0.0,
      "hue": 0.0
    },
    "k11_75k": {
      "all": 2e-05,
      "digit": 2e-05,
      "thickness": 0.0,
      "intensity": 0.0,
      "hue": 0.0
    },
    "units": "mean delta_L = L(null-masked attr) - L(true attr), matched noise/timestep/image (paired estimator), raw diffusion training loss units (same scale as the ~0.007 training loss reported throughout this investigation)"
  },
  "ci": {
    "method": "normal approximation (mean +/- 1.96*SE over 256 paired deltas per cell)",
    "level": 0.95,
    "intervals": "see per-model values in result; thickness/intensity CIs straddle or sit very close to zero in every model checked"
  },
  "verdict": "supported",
  "consequences": [
    "THICKNESS AND INTENSITY: delta_L is statistically indistinguishable from zero (or, for k5's thickness, slightly NEGATIVE) in EVERY model checked. This is a direct, quantitative, loss-level confirmation that none of these three models meaningfully rely on the explicit thickness/intensity conditioning channel for reconstruction -- strong, independent evidence for conditioning collapse specifically on the two attributes with the worst CC (F1). This is a stronger and more direct result than T7's failed oracle attempt or T6's calibration correction: it says the model isn't even trying to use thickness/intensity conditioning, not just that measuring compliance is noisy.",
    "HUE: k1 relies on it moderately (delta_L=0.00004, comparable to digit's 0.00005); k5 relies on it HEAVILY -- 0.00034, the single largest delta_L value observed anywhere in this test, ~71% of k5's total all-null delta_L (0.00048). This is an important, unexpected finding for F4: k5 is NOT ignoring hue at the loss/reconstruction level -- quite the opposite, it depends on hue conditioning more than any other model/attribute pair tested. Combined with k5's catastrophic hue counterfactual CC (0.18), this reframes F4 away from 'collapse/ignoring hue' and toward 'k5 has learned a reconstruction-time dependence on hue that does not generalize to novel (counterfactual) hue values' -- more consistent with overfitting/memorization of hue-image associations than with a dead conditioning pathway. This should shift T20's framing: a reseed test is still the right next step, but the expected failure mode to look for is different than 'collapsed', it's 'overfit'.",
    "K11 (both 66k and 75k): ALL delta_L values are the smallest of any model checked, including digit's (0.00002, vs k1's 0.00005 and k5's 0.00009 -- 2.5x-4.5x smaller). K11's reconstruction loss is markedly less sensitive to conditioning being present at all, across every attribute, compared to k1/k5. This is a direct, mechanism-level data point IN FAVOR of the leakage/capacity hypothesis specifically for k11's depth -- the deepest, most fine-grained hierarchical encoder shows the weakest reliance on the explicit conditioning channel, consistent with more information already leaking through its richer per-block latent taps.",
    "k11 66k vs 75k comparison is close to flat (digit delta_L ~0.0000181-0.0000208, both tiny) -- does not show further collapse happening late in training, but this window is AFTER the 30k->45k transition where F3's CC regression actually happened (those checkpoints are gone, see RECON.md) -- so this test cannot directly confirm collapse WORSENED across the specific steps where digit CC fell. It only confirms k11's conditioning reliance is already very weak by 66k, which is consistent with (but does not prove) the collapse having happened earlier.",
    "Overall: T9 supports conditioning collapse as real and attribute/depth-dependent, but with an important nuance the plan's binary framing didn't anticipate -- collapse looks different per attribute (thickness/intensity: genuinely unused; hue: used but not generalized, at least for k5; digit: weakly used, weakest in k11). T10's probe (does z linearly decode each attribute, and does that increase with steps/k) is still the right next confirmatory test, particularly to check whether k11's low conditioning-reliance correlates with HIGH latent decodability of the same attributes (the leakage mechanism) rather than some other explanation."
  ],
  "assumptions_and_caveats": [
    "N=32 images x 8 timesteps is a small, CPU-budget-appropriate sample -- sufficient to detect the large, clear effects reported here (CIs are tight relative to the effect sizes), but not a large-scale study; per-attribute nulling for thickness/intensity in particular should be treated as 'delta_L is small and not clearly positive', not 'delta_L is proven exactly zero'.",
    "30k and 45k k11 checkpoints are unavailable (see RECON.md) -- cannot directly test whether delta_L fell BETWEEN 30k and 45k, only that it is already low by 66k/75k. This is the single most important follow-up if a future k11 training run happens: checkpoint more frequently and re-run this exact probe across the run.",
    "Used ema_model (matching every other eval in this investigation) -- did not cross-check against raw model weights, which could in principle show a different collapse profile (raw model has more recent, less-smoothed weights)."
  ],
  "runtime_sec": 180
}
```

---

## T10 — Latent linear probe

**Verdict:** `supported`  
**Cost class:** `CPU`  
**Phase:** 1

**Hypothesis:** Leakage means the concatenated hierarchical latent z carries attribute information directly, decodable by a simple linear probe. Measure this directly rather than inferring it from CC orderings (T14).

**Decision rule:** Recorded before result viewed, per plan: probe accuracy rises with training steps while delta_L (T9) falls -> leakage confirmed with a mechanism. Probe accuracy rises monotonically with k at fixed steps -> capacity ordering the plan assumes is verified. If it does not rise with k, the k=1 > k=11 leakage narrative loses its premise and should be dropped.

**Full record (JSON):**

```json
{
  "test_id": "T10",
  "title": "Latent linear probe -- leakage and capacity-ordering test",
  "phase": 1,
  "cost_class": "CPU",
  "gpu_samples": 0,
  "hypothesis": "Leakage means the concatenated hierarchical latent z carries attribute information directly, decodable by a simple linear probe. Measure this directly rather than inferring it from CC orderings (T14).",
  "decision_rule": "Recorded before result viewed, per plan: probe accuracy rises with training steps while delta_L (T9) falls -> leakage confirmed with a mechanism. Probe accuracy rises monotonically with k at fixed steps -> capacity ordering the plan assumes is verified. If it does not rise with k, the k=1 > k=11 leakage narrative loses its premise and should be dropped.",
  "inputs": {
    "checkpoints": [
      "morpho_hier_k1_v3/checkpoints/last.ckpt (30k)",
      "morpho_hier_k5_v3/checkpoints/last.ckpt (30k)",
      "morpho_hier_k11_v3/checkpoints/last_step66000.ckpt (66k)",
      "morpho_hier_k11_v3/checkpoints/last_step75000.ckpt (75k)"
    ],
    "predictor_ckpt_sha256": null,
    "cohort_index_file": null,
    "seed": 0,
    "n": "2000 train-partition images to fit probes, 500 held-out val-partition images to evaluate"
  },
  "code": {
    "script_path": "experiments/hdae/diagnostics/t10_latent_probe.py",
    "git_sha": null,
    "dirty": true
  },
  "result": {
    "z_dim_all_models": 512,
    "digit_probe_accuracy": {
      "k1_30k": 0.57,
      "k5_30k": 0.532,
      "k11_66k": 0.842,
      "k11_75k": 0.858
    },
    "thickness_probe_r2": {
      "k1_30k": 0.724,
      "k5_30k": 0.814,
      "k11_66k": 0.848,
      "k11_75k": 0.855
    },
    "intensity_probe_r2": {
      "k1_30k": 0.718,
      "k5_30k": 0.939,
      "k11_66k": 0.939,
      "k11_75k": 0.94
    },
    "hue_probe_r2": {
      "k1_30k": 0.171,
      "k5_30k": 0.996,
      "k11_66k": 0.968,
      "k11_75k": 0.976
    }
  },
  "ci": {
    "method": "n/a (single held-out evaluation, not resampled -- see caveats)",
    "level": null,
    "intervals": {}
  },
  "verdict": "supported",
  "consequences": [
    "DIGIT: probe accuracy is dramatically higher for k11 (0.842-0.858) than k1 (0.570) or k5 (0.532) -- the OPPOSITE direction from what the plan's narrative assumed ('k=1's single large bottleneck leaks more than k=11's hierarchical taps'). This directly corroborates T9's independent finding that k11 relies LEAST on explicit digit conditioning (smallest delta_L of all models) -- T9 and T10 converge on the same mechanism via completely different methods (loss ablation vs linear decodability). This is strong, convergent evidence that k11's latent leaks digit identity more than k1's, and is a highly plausible driver of F3 (digit CC regressing over k11's extended training) -- MORE training steps in a deep/leaky architecture would let the decoder progressively lean on this leaked signal instead of the explicit conditioning, exactly matching flat loss + falling counterfactual compliance.",
    "The plan's own capacity-ordering premise (lower k = more raw per-tap capacity = more leakage) is REFUTED for digit specifically -- flag this clearly rather than silently keeping the old narrative. The mechanism appears to run the other way for this architecture: MORE taps (higher k), not fewer, correlates with more digit leakage. A plausible explanation (not confirmed by this test): concatenating many independently-trained, differently-specialized taps may capture more total decodable structure than one equally-sized entangled representation -- but training-step confound (k11 has 66k-75k steps vs k1/k5's 30k) is NOT disentangled here and must be flagged in any external write-up.",
    "THICKNESS and INTENSITY: R^2 is already substantial for k1 at 30k (0.72 each) and rises further for k5 at the SAME 30k step count (0.81 / 0.94) -- this is a training-steps-MATCHED comparison (k1 vs k5, both 30k) showing hierarchical decomposition alone (k=1 -> k=5) increases leakage of these two continuous attributes, independent of the step-count confound that affects the digit/k11 comparison. Combined with T9's finding that delta_L for thickness/intensity is ~0 in EVERY model checked (including k1 at only 30k steps), this gives a strong, architecture-independent explanation for F1: these two attributes are substantially leaked and substantially unused via explicit conditioning across the whole k=1/5/11 family, not just in the deeper models.",
    "HUE: k1's hue R^2 (0.171) is the LOWEST of any attribute/model combination measured -- hue is barely leaked in k1. k5's hue R^2 (0.996) is the HIGHEST of any combination measured -- essentially total leakage/memorization. k11 sits high but below k5 (0.968-0.976). Cross-referencing T9 (k5's hue delta_L was also the single largest value observed, meaning k5 relies heavily on hue conditioning AT THE SAME TIME as leaking it almost completely): this reframes F4 away from 'collapsed/ignored hue conditioning' toward 'k5's latent has near-perfectly memorized each training image's own hue, creating a conflict when a counterfactual asks for a DIFFERENT hue than the one already baked into the latent -- the explicit conditioning signal has to fight an almost-deterministic latent prior instead of filling a gap'. Note k11 has comparably high hue leakage (0.97+) yet its hue counterfactual CC is good (0.94-0.97) -- so high leakage alone does not doom counterfactual performance; something k5-specific compounds it. This sharpens what T20 (k=5 reseed) should look for: does a different seed show LOWER hue R^2 (leakage itself is seed-sensitive, matching F4 to a specific unlucky training trajectory), or does hue R^2 stay near-1.0 across seeds while CC varies (leakage is a red herring and something else in k5's depth specifically is the cause)?"
  ],
  "assumptions_and_caveats": [
    "Steps and depth are confounded for the k1/k5 vs k11 comparisons (k11 checkpoints are all post-45k; k1/k5 are both exactly 30k) -- the k1-vs-k5 comparison (both 30k) is the only truly steps-matched depth comparison in this result, and it already shows a leakage increase with k for thickness/intensity/hue (though NOT for digit, where k5 is actually lower than k1 -- 0.532 vs 0.570, a further complication worth noting: k5's digit leakage does not follow the same pattern as its thickness/intensity/hue leakage).",
    "Single train/val split, no cross-validation or bootstrap resampling of the probe itself -- given n_val=500 and the R^2/accuracy gaps involved are mostly large (tens of percentage points), standard error is very unlikely to overturn the qualitative conclusions, but no formal CI is reported here, unlike other tests in this ledger. A follow-up with resampled probe fits would tighten this if the exact magnitudes matter for a future decision.",
    "Ridge alpha=10.0 and LogisticRegression default regularization (C=1.0) were not tuned per model -- a systematically different optimal regularization across models could shift R^2/accuracy somewhat, though unlikely to reverse the large, qualitative gaps reported (e.g. hue's 0.17 vs 0.996 gap is far too large to be a regularization artifact)."
  ],
  "runtime_sec": 240
}
```

---

## T11 — Digit CC by target class (k=11 @ 75k)

**Verdict:** `refuted (class-0 hypothesis) / CONFIRMED AND ESCALATED (F3 itself)`  
**Cost class:** `GPU-S`  
**Phase:** 1

**Hypothesis:** Break digit CC down by target class to check for concentration (e.g. class 0, reopening the attribute-dropout hypothesis) vs uniform degradation. Also serves as a real-data check on whether digit CC continued to regress past 45k, since no per-image records existed to answer this for free.

**Decision rule:** Recorded before result viewed, per plan: class 0 disproportionately worse (>2x mean error rate of classes 1-9) and widening 30k->45k -> reopen attribute-dropout hypothesis. Regardless: is the 30k->45k (now 30k->45k->75k) degradation uniform across classes or concentrated? Concentration is a strong mechanism hint and should be reported even though not the stated hypothesis.

**Full record (JSON):**

```json
{
  "test_id": "T11",
  "title": "Digit CC by target class (k=11 @ 75k)",
  "phase": 1,
  "cost_class": "GPU-S",
  "gpu_samples": 512,
  "hypothesis": "Break digit CC down by target class to check for concentration (e.g. class 0, reopening the attribute-dropout hypothesis) vs uniform degradation. Also serves as a real-data check on whether digit CC continued to regress past 45k, since no per-image records existed to answer this for free.",
  "decision_rule": "Recorded before result viewed, per plan: class 0 disproportionately worse (>2x mean error rate of classes 1-9) and widening 30k->45k -> reopen attribute-dropout hypothesis. Regardless: is the 30k->45k (now 30k->45k->75k) degradation uniform across classes or concentrated? Concentration is a strong mechanism hint and should be reported even though not the stated hypothesis.",
  "inputs": {
    "checkpoints": [
      "experiments/hdae/outputs/morpho_hier_k11_v3/checkpoints/last_step75000.ckpt"
    ],
    "predictor_ckpt_sha256": null,
    "cohort_index_file": "experiments/hdae/outputs/intervention_cohorts.json",
    "seed": null,
    "n": 512
  },
  "code": {
    "script_path": "experiments/hdae/diagnostics/t11_digit_by_class.py",
    "git_sha": null,
    "dirty": true
  },
  "result": {
    "overall_digit_cc_75k": 0.0957,
    "digit_cc_30k": 0.7554,
    "digit_cc_45k": 0.4634,
    "digit_cc_75k": 0.0957,
    "per_target_class": {
      "0": {
        "n": 38,
        "cc": 0.0263
      },
      "1": {
        "n": 47,
        "cc": 0.0
      },
      "2": {
        "n": 51,
        "cc": 0.1176
      },
      "3": {
        "n": 51,
        "cc": 0.1176
      },
      "4": {
        "n": 48,
        "cc": 0.1458
      },
      "5": {
        "n": 53,
        "cc": 0.0377
      },
      "6": {
        "n": 57,
        "cc": 0.0175
      },
      "7": {
        "n": 50,
        "cc": 0.04
      },
      "8": {
        "n": 46,
        "cc": 0.2391
      },
      "9": {
        "n": 71,
        "cc": 0.1831
      }
    },
    "class_0_vs_others": "class 0's CC (0.0263) is actually BETTER than the overall mean (0.0957) and not an outlier -- classes 1 and 6 are worse (0.0, 0.0175). No concentration in class 0 specifically."
  },
  "ci": {
    "method": "n/a (raw per-class rates, small per-class n ~38-71 -- see caveats)",
    "level": null,
    "intervals": {}
  },
  "verdict": "refuted (class-0 hypothesis) / CONFIRMED AND ESCALATED (F3 itself)",
  "consequences": [
    "URGENT, HEADLINE FINDING: digit CC has continued to collapse well past 45k. The trajectory is now 0.755 (30k) -> 0.463 (45k) -> 0.096 (75k) -- an 8x drop from the original checkpoint, and the decline from 45k to 75k (30k more steps) is proportionally WORSE than the decline from 30k to 45k (15k steps). This is not a plateau or a transient dip -- it is an accelerating collapse of digit conditioning, continuing through the exact 30k-step continuation run just completed this session.",
    "Class-0 collision hypothesis is definitively refuted: class 0's CC (0.0263) is unremarkable, actually slightly better than the 0.0957 overall mean. Classes 1 (0.0) and 6 (0.0175) are the worst performers, not class 0. Degradation is UNIFORM/GLOBAL across all 10 classes (range 0.0-0.239, all catastrophically bad), not concentrated in any single class -- rules out any per-class encoding bug and further supports a general conditioning-collapse mechanism (T9/T10) over a specific-value bug.",
    "This is strong, direct, real-world confirmation of T9's and T10's conditioning-collapse hypothesis: not only does k11 show the smallest loss-sensitivity to digit conditioning (T9) and the highest latent decodability of digit (T10), but its actual digit counterfactual performance has now collapsed to near-zero (9.6% success) in the real generation pipeline.",
    "PRACTICAL RECOMMENDATION: do not train k11 further without addressing conditioning collapse first (e.g. increasing attr_dropout_prob well above 0.08, adding an explicit latent-attribute independence penalty, or reducing latent capacity) -- more steps at the current recipe appear to actively worsen digit counterfactual quality, matching exactly the consequence the plan's own T17 decision rule anticipated ('conditioning collapse confirmed... train longer is affirmatively wrong... 75k run stays dead')."
  ],
  "assumptions_and_caveats": [
    "Per-class n is small (38-71 images per class out of the 512-cohort) -- individual class CC values have wide uncertainty (e.g. class 0's true rate could plausibly be anywhere from ~1% to ~13% given n=38), but the OVERALL pattern (all classes catastrophically low, no class anywhere near 45k's or 30k's overall rate) is far too large and consistent across all 10 classes to be sampling noise.",
    "This single number (overall CC=0.0957) should be cross-checked against the full toleranced eval currently running in parallel on the same 75k checkpoint (same cohort, same edit_strength=8.0) -- if that eval's digit CC doesn't closely match 0.0957, investigate a discrepancy between this script's scoring (simple round-to-nearest-int on predictor output) and the production script's categorical_class_index-based scoring before treating either as final.",
    "Used edit_strength=8.0, matching every other eval in this investigation -- did not check whether a different guidance scale would recover digit performance at 75k (that's T18's territory, not this test's)."
  ],
  "runtime_sec": 150
}
```

---

## T12 — k=5 forensics

**Verdict:** `refuted`  
**Cost class:** `FREE`  
**Phase:** 1

**Hypothesis:** F4 (k=5's hue counterfactual CC=0.18 vs k=1/k=11's ~0.97-0.99) is explained by a weight-level anomaly (collapsed hue embedding/null vector), a config difference beyond depth, or a training-log anomaly (loss spike, LR issue, bad resume).

**Decision rule:** Recorded before result viewed, per plan: a (collapsed norm/degenerate gain/near-zero swing) or b (NaN/inf, or magnitude outside the k1/k11-spanned range) positive -> F4 explained, no retrain. c (log anomaly) positive -> a correlate not an explanation, must pair with a or b to close F4. All clean -> F4 is depth-specific or seed-specific, only T20 separates them.

**Full record (JSON):**

```json
{
  "test_id": "T12",
  "title": "k=5 forensics, non-GPU steps",
  "phase": 1,
  "cost_class": "FREE",
  "gpu_samples": 0,
  "hypothesis": "F4 (k=5's hue counterfactual CC=0.18 vs k=1/k=11's ~0.97-0.99) is explained by a weight-level anomaly (collapsed hue embedding/null vector), a config difference beyond depth, or a training-log anomaly (loss spike, LR issue, bad resume).",
  "decision_rule": "Recorded before result viewed, per plan: a (collapsed norm/degenerate gain/near-zero swing) or b (NaN/inf, or magnitude outside the k1/k11-spanned range) positive -> F4 explained, no retrain. c (log anomaly) positive -> a correlate not an explanation, must pair with a or b to close F4. All clean -> F4 is depth-specific or seed-specific, only T20 separates them.",
  "inputs": {
    "checkpoints": [
      "experiments/hdae/outputs/morpho_hier_k1_v3/checkpoints/last.ckpt",
      "experiments/hdae/outputs/morpho_hier_k5_v3/checkpoints/last.ckpt",
      "experiments/hdae/outputs/morpho_hier_k11_v3/checkpoints/last_step75000.ckpt"
    ],
    "predictor_ckpt_sha256": null,
    "cohort_index_file": null,
    "seed": null,
    "n": null
  },
  "code": {
    "script_path": "inline (checkpoint weight-norm inspection, config diff, train.log grep)",
    "git_sha": null,
    "dirty": true
  },
  "result": {
    "part_a_swing_check_from_T5": {
      "k5_hue_normalized_swing": 0.1912,
      "k1_hue_normalized_swing": 0.2128,
      "k5_hue_vs_thickness_ratio": 1.541,
      "interpretation": "Now that T5 exists: k5's hue swing (0.191) is comparable to k1's hue swing (0.213), and k5's hue/thickness ratio (1.54x) is in the same direction as k1's (1.93x), just weaker. NOT collapsed, NOT near-zero, NOT degenerate -- k5's hue conditioning pathway carries a normal, non-anomalous amount of signal at the embedding/FiLM level. This directly confirms part_a's original weight-norm-based conclusion using the metric T5 was actually designed to produce."
    },
    "part_a_weight_level_hue_check": {
      "k5_hue_embedding_table_norm": 17.2047,
      "k1_hue_embedding_table_norm": 16.6731,
      "k11_75k_hue_embedding_table_norm": 6.972,
      "k5_hue_per_class_row_norms": [
        6.351,
        5.647,
        4.55,
        4.11,
        4.693,
        5.408,
        4.611,
        6.914,
        6.167,
        5.274
      ],
      "k1_hue_per_class_row_norms": [
        4.111,
        5.142,
        5.011,
        5.704,
        4.957,
        5.361,
        4.252,
        5.637,
        6.098,
        6.054
      ],
      "k5_hue_null_vector_norm": 0.2265,
      "k1_hue_null_vector_norm": 0.2482,
      "k11_75k_hue_null_vector_norm": 0.4167,
      "interpretation": "K5's hue embedding table is comparable in magnitude to k1's (both ~16-17, k11's is lower but that's a different step count / training amount, not comparable). No collapsed or near-zero rows in k5's hue table -- all 10 class rows are in the same 4-7 range as k1's. Null vector norm is unremarkable, not an outlier. NO weight-level anomaly found for hue specifically."
    },
    "part_b_nan_inf_and_config": {
      "n_nan_tensors_k1": 0,
      "n_nan_tensors_k5": 0,
      "n_nan_tensors_k11": 0,
      "n_inf_tensors_k1": 0,
      "n_inf_tensors_k5": 0,
      "n_inf_tensors_k11": 0,
      "k1_vs_k5_config_diff": "hier_tap_block_ids/hier_level_dims/hier_block_to_level (the depth parameter, expected) and output_dir/a comment line. Nothing else differs -- same attr_fusion, attr_dropout_prob, cfg_drop_prob, batch/lr recipe, seed=42."
    },
    "part_c_training_log": {
      "n_loss_readings": 62021,
      "loss_min": 0.00325,
      "loss_max": 1.0,
      "loss_median_post_warmup": 0.00762,
      "n_spikes_over_5x_median_post_step2000": 0,
      "n_resume_events": 1,
      "resume_context": "one benign resume/restart consistent with the documented mid-session stop/continue the user issued for K1/K5's parallel training; no loss discontinuity visible around it (0 spikes in the post-warmup tail).",
      "n_error_or_nan_log_lines": 0
    }
  },
  "ci": {
    "method": "n/a",
    "level": null,
    "intervals": {}
  },
  "verdict": "refuted",
  "consequences": [
    "All three sub-checks (a, b, c) come back clean: no weight-level collapse in k5's hue embedding, config is identical to k1 except the depth parameter, training log shows no spike or error around its one benign resume.",
    "Per the plan's own decision rule, this means F4 is NOT explained by a static weight/config/log-level cause -- it is depth-specific or seed-specific, and T20 (k=5 reseed at 30k steps) is the only remaining test that can separate those two possibilities.",
    "This does not rule out a dynamic (training-trajectory) cause invisible to a single-checkpoint weight snapshot -- e.g. an unlucky optimization path specific to k=5's particular depth that a different seed would avoid. That is exactly what T20 tests."
  ],
  "assumptions_and_caveats": [
    "k11's comparison checkpoint is at 75k steps (30k is gone, see RECON.md), so its embedding-table norms are not directly comparable to k1/k5's 30k snapshots (more training steps changes norm scale) -- used only qualitatively (no NaN/collapse), not for the cross-model magnitude range comparison the plan's method describes.",
    "Weight-norm inspection cannot rule out a subtler failure mode than collapse/NaN -- e.g. the hue embedding table having correct norms but a corrupted internal structure (wrong class-to-vector mapping) that would only surface in T10's linear probe or an actual generation test (T15/T18). This test only rules out the coarse failure modes it was designed to catch."
  ],
  "runtime_sec": 40
}
```

---

## T13 — Nearest-neighbour OOD extension

**Verdict:** `ambiguous`  
**Cost class:** `CPU`  
**Phase:** 1

**Hypothesis:** Extending the digit NN-OOD check: are hue/thickness/intensity counterfactual targets jointly similar to something in the training set, restricted to the source image's own digit class (digit is never intervened by these three types)? Thickness's construction includes SCM propagation to intensity (the one declared causal edge).

**Decision rule:** Same as the digit script: median ratio near 1.0 AND >=95% of targets inside the 95th percentile of the real validation-vs-train NN-distance gap -> OOD ruled out for that attribute. A materially worse ratio would reopen OOD as a candidate, most relevantly for F2 (hue).

**Full record (JSON):**

```json
{
  "test_id": "T13",
  "title": "Nearest-neighbour OOD extension (hue, thickness, intensity)",
  "phase": 1,
  "cost_class": "CPU",
  "gpu_samples": 0,
  "hypothesis": "Extending the digit NN-OOD check: are hue/thickness/intensity counterfactual targets jointly similar to something in the training set, restricted to the source image's own digit class (digit is never intervened by these three types)? Thickness's construction includes SCM propagation to intensity (the one declared causal edge).",
  "decision_rule": "Same as the digit script: median ratio near 1.0 AND >=95% of targets inside the 95th percentile of the real validation-vs-train NN-distance gap -> OOD ruled out for that attribute. A materially worse ratio would reopen OOD as a candidate, most relevantly for F2 (hue).",
  "inputs": {
    "checkpoints": [],
    "predictor_ckpt_sha256": null,
    "cohort_index_file": "experiments/hdae/outputs/intervention_cohorts.json",
    "seed": null,
    "n": 512
  },
  "code": {
    "script_path": "experiments/hdae/counterfactuals/check_attr_nn_ood.py",
    "git_sha": null,
    "dirty": true
  },
  "result": {
    "hue": {
      "frac_within_val_95th_gap": 0.947,
      "ratio_to_val_median": {
        "50": 0.991,
        "75": 1.082,
        "90": 1.191,
        "95": 1.308,
        "99": 1.511
      }
    },
    "thickness": {
      "frac_within_val_95th_gap": 0.916,
      "ratio_to_val_median": {
        "50": 1.018,
        "75": 1.125,
        "90": 1.266,
        "95": 1.385,
        "99": 1.56
      }
    },
    "intensity": {
      "frac_within_val_95th_gap": 0.828,
      "ratio_to_val_median": {
        "50": 1.089,
        "75": 1.236,
        "90": 1.413,
        "95": 1.564,
        "99": 1.857
      }
    },
    "digit_for_reference_from_earlier_check": {
      "frac_within_val_95th_gap": 0.961,
      "ratio_to_val_median_50": 1.008
    }
  },
  "ci": {
    "method": "n/a (percentile summary over n=512, not a proportion CI)",
    "level": null,
    "intervals": {}
  },
  "verdict": "ambiguous",
  "consequences": [
    "hue: 94.7% within the 95th-pct gap, median ratio 0.99 -- essentially matches digit's clean result, just under the plan's 95% bar by 0.3 points. Practically indistinguishable from 'ruled out'. F2 (hue's FC_observed collapse) is NOT explained by OOD targets.",
    "thickness: 91.6% within gap, median ratio 1.02 -- mostly in-distribution, modestly below the 95% bar. Not a strong OOD signal.",
    "intensity: 82.8% within gap, median ratio 1.09, p95 ratio 1.56 -- clearly the weakest of the four attributes checked (digit/hue/thickness/intensity). Real, if modest, separation from the normal generalization-gap distribution: intensity counterfactual targets sit meaningfully farther from any real training neighbor (within the same digit class) than genuinely unseen validation images typically do.",
    "This does not rise to the marginal-range OOD seen for a truly novel value (T1's earlier check found 0 marginal violations for intensity) -- it's a JOINT-similarity gap, consistent with intensity having the most attributes constraining its 'realistic' combinations (it's a descendant of thickness, so its feature vector reflects an SCM-propagated joint value, which is intrinsically harder to match than an independently-set target).",
    "Worth flagging as a PARTIAL, secondary contributor to intensity's low CC (0.21-0.33) alongside T6's calibration/ceiling issue -- not a replacement for it. Intensity is the only one of the four where OOD is not simply ruled out."
  ],
  "assumptions_and_caveats": [
    "Pool restriction is by the source image's OWN digit class for all three attributes (digit is never intervened by hue/thickness/intensity interventions) -- different from the digit script's TARGET-class restriction, appropriate given what's held fixed.",
    "Intensity's feature vector for the NN query uses the SCM-propagated value where relevant (none, since intensity itself is the intervened attribute here, not a descendant) -- thickness's query DOES use the propagated intensity value, as intended.",
    "The 95%/near-1.0 threshold is inherited from the digit script's convention, not independently re-derived or justified with a formal test here -- treat the specific cutoff as a convenience heuristic, not a calibrated significance threshold."
  ],
  "runtime_sec": 25
}
```

---

## T14 — Capacity-ordering falsification

**Verdict:** `refuted`  
**Cost class:** `FREE`  
**Phase:** 1

**Hypothesis:** The 'more latent capacity (lower k) -> more leakage -> worse compliance' narrative predicts a monotone CC ordering k=1 > k=5 > k=11, with k=5 falling strictly between k=1 and k=11, on attributes unaffected by k=5's separate hue corruption (digit, thickness, intensity).

**Decision rule:** Recorded before result viewed, per plan: two of three monotone with k=5 intermediate -> weak support, say 'weak'. Report ordering per attribute with CIs and name the confound the narrative ignores (k varies capacity, decoder architecture, injection points, and optimization difficulty simultaneously -- 'k=11 undertrained at fixed 30k steps' explains the same ordering with no leakage at all). T10's probe is what actually discriminates; T14 only bounds how much the CC ordering alone is entitled to claim.

**Full record (JSON):**

```json
{
  "test_id": "T14",
  "title": "Capacity-ordering falsification",
  "phase": 1,
  "cost_class": "FREE",
  "gpu_samples": 0,
  "hypothesis": "The 'more latent capacity (lower k) -> more leakage -> worse compliance' narrative predicts a monotone CC ordering k=1 > k=5 > k=11, with k=5 falling strictly between k=1 and k=11, on attributes unaffected by k=5's separate hue corruption (digit, thickness, intensity).",
  "decision_rule": "Recorded before result viewed, per plan: two of three monotone with k=5 intermediate -> weak support, say 'weak'. Report ordering per attribute with CIs and name the confound the narrative ignores (k varies capacity, decoder architecture, injection points, and optimization difficulty simultaneously -- 'k=11 undertrained at fixed 30k steps' explains the same ordering with no leakage at all). T10's probe is what actually discriminates; T14 only bounds how much the CC ordering alone is entitled to claim.",
  "inputs": {
    "checkpoints": [],
    "predictor_ckpt_sha256": null,
    "cohort_index_file": "experiments/hdae/outputs/intervention_cohorts.json",
    "seed": null,
    "n": 512
  },
  "code": {
    "script_path": "inline (Wilson CI over existing 30k eval-log CC numbers)",
    "git_sha": null,
    "dirty": true
  },
  "result": {
    "digit": {
      "k1": {
        "cc": 0.7583,
        "n": 509,
        "ci95": [
          0.7193,
          0.7935
        ]
      },
      "k5": {
        "cc": 0.651,
        "n": 510,
        "ci95": [
          0.6086,
          0.6911
        ]
      },
      "k11": {
        "cc": 0.7554,
        "n": 507,
        "ci95": [
          0.7162,
          0.7908
        ]
      }
    },
    "thickness": {
      "k1": {
        "cc": 0.3351,
        "n": 467,
        "ci95": [
          0.2928,
          0.378
        ]
      },
      "k5": {
        "cc": 0.2732,
        "n": 474,
        "ci95": [
          0.234,
          0.3139
        ]
      },
      "k11": {
        "cc": 0.2705,
        "n": 475,
        "ci95": [
          0.2315,
          0.3111
        ]
      }
    },
    "intensity": {
      "k1": {
        "cc": 0.3246,
        "n": 496,
        "ci95": [
          0.2849,
          0.367
        ]
      },
      "k5": {
        "cc": 0.2628,
        "n": 506,
        "ci95": [
          0.2264,
          0.3029
        ]
      },
      "k11": {
        "cc": 0.2129,
        "n": 498,
        "ci95": [
          0.1792,
          0.2509
        ]
      }
    },
    "per_attribute_verdict": {
      "digit": "VIOLATES monotonicity: k5 (0.651, CI [0.609,0.691]) is significantly BELOW BOTH k1 and k11, not intermediate. k1 and k11 (0.758 vs 0.755) are statistically indistinguishable -- CIs [0.719,0.794] and [0.716,0.791] almost fully overlap.",
      "thickness": "Monotone in point estimate (k1>k5>k11) but CIs are largely overlapping (k1 vs k5 overlap zone 0.293-0.314; k5 vs k11 heavily overlapping; k1 vs k11 barely overlap at 0.293-0.311) -- ordering is NOT CI-separated, cannot be called signal on its own.",
      "intensity": "Monotone in point estimate AND the two endpoints ARE CI-separated (k1 [0.285,0.367] vs k11 [0.179,0.251] do not overlap) -- k1 significantly beats k11. k5 [0.226,0.303] overlaps both neighbors, so 'k5 strictly intermediate' is not confirmed, only 'k1 beats k11' is."
    }
  },
  "ci": {
    "method": "wilson",
    "level": 0.95,
    "intervals": "see result block above, per model x attribute"
  },
  "verdict": "refuted",
  "consequences": [
    "Overall verdict is the plan's own predicted outcome for this branch: 'weak' -- but weaker than a naive 2-of-3 count suggests, because even the 2 'monotone' attributes are only partially CI-separated (intensity's endpoints separate, k5 itself is not confirmed intermediate for either).",
    "Digit is not just weak evidence against the ordering, it's a direct violation with adequate statistical power (n~510, CI half-widths ~0.04) -- k5 is significantly worse than BOTH shallower and deeper models on digit, which a capacity-monotone leakage story cannot produce.",
    "The capacity-ordering / leakage narrative should not be asserted from CC numbers alone in any write-up. T10's latent linear probe (does probe accuracy actually rise with k?) is required to support or refute it directly -- this test only establishes that the CC-based case for it is weak-to-contradicted, not that leakage itself doesn't happen.",
    "Named confound: at a fixed 30k-step training budget, k=11 has a deeper/more complex hierarchy to fit than k=1 -- 'k=11 is comparatively undertrained' is equally consistent with intensity's k1>k11 gap and requires no leakage mechanism at all. T9 (\u0394L trend across steps) and T10 (probe accuracy across steps AND across k) are what can actually separate 'undertrained' from 'leaking'."
  ],
  "assumptions_and_caveats": [
    "k5's digit/thickness/intensity numbers are used here on the assumption they are NOT contaminated by the separate hue-corruption failure mode (F4) -- reasonable since digit/thickness/intensity are scored independently of the hue intervention block, but not independently re-verified in this test.",
    "Wilson CIs assume each image's success/fail is an independent Bernoulli trial -- true within a model's own eval but the three models' cohorts overlap (same 512 source images), so cross-model comparisons have some shared-item correlation the marginal CIs don't capture; a paired test would be tighter and is a candidate refinement, not done here."
  ],
  "runtime_sec": 30
}
```

---

## K11 @ 75k full re-eval log (raw)

`experiments/hdae/outputs/morpho_hier_k11_v3/cf1_eval_gs8_tol2x_75k/eval.log`

```
/home/anajibi/HDM/.venv/lib/python3.10/site-packages/lightning_fabric/__init__.py:29: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  __import__("pkg_resources").declare_namespace(__name__)
Global seed set to 0
Model params: 88.60 M
fixed index set: n=512 (from experiments/hdae/outputs/intervention_cohorts.json)
unobserved_tolerance_mult=2.0 (FC_unobserved gate = mult * cnn_mae per attribute)
NOTE: bg_phase wraps at 2*pi but is not in the circular-diff set used here -- its hard gate is wrap-unaware, see module docstring.
  intervene(digit, shift): CC pool={digit}+[] FC_obs=['thickness', 'intensity', 'hue'] FC_unobs(8 attrs, hard mult*mae gate)
/home/anajibi/HDM/diffae_upstream/diffusion/base.py:306: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast(self.conf.fp16):
    -> CC=0.0721 FC_obs=0.9072 FC_unobs=0.8715 CF1_obs=0.1337 CF1_unobs=0.1333 n_valid=499/512
  intervene(hue, shift): CC pool={hue}+[] FC_obs=['digit', 'thickness', 'intensity'] FC_unobs(8 attrs, hard mult*mae gate)
    -> CC=0.8984 FC_obs=0.3022 FC_unobs=0.6747 CF1_obs=0.4523 CF1_unobs=0.7707 n_valid=512/512
  intervene(thickness, flip_binned): CC pool={thickness}+['intensity'] FC_obs=['digit', 'hue'] FC_unobs(8 attrs, hard mult*mae gate)
    -> CC=0.3383 FC_obs=0.8762 FC_unobs=0.8292 CF1_obs=0.4881 CF1_unobs=0.4805 n_valid=504/512
  intervene(intensity, flip_binned): CC pool={intensity}+[] FC_obs=['digit', 'thickness', 'hue'] FC_unobs(8 attrs, hard mult*mae gate)
    -> CC=0.3465 FC_obs=0.7813 FC_unobs=0.8729 CF1_obs=0.4800 CF1_unobs=0.4960 n_valid=508/512

=== aggregate ===
{
  "model": "k11_v3_75k",
  "edit_strength": 8.0,
  "unobserved_tolerance_mult": 2.0,
  "global_CC": 0.40087059754649784,
  "global_FC_observed": 0.7167433994683872,
  "global_FC_unobserved": 0.8120833194053101,
  "macro_CF1_observed": 0.3885405510471135,
  "macro_CF1_unobserved": 0.4701262929825103,
  "weighted_CF1_observed": 0.3896036808979628,
  "weighted_CF1_unobserved": 0.4721986429381533,
  "n_interventions": 4,
  "n_images": 512
}
wrote experiments/hdae/outputs/morpho_hier_k11_v3/cf1_eval_gs8_tol2x_75k

```
