# HDAE counterfactual quality problem — summary for external review

## Context

We're training Hierarchical Disentangled AutoEncoders (HDAE, a diffusion-autoencoder
variant) on MorphoMNIST++ (64x64 digit images with 14 tracked generative attributes:
digit, thickness, intensity, hue, slant, rotation, scale, translate_x, translate_y,
bg_freq, bg_phase, bg_amplitude, texture_seed, texture_amplitude). The model conditions
on 4 causally-structured attributes (digit, thickness, intensity, hue; thickness→intensity
is the one declared causal edge, rest are roots) via per-block FiLM-style attribute
embeddings injected into a diffusion decoder.

We're comparing three encoder hierarchy depths — **k=1** (single mid-level tap, most
capacity concentrated), **k=5**, **k=11** (finest-grained hierarchical tap) — on their
ability to generate *counterfactual* images: intervene on one attribute (e.g. "change
this digit's hue to X"), keep the rest fixed via the causal graph, and check whether the
generated image (a) actually reflects the new attribute value, (b) leaves everything else
alone. This is scored by a downstream CNN attribute predictor (trained separately, not
part of the generative model) that measures each output image's attributes.

All three models are same recipe otherwise: `attr_fusion: concat_film` (each attribute
gets a protected embedding slice, FiLM-modulated into the style vector — not summed),
`attr_dropout_prob: 0.08` (forces the model to rely on each attribute independently
during training), batch/lr scaled together, EMA, T=1000 train / T=100 eval diffusion
steps, guidance scale (edit strength) = 8.0 for counterfactual generation.

## Metrics (per intervention type, e.g. "intervene on thickness")

- **CC** (Correctness/Compliance): did the intervened attribute actually land at the
  target value, among images where an edit was actually necessary (i.e. excluding images
  already at/near the target)? Continuous attrs use a tolerance window; categorical attrs
  (digit, hue) require exact class match.
- **FC_observed**: did *other modeled-but-not-intervened* attributes (e.g. thickness/
  intensity/hue when digit is the intervention) stay put? Soft, std-normalized score.
- **FC_unobserved**: did the 8 *structurally unmodeled* attributes (rotation, scale,
  translate_x/y, bg_freq/phase/amplitude, texture_amplitude) stay put? Hard gate at
  `mult * predictor's own test-set MAE` for that attribute (we use mult=2).
- **CF1**: harmonic mean of CC and FC (observed/unobserved variants), the overall
  "did the edit succeed cleanly" score.

## The problem: results are bad, and one number is actively getting worse with more training

### 30,000-step checkpoints, all three models, edit_strength=8.0, 512-image eval cohort

| model | intervention | CC | FC_obs | FC_unobs | CF1_obs |
|---|---|---|---|---|---|
| k=1  | digit     | 0.758 | 0.852 | 0.839 | 0.802 |
| k=1  | hue       | 0.990 | 0.626 | 0.742 | 0.767 |
| k=1  | thickness | 0.335 | 0.970 | 0.795 | 0.498 |
| k=1  | intensity | 0.325 | 0.711 | 0.796 | 0.446 |
| k=5  | digit     | 0.651 | 0.847 | 0.767 | 0.736 |
| k=5  | **hue**   | **0.182** | 0.398 | 0.669 | 0.249 |
| k=5  | thickness | 0.273 | 0.976 | 0.803 | 0.427 |
| k=5  | intensity | 0.263 | 0.774 | 0.768 | 0.392 |
| k=11 | digit     | 0.755 | 0.835 | 0.764 | 0.793 |
| k=11 | hue       | 0.971 | 0.293 | 0.585 | 0.450 |
| k=11 | thickness | 0.271 | 0.909 | 0.772 | 0.417 |
| k=11 | intensity | 0.213 | 0.732 | 0.783 | 0.330 |

Global CC (all interventions pooled): k=1 = 0.558, k=5 = 0.330, k=11 = 0.504.

Takeaways at 30k:
- **thickness and intensity CC are bad across all three models** (0.21–0.34), fairly
  uniformly — looks architecture-independent, more like a systemic issue.
- **hue shows an odd trade: CC is near-perfect (0.97–0.99) but FC_observed tanks
  (0.29–0.63)** — i.e. the model *does* successfully change hue, but wrecks
  digit/thickness/intensity while doing it. Opposite pattern from thickness (CC bad,
  FC_obs good — barely edits, but doesn't break anything either).
- **k=5's hue is a separate, worse anomaly** (CC=0.18 vs k=1/k=11's ~0.97–0.99) —
  visually confirmed as genuinely corrupted output, not a metric artifact.

### K11 continued to 45,000 steps, re-evaluated (tolerance methodology also redesigned
between these two evals, but digit/hue's exact-match scoring is mathematically
unaffected by that redesign — thickness/intensity aren't directly comparable step-for-step
here, only digit/hue are apples-to-apples):

| intervention | CC (30k) | CC (45k) | delta |
|---|---|---|---|
| **digit** | **0.755** | **0.463** | **-0.29 (regression)** |
| hue | 0.971 | 0.936 | -0.04 (roughly stable) |

**Digit CC dropped by 29 points with 15,000 more training steps, while training loss
stayed flat at ~0.007 the whole time.** This is the single most concerning number in the
whole investigation — it means "just train longer" is not obviously a fix, and might be
actively hurting the thing we care about.

We're currently mid-way through continuing K11 from 45k → 75k steps to see if this
digit-CC decline continues, plateaus, or reverses. No results yet.

## Investigation so far — what's ruled out, what's confirmed, what's open

**1. Are the counterfactual targets simply out-of-distribution for the training data?**
Hypothesis: maybe the models perform badly because we're asking them to render attribute
combinations they never saw.
- Checked marginal range (per-attribute, is the target's raw value inside
  [train_min, train_max])? **Essentially never** — 1/2048 counterfactual vectors across
  all 4 intervention types have any attribute outside its train range (0.05%).
- Checked joint similarity for digit specifically (digit is a causal-graph root node, so
  a digit intervention leaves every other attribute unchanged — clean, well-posed nearest-
  neighbor query): for each of the 512 digit-intervention targets, find the nearest real
  training image *of the target digit class* in normalized 12-dim attribute space, and
  compare that distance to (a) real training images' typical spacing from each other, and
  (b) real *held-out validation* images' typical distance to their nearest training
  neighbor (the model's normal, already-solved generalization gap).
  Result: **the counterfactual targets' nearest-neighbor distances track the real
  held-out validation distribution almost exactly** (median ratio 1.01; 96% of targets
  fall within the 95th percentile of the real validation-vs-train gap). So digit
  counterfactual targets are no more "novel" than ordinary unseen data the model already
  has to handle.
- **Conclusion: OOD is ruled out as the explanation**, at least for digit. (Not yet
  re-run for thickness/intensity/hue, though thickness/intensity have a causal edge
  between them which makes constructing the "other attributes changed" comparison more
  involved — hue is a root like digit so should be equally tractable if useful.)

**2. Does attribute-dropout (`attr_dropout_prob=0.08`, new in this generation of
configs) collide with digit's raw value encoding?**
Hypothesis: if a "dropped" attribute during training were represented by the literal
value `0.0`, that would be indistinguishable from digit class 0 (digit's raw storage is
integers 0..9) but *not* from any other attribute (none of whose valid ranges include
exactly 0) — which could selectively corrupt digit=0 and get worse the more training
steps accumulate dropout events.
- Read `attr_conditioner.py`'s `ConcatAttributeEmbedding` (the fusion mode actually in
  use, `concat_film`): dropout is implemented as an explicit boolean mask that substitutes
  a dedicated **learned null embedding vector** per attribute, not a raw sentinel value.
  This was deliberately built to avoid exactly this class of bug (its docstring says so).
- **Conclusion: ruled out** by direct code inspection. Not yet double-checked empirically
  (e.g. breaking digit CC down by target class to see if class 0 is disproportionately
  worse) — would be a cheap confirmation if we want extra certainty.

**3. Is thickness's bad CC (~0.21–0.34 across all models) partly a measurement-ceiling
artifact rather than a real model failure?**
Hypothesis: the tolerance window used to score "did thickness land close enough to
target" might be tighter than the CNN attribute predictor's own noise floor, capping CC
below 1.0 no matter how good the model is.
- Thickness predictor's own held-out test-set MAE: **0.115** (thickness units).
- Median tolerance half-width (half the width of the image's target population-quantile
  bin): **0.055**.
- The predictor's own measurement noise (0.115) is **~2x wider** than the tolerance
  window (0.055) for a typical image.
- **Conclusion: confirmed, at least partially** — thickness CC has a real ceiling well
  below 1.0 baked into the eval design, independent of model quality. This does *not*
  explain digit's regression (digit uses exact categorical match, no predictor-noise
  tolerance involved) or hue's FC_observed problem.

**4. Leading open hypothesis: per-attribute edit-strength mismatch.**
Guidance/edit strength is one global knob (8.0) applied identically to all 4 attributes
at eval time. The 30k table shows literally opposite failure modes at the same strength:
hue over-edits (CC great, collateral damage severe) while thickness under-edits (CC bad,
collateral damage minimal). That's the classic signature of a per-attribute operating
point being wrong, not a uniformly bad model — different attributes likely need
different guidance strengths to hit a good CC/FC tradeoff.
- **Not yet tested.** Proposed check: sweep edit_strength over ~{2, 4, 6, 8, 12} per
  attribute and plot CC vs FC_observed, using the existing eval scripts (no retraining
  needed). This is considered the single highest-value next diagnostic, but it requires
  real GPU inference time (512-image cohort x several strengths x however many models),
  which currently competes with the K11 75k training run for GPU capacity. Not yet run.

## Open questions we don't have an answer to yet

- Does digit CC keep degrading past 45k steps, or does it recover/plateau? (75k
  continuation in progress, no results yet.)
- Is the per-attribute edit-strength mismatch (hypothesis 4) actually the cause of the
  hue/thickness opposite-failure-mode pattern, or is something else going on?
- Why is k=5's hue counterfactual specifically corrupted (CC=0.18) when k=1 and k=11
  (shallower and much deeper hierarchies) both do fine (~0.97–0.99) on the same
  attribute? No hypothesis yet.
- Is 70,000 training images (60k train / 10k val, ~6k per digit class) enough coverage
  for 11+ semantic attributes, beyond what the digit-specific nearest-neighbor check
  showed? We've only rigorously checked digit; thickness/intensity/hue haven't had the
  same joint-similarity treatment.

## Files referenced

- `experiments/hdae/outputs/counterfactual_ood_check.json` — marginal-range OOD check
- `experiments/hdae/outputs/digit_nn_ood_check.json` — nearest-neighbor OOD check (digit)
- `experiments/hdae/outputs/k1_k5_k11_tolerance_comparison_full.csv` — full 30k 3-way
  comparison table (all mult=2x/3x variants)
- `experiments/hdae/outputs/morpho_hier_k11_v3/cf1_eval_gs8_tol2x/eval.log` — 45k K11 eval
- `experiments/hdae/hdae/attr_conditioner.py` — conditioning/dropout implementation
- `experiments/hdae/outputs/attr_predictors_70k/comparison_results.json` — predictor MAEs
