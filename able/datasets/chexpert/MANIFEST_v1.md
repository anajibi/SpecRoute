# CheXpert · protocol v1 — amended and refrozen 2026-11-07

v1 changes exactly one thing: **how eval-only attributes are gated.** The data
(§3, §4), the SCM (§2) and the cohort (§6) are byte-identical to v0 — same
sha256s, verified by the determinism test. `MANIFEST_v0.md` remains the
description of those; this file records the amendment and what it moved.

## Amendment — §5 gate for eval-only attributes

**Before (v0):** eval-only label kept iff held-out AUC ≥ 0.85.
**After (v1):** eval-only label kept iff **radiologist_val AUC ≥ 0.85 AND the
95% bootstrap CI lower bound ≥ 0.80.**

**Why.** Held-out targets come from the report-text labeller. A label whose
*reports* are inconsistent scores low however well the predictor reads the
image, so the v0 gate was measuring label noise, not predictor capability.
The label-noise audit (below) makes this concrete.

**Why the CI condition.** radiologist_val is 200 images. A bare point estimate
admits a label on luck. The CI floor is what disqualifies pneumothorax, whose
interval runs [.60, .97].

**Scope.** Eval-only attributes only. Modeled attributes keep the v0 gate.

| label | scope | radiologist_val AUC [95% CI] | held-out (report) | v0 | **v1** |
|---|---|---|---|---|---|
| sex | modeled | .9945 [.9858,.9997] | .9964 | kept | **kept** |
| ap_pa | eval-only | 1.0000 [1.000,1.000] | .9962 | kept | **kept** |
| consolidation | eval-only | .9126 [.8658,.9495] | .7194 | dropped | **re-admitted** |
| edema | eval-only | .9111 [.8612,.9517] | .8262 | dropped | **re-admitted** |
| lung_opacity | eval-only | .9027 [.8560,.9425] | .7172 | dropped | **re-admitted** |
| effusion | modeled | .9049 [.8580,.9442] | .8671 | kept | **kept** |
| support_devices | eval-only | .8483 [.7934,.8968] | .8397 | dropped | dropped |
| pneumothorax | eval-only | .8110 [.6007,.9722] | .8694 | kept | **revoked** |
| cardiomegaly | modeled | .8080 [.7376,.8684] | .8813 | kept | **kept** (see below) |
| atelectasis | eval-only | .7612 [.6896,.8234] | .6832 | dropped | dropped |
| age_bin | modeled | .9523 OvR | .9299 | kept | **kept** |

**Frozen v1 kept set (8, was 6):** `sex`, `age_bin`, `effusion`,
`cardiomegaly` (modeled) + `ap_pa`, `edema`, `lung_opacity`, `consolidation`.

**Flagged, not acted on:** `cardiomegaly` scores .8080 [.7376,.8684] against
radiologists and **would not pass the new gate.** It is a modeled attribute
and the amendment is scoped to eval-only, so it keeps the v0 gate, which it
passes at .8813. If the amendment is ever extended to modeled attributes, the
graph itself has to be reconsidered — cardiomegaly is a parent of the image
in the SCM.

## T6 recomputed on radiologist_val

Thresholds are still Youden-tuned on the **select** set, i.e. at a
report-calibrated operating point; only the set they are measured on changed.

| label | thr | radiologist_val TPR / FPR | held-out (report) TPR / FPR |
|---|---|---|---|
| sex | .249 | .972 / .021 | .969 / .026 |
| ap_pa | .780 | 1.000 / .006 | .963 / .003 |
| consolidation | .416 | .844 / .161 | .687 / .371 |
| edema | .517 | .833 / .133 | .788 / .287 |
| lung_opacity | .396 | .733 / .131 | .829 / .510 |
| effusion | .459 | .656 / .118 | .813 / .233 |
| cardiomegaly | .431 | **.515** / .082 | .806 / .192 |
| age_bin | — | .815 accuracy | .794 accuracy |

This corroborates the amendment from the other side. **FPR floors collapse
when measured against image truth**: lung_opacity .510 → .131, consolidation
.371 → .161, effusion .233 → .118, cardiomegaly .192 → .082. Those "false
positives" were largely findings present in the image that the report never
mentioned. TPR moves the other way for the two modeled findings — radiologists
mark positives a report-trained predictor misses, and cardiomegaly's .515 is
the weakest number in the protocol.

## Label-noise audit — what it actually found

Reader: TorchXRayVision `densenet121-res224-all`, no training, 191,027 frontal
images swept. Full tables in `outputs/chexpert_noise/noise_report.{json,md}`.

**The plan's `gap` as literally defined came out negative for most labels, and
that is an artefact.** It compares AUC against radiologists on all 234 images
with AUC against reports on only the rows carrying an explicit 0 or 1 — which
is 20–60% of the training set and precisely the rows where the labeller *named*
the finding, i.e. the unambiguous ones. Recomputing the report side under the
convention the protocol actually uses for eval-only labels — blank counts as
negative, so both sides score every image — reverses it:

| label | gap (plan's defn) | **gap (comparable)** | blank % | verdict |
|---|---|---|---|---|
| Enlarged Cardiomediastinum | −.006 | **+.258** | 79.9% | unreliable |
| Atelectasis | +.103 | **+.217** | 69.4% | unreliable |
| Consolidation | −.040 | **+.199** | 68.4% | unreliable |
| Lung Opacity | +.032 | **+.189** | 47.3% | unreliable |
| Edema | −.014 | **+.073** | 61.5% | noisy |
| Effusion | −.051 | **+.067** | 40.4% | noisy |
| Cardiomegaly | −.084 | **+.041** | 79.3% | clean |
| Pneumothorax | −.097 | **−.080** | 64.7% | not auditable (8 positives in valid) |

**The noise lives in the blank-means-negative convention**, and it scales with
how often the labeller stays silent. The three labels v1 re-admits
(consolidation, lung_opacity, edema) sit at +.199, +.189, +.073.

**Two S6 checks fail, and both are recorded rather than smoothed over.**

- **S6.2 (ordering)** — FAIL. Consolidation ranks 5th by the plan's gap but
  1st by confident-disagreement, a 4-place shift. Under the comparable gap it
  ranks 3rd and the inversion nearly vanishes. The plan also predicted that
  labels with uncertain ≥ positive (consolidation, atelectasis, enlarged
  cardiomediastinum) would show a *large* gap: they are the top three under
  the comparable gap and scattered under the literal one. The plan's own
  prediction therefore discriminates between the two definitions.
- **S6.4 (chex control)** — FAIL, and uninformative rather than contradictory.
  The plan expected `densenet121-res224-chex` to beat `-all` against the
  report labels. It loses on **10 of 10** — and also loses against the
  radiologists on 7 of 9. It is simply the weaker reader, so the control
  cannot isolate what it was designed to isolate.
- **S6.1** — not runnable. The 668-image test set's labels are public
  (`rajpurkarlab/cheXpert-test-set-labels`, 499 studies, 5-radiologist
  majority) but its images are behind Stanford AIMI registration and are
  absent from the Kaggle mirror. **The audit has one reference set, not two.**
- **S6.3** — PASS, after calibration. At the fixed 0.9/0.1 thresholds,
  confident disagreement flags 2.6–9.4% of the *radiologist* set for six of
  eight labels, where no report-label error exists to find; that number is
  reported throughout as a **model-error floor**. Per-label thresholds
  calibrated on the radiologist set (hi = 99th percentile among negatives,
  lo = 1st percentile among positives) hold the false-flag rate at ≤1.73%.

**Independence caveat, which weakens the audit and should not be glossed:**
`densenet121-res224-all` was trained on NIH + PadChest + **CheXpert** + MIMIC
+ RSNA. Its CheXpert component was trained on these very report labels, so the
reader is only *partly* independent of what it is auditing, and AUC-against-
report is biased upward. That bias pushes the literal gap negative and is the
likeliest reason it comes out so.

**Not auditable with XRV:** `support_devices` (no XRV output at all),
`pneumothorax` (8 positives in the 234-image reference; AUC CI [.50,.79] and
the head never exceeds p=.791 on any of the 234), `lung_lesion` (1 positive),
`fracture` (0 positives). Kept in every table and flagged, per instruction.

**Age** (sanity only — age labels do not come from reports): MAE **8.37 years**,
bias −1.34, 3-bin accuracy .706 over 191,024 images.

## §7 — 7 passed, 0 failed against v1
Unchanged tests, rerun with the v1 kept set. Determinism still reproduces
meta, split, SCM params and cohort to identical sha256, which is the evidence
that v1 touched only the gate.

## Artifacts
`/home/exouser/able_data/chexpert/SHA256SUMS`; mirrored to
`s3://najibi-research-7f2a/able/chexpert/v1/`. `images_224.npy` (8.6 GB) is
not mirrored — it rebuilds from Kaggle in ~3 min against a recorded digest,
and that digest was re-verified this session when the archive was re-fetched.
