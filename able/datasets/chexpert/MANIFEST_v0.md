# CheXpert · data protocol v0 — frozen 2026-11-07

All seven section-7 tests pass. The artifacts below are frozen; changing any
of them means a v1, not an edit.

## Source
`ashery/chexpert` on Kaggle (CheXpert-v1.0-small, CC0-1.0),
11,496,130,509 bytes, sha256 `25b657223f4698dd838cdf4b4b1f12331678f83cb7f9a1c56dbf910fa0ace727`.
The archive is not kept on disk — it rebuilds the memmap in about three
minutes and storing 11.5 GB twice buys nothing a digest does not. See
`/home/exouser/able_data/chexpert/PROVENANCE.md`.

## §3 filtering — 223,414 → 171,996 images, 61,407 patients
| step | out | dropped |
|---|---|---|
| frontal only | 191,027 | 32,387 lateral |
| sex ∈ {F,M}, age 18–100 | 191,023 | 1 sex, 3 age |
| modeled findings certain | 175,135 | 9,578 effusion −1, 6,310 cardiomegaly −1 |
| eval-only −1/blank → 0 | 175,135 | — |
| one image per study | 171,996 | 3,139 extra frontals |

`ap_pa` is an eval-only label, so the blank → 0 rule reads a blank view
position as AP. 0 blanks occurred; 12 rows carried a value that was neither
AP nor PA and became 0. Counted in `chexpert_meta.csv.counts.json`, not
hidden.

## §4 split and images
Patient-level 90/10, seed 20261107 — 55,266 / 6,141 patients, 154,256 /
17,740 images. Splitting on patients, not images: a patient contributes many
studies and two frontals a week apart are near-duplicates, so an image-level
split would report memorisation as held-out AUC.

`images_224.npy` (171996, 224, 224) uint8, 8.63 GB. decode → grayscale →
resize short side 224 (PIL BICUBIC, antialiased on reduction; CheXpert-small
is 320 on the short side so everything is downscaled) → centre crop 224 →
uint8. No normalisation, no augmentation — those stay model-side so the
stored pixels are the one fixed artifact every arm sees.

`radiologist_val`: valid.csv, 234 → 200 after the same filtering. Labelled by
radiologists reading the image rather than by the report-text labeller. Held
apart — it checks predictors and never tunes or generates. Its patients start
at 64541 and the training pool ends at 64540.

## §2 SCM
`sex → x`; `age → effusion → x`; `age → cardiomegaly → x`. No edge between
the findings: they correlate through an unmodeled common cause (cardiac
failure), and an arrow there would make every do(effusion) move cardiomegaly
for a reason that does not exist. The residual dependence is handled by the
`r_k(a)` correction instead.

Every variable is one Gumbel-max over K logits; for K=2 that is exactly the
protocol's logistic-with-Gumbel, so one mechanism and one exact abduction
(Maddison's top-down construction) serve all four. Fitted saturated on the
train split only — `scm_params.json`.

P(effusion=1 | age) = .3121 / .4222 / .4796 · P(cardiomegaly=1 | age) = .0956 / .1136 / .1521

## §5 predictors — DenseNet-121, shared trunk, 10-way multilabel + 3-way age
Three disjoint patient sets: fit (138,727), select (15,529, inside the train
split), held-out (17,740, never selected on because the cohort is drawn from
it). 5 epochs, batch 96, AdamW 1e-4 one-cycle, AMP. Selection AUC by epoch:
.8191 → .8339 → .8390 → (ep3) → .8418; epoch 4 selected.

**No horizontal flip.** A chest radiograph is not left-right symmetric — the
heart sits left of midline and cardiomegaly is judged from the cardiothoracic
ratio on that silhouette. Mirroring would teach the predictor that situs
inversus is ordinary and corrupt the finding the SCM leans on hardest.

| label | held-out AUC [95% CI] | radiologist_val AUC [95% CI] | gate |
|---|---|---|---|
| sex | .9964 [.9958,.9969] | .9945 [.9853,.9997] | **kept** |
| ap_pa | .9962 [.9950,.9974] | 1.0000 | **kept** |
| cardiomegaly | .8813 [.8734,.8886] | .8080 [.7415,.8685] | **kept (modeled)** |
| pneumothorax | .8694 [.8596,.8782] | .8110 [.5805,.9712] | **kept** |
| effusion | .8671 [.8615,.8721] | .9049 [.8602,.9413] | **kept (modeled)** |
| support_devices | .8397 [.8340,.8455] | .8483 [.7938,.8967] | dropped |
| edema | .8262 [.8194,.8326] | .9111 [.8620,.9539] | dropped |
| consolidation | .7194 [.7071,.7331] | .9126 [.8664,.9504] | dropped |
| lung_opacity | .7172 [.7089,.7242] | .9027 [.8575,.9430] | dropped |
| atelectasis | .6831 [.6731,.6940] | .7612 [.6921,.8264] | dropped |
| age_bin (OvR macro) | .9299 | — | **kept (modeled)** |

**Gate passed: all four modeled attributes clear 0.85.** The graph does not
need reconsidering.

Frozen kept set: `sex`, `age_bin`, `effusion`, `cardiomegaly` (modeled) +
`ap_pa`, `pneumothorax` (eval-only). Two of the seven candidate eval-only
labels survive, not seven.

### Two things the table should not be read past
**The gate is applied to noisy labels.** Held-out targets come from the
report-text labeller; radiologist_val targets come from radiologists reading
the image. `edema`, `consolidation` and `lung_opacity` fail on held-out
(.83/.72/.72) yet score .91/.91/.90 on image truth with CIs whose lower bound
is above .85. Their held-out AUC is bounded by noise in their own targets,
not by the predictor. The protocol says to gate on held-out AUC and that is
what was done, but these three are dropped for a reason that is not what the
gate is nominally measuring.

**`pneumothorax` is the mirror case.** Kept on .8694 held-out, but its
image-truth interval is [.5805,.9712] — 200 images with few positives says
almost nothing. It is in the kept set on the noisy-label evidence alone.

### T6 calibration (thresholds by Youden's J on the SELECT set, never held-out)
TPR ceiling / FPR floor on held-out REAL images — the bound a perfect
generator could reach, since a counterfactual is scored by this same
instrument:

| kept label | thr | TPR ceiling | FPR floor |
|---|---|---|---|
| sex | .249 | .969 | .026 |
| ap_pa | .780 | .963 | .003 |
| effusion | .459 | .813 | .233 |
| cardiomegaly | .431 | .806 | .192 |
| pneumothorax | .377 | .796 | .213 |
| age_bin | — | .794 accuracy | — |

Effusion and cardiomegaly carry FPR floors near .2. CC and FC on those two
have to be read against that, not against 1.0.

### `r_k(a)` — what FC must discount
Largest associations already present in real held-out images:
do(cardiomegaly)→edema .188 · do(effusion)→lung_opacity .164 ·
do(effusion)→consolidation .160 · do(effusion)→edema .155 ·
do(age)→edema .154 · do(age)→effusion .153.

## §6 cohort
512 held-out sources balanced over **all 24** joint cells of
(sex, age_bin, effusion, cardiomegaly), 21–22 per cell — the thinnest cell
has 46 available. Balanced rather than proportional on purpose: the rare
cells are where a model that has only learned the marginals gives itself
away, and proportional sampling would have yielded about three of them.
2,048 counterfactuals. One noise draw per source shared by its four
counterfactuals, so do(age) moves a finding only where the new logits cross
that unit's own draw — 105 of 512 sources.

## §7 — 7 passed, 0 failed
Prevalence is checked against raw `train.csv`'s own rates, not figures copied
from the paper. Widest gap `support_devices` +4.81 of the ±5 point budget
(dropping laterals over-weights portable AP films); `effusion` +4.01; the
rest under 3. Both finding rates strictly monotone across age bins.
Determinism rebuilds meta, split, SCM params and cohort to identical sha256;
the memmap cannot be rebuilt without the deleted archive, so for it the test
verifies the recorded digest still matches the bytes — integrity, not
determinism, and labelled as such.

Image integrity: 0 constant rows, and 20 random rows were rendered with their
meta captions and looked at (`contact_sheet_20.png`) — all frontal chest
films, centred, no decode artefacts, captions matching what the images show,
so memmap and meta rows are aligned.

## Artifacts
Digests in `/home/exouser/able_data/chexpert/SHA256SUMS`. Mirrored to
`s3://najibi-research-7f2a/able/chexpert/v0/`, except `images_224.npy`
(8.6 GB, rebuilt from Kaggle in ~3 min against a recorded digest).
