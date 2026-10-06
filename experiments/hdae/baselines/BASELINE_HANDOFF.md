# Baseline counterfactual evaluation — CelebA-HQ 256

Everything needed to train a baseline on another cluster and produce CC / FC / CF1 numbers that
sit in the same table as the HDAE results.

## What's in S3

    s3://najibi-research-7f2a/hdae-handoff/

    data_packed/celebahq_256_pack.tar.gz        5.7 GB  LMDB + attribute npz (30,000 images, 256px)
    celebahq256/readers40_per_attribute.tar.gz  3.7 GB  40 attribute readers (the evaluators)
    celebahq256/baseline_kit.tar.gz             small   scripts, frozen cohort, reference results
    celebahq256/k11_cd015r/                     2.8 GB  HDAE checkpoints, for reference comparison

Unpack the dataset so you get `celebahq_256.lmdb/` and `celebahq_256_attrs.npz` side by side.

## The evaluation contract

A baseline does **not** need to match our architecture, sampler, or even be diffusion-based.
It writes PNGs and the shared scorer reads them:

    <root>/recon/<index>.png                  model's reconstruction, NO edit
    <root>/<attr>/<strength>/<index>.png      counterfactual with <attr> flipped

`<index>` is the CelebA-HQ index from `cohort_2048.json`, zero-padded to 5 digits.
`<attr>` is one of Male, Young, Beard, Bald.
`<strength>` is a free-form label for whatever knob the model exposes:

| model family | strength | example |
|---|---|---|
| diffusion + classifier-free guidance | guidance scale | `g3.0` |
| GAN latent editing | edit magnitude | `alpha2.5` |
| single deterministic counterfactual | n/a | `fixed` |

Then:

    python score_counterfactuals.py --root <root> --readers <readers_dir> --out results.json

## Why a reconstruction set is required

FC asks whether attributes the edit should not touch stayed put. Measured against the **source
photograph**, it would also charge the model for its own encode/decode error, which has nothing to
do with the intervention. Measured against the model's **own reconstruction**, that error cancels.

So every model supplies its reconstruction of each cohort image. A model with no reconstruction
path can pass `--recon-is-source`, but that is a different, stricter measurement and must not be
put in the same column.

## The cohort is frozen — do not re-derive it

`cohort_2048.json` holds 2048 explicit indices: held-out males (partitions 1 and 2) with Bald
flipped in **both** directions — 121 already-bald asked to become haired, 1927 the reverse.

The draw depends on a seed *and* on the partition/attribute arrays. A baseline that regenerates
the cohort itself could silently get different subjects and produce numbers that look comparable
but are not. Read the indices from the file.

## Metrics

    CC        fraction where the target reader returns the REQUESTED (flipped) value
    FC_obs    mean over the other three conditioned attributes of the fraction whose reading is
              UNCHANGED from that subject's reconstruction
    FC_unobs  same, over 22 unobserved attributes (see reader_keep.json)
    CF1       harmonic mean of CC with the corresponding FC

### Which readers count, and why some don't

`reader_keep.json` lists **28 readers for FC_unobs**, plus the 4 the model conditions on. Eight
were cut at **F1 < 0.55**.

**The selection was made on the partition-0 held-out slice (2,400 images), never on the cohort.**
This matters more than it may look. Scored on the all-male cohort, several readers appear broken
that are in fact fine — Wearing_Lipstick reads F1 0.367 there and 0.950 on the selection slice,
Heavy_Makeup 0.211 versus 0.909 — because those attributes have almost no positives among men.
Selecting on cohort numbers would have discarded good instruments *and* kept only the readers that
happen to look good on the evaluation set, which is selection on test data.

Cut (F1 on the selection slice): Blurry 0.033, Chubby 0.480, Wearing_Necklace 0.497,
Pale_Skin 0.512, Oval_Face 0.514, Narrow_Eyes 0.523, Mustache 0.532, Double_Chin 0.534.

Tuning and threshold fitting **may** use the cohort; attribute selection may not. If you add or
replace readers, select them on data disjoint from your evaluation cohort.

Two failure modes to watch, which pull FC in opposite directions:

- **noisy** readers flip on near-identical images and *manufacture* drift the model never caused.
- **saturated** readers never change their answer, so they report FC ≈ 1.0 and *hide* drift.
  Blurry was ours: F1 0.033, firing on 25% of images at a 0.4% base rate. Balanced accuracy does
  not catch this — compare prediction rate against base rate.

### A caveat about FC_obs on an all-male cohort

`Male` is one of the four conditioned attributes and therefore enters FC_obs for interventions on
the other three. On this cohort every subject is male, so that reader reads "preserved" on
essentially every subject and contributes a near-1.0 term. FC_obs is inflated by roughly a third
as a result. FC_unobs does not have this problem.

## Thresholds

Readers are applied **untuned** (`logit > 0`). Tuning thresholds on real photographs was measured
to *hurt* accuracy on generated images — it helped only 15 of 37 attributes and lowered the mean
from 0.8356 to 0.8317. The untuned operating point is the honest default.

`--target-f1-threshold` switches only the attribute being intervened on to its F1-optimal
threshold. This matters because CC counts positive readings: at the balanced-accuracy threshold
our Bald reader predicted bald at 13% against a 6% base rate (precision 0.44), which inflated CC.

## Evaluator transfer to generated images — measured, and it is not a problem

An earlier version of this document warned that the readers lose ~5 points of balanced accuracy on
generated images and that the handicap differs by architecture. **That was an error**, and it is
withdrawn. It came from comparing the 3,000-image mixed-sex test split against the 2,048 all-male
cohort and attributing the difference to real-vs-generated, when it was almost entirely cohort
composition — attribute base rates differ sharply between men and the general population.

Measured correctly — the **same** 2,048 subjects, their source photographs against their HDAE
reconstructions, across the 35 attributes with at least 30 positives and 30 negatives:

    mean on source photographs   0.8473
    mean on reconstructions      0.8462
    mean gap                    -0.0010
    attributes losing >5 points   0 / 35   (worst single attribute: -0.022)

Several attributes score *higher* on reconstructions (Wearing_Necklace +0.039, Arched_Eyebrows
+0.027, Big_Nose +0.023). So a shared evaluator is sound, and you do not need a per-model
calibration pass for domain shift.

What the readers *are* limited by is CelebA's own label noise on a few attributes — Big_Lips,
Pointy_Nose and Oval_Face sit near 0.70 balanced accuracy on photographs, and three independent
architectures hit the same wall. Those are excluded from FC_unobs for that reason, not for any
generation-related one.

Caveat that does still apply: the cohort is all-male, so attributes that are rare in men
(Rosy_Cheeks 15 positives, Wearing_Lipstick 22, Blond_Hair 44) cannot be estimated reliably on it
regardless of the reader's quality. Those are excluded too.

## FC has a floor — measure it

Running the identical pipeline with **no intervention** (`do(A = its existing value)`) gives FC
well below 1.0, because guidance perturbs the image anyway and readers disagree with themselves on
near-identical images. For HDAE that floor was 0.9232–0.9627 depending on configuration.

Normalise as `FC / FC_floor`, **not** `(FC − floor)/(1 − floor)` — the floor is a ceiling on
achievable preservation, not a baseline to subtract.

A baseline that skips this control will report FC values that cannot be compared to normalised ones.

## HDAE reference numbers

`reference_hdae.json` carries our results at matched protocol. Headline, target-only nulling,
best guidance per attribute:

| attribute | best g | CC | FC_obs | FC_unobs | CF1 |
|---|---|---|---|---|---|
| Beard | 2 | 0.9458 | 0.9207 | 0.8069 | 0.8708 |
| Young | 3 | 0.7710 | 0.8804 | 0.7806 | 0.7758 |
| Male  | 5 | 0.9194 | 0.7686 | 0.7458 | 0.8236 |
| Bald  | 6 | 0.8662 | 0.7601 | 0.6942 | 0.7708 |

Optimal guidance varies **3×** across attributes of the same model, so a single fixed strength
handicaps some attributes badly (Beard at g=6 loses 2.8 CF1 points; Bald at g=2 loses 18.2).
Sweep the strength parameter per attribute.

## For diffusion baselines specifically

If the baseline uses classifier-free guidance, the **null set** matters as much as the scale.
Nulling only the attribute being edited beat nulling all four at every attribute we tested:

| attribute | g | ΔCC (all-4 − target-only) | ΔCF1 |
|---|---|---|---|
| Beard | 2 | −0.0073 | −0.0087 |
| Young | 3 | −0.1104 | −0.0625 |
| Male  | 5 | −0.2671 | −0.1144 |
| Bald  | 6 | −0.0664 | −0.0213 |

Note FC often *favours* all-four nulling. That is an artifact: nulling everything amplifies the
untouched attributes toward their conditioned values, so a **binary** reader still answers "same"
and scores them preserved while they are being intensified. CF1 is the honest summary.

Record which null-set convention a baseline uses; results are not comparable across conventions.
