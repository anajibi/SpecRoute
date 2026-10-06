# Agent brief — baseline counterfactual models on CelebA-HQ 256

You are picking up a research thread. Your job: **train baseline counterfactual models from the
literature and report CC / FC_obs / FC_unobs / CF1 for them**, so they can sit in one table beside
the HDAE results already produced.

Everything you need is in `s3://najibi-research-7f2a/hdae-handoff/`. You do not need the machine
this was run on, and you do not need to reproduce HDAE.

---

## 1. What this project measures

A counterfactual edit flips one attribute of a face and should leave everything else alone.

    CC        did the requested edit land  (target reader returns the flipped value)
    FC_obs    did the other CONDITIONED attributes stay put
    FC_unobs  did the 28 UNCONDITIONED attributes stay put
    CF1       harmonic mean of CC with the corresponding FC

Conditioned attributes: **Male, Young, Beard, Bald**. Everything else is unconditioned.

---

## 2. Get set up

```bash
A=s3://najibi-research-7f2a/hdae-handoff
aws s3 cp $A/data_packed/celebahq_256_pack.tar.gz .   # 4 GB  -> celebahq_256.lmdb + attrs npz
aws s3 cp $A/celebahq256/readers40_per_attribute.tar.gz .  # 3.7 GB -> single/<attr>.pt x40
aws s3 cp $A/celebahq256/baseline_kit.tar.gz .        # scorer, cohort, keep list, reference
aws s3 cp $A/celebahq256/ffhq256_autoenc_model.pt .   # 1.3 GB, for a DiffAE-family baseline
```

**Code comes from git, not S3:**

```bash
git clone git@github.com:anajibi/SpecRoute.git
cd SpecRoute && git checkout celebahq-ladder      # commit 4f712b3 or later
```

Everything you need is on that branch: `experiments/hdae/baselines/` (this kit),
`experiments/hdae/scripts/*celebahq*.py` (the evaluation code), `experiments/hdae/configs/`,
and `diffae_upstream/`. S3 holds only data, weights and results.

Python env: torch + torchvision (ConvNeXt), lmdb, numpy, PIL. The readers are
torchvision `convnext_small` with a 1-unit head, fp16 weights, 256px input.

---

## 3. The evaluation contract

Your baseline does **not** have to be diffusion-based, share our sampler, or expose a guidance
scale. It writes PNGs; the shared scorer reads them.

```
<root>/recon/<index>.png                reconstruction of the cohort image, NO edit
<root>/<attr>/<strength>/<index>.png    counterfactual with <attr> flipped
```

`<index>` is zero-padded to 5 digits, from `cohort_2048.json`.
`<attr>` ∈ {Male, Young, Beard, Bald}.
`<strength>` is a free label — `g3.0` for guidance, `alpha2.5` for a GAN edit magnitude, `fixed`
for a single deterministic counterfactual. The scorer treats it as opaque.

```bash
python score_counterfactuals.py --root <root> --readers <dir of .pt> --out results.json
```

**Sweep the strength parameter.** Our data shows the optimum varies 3x across attributes of the
same model (Beard best at 2, Bald at 6). A single fixed strength handicaps some attributes badly.

---

## 4. Rules that are not negotiable

**FC is measured against the model's own reconstruction, not the source photograph.** Against the
photo you also charge the model for its encode/decode error, which has nothing to do with the
intervention. A model with no reconstruction path can pass `--recon-is-source`, but that is a
different, stricter measurement and must not share a column with reconstruction-based FC.

**Read the cohort indices from the file. Do not re-derive them.** The draw depends on a seed *and*
on the partition/attribute arrays; regenerating it can silently give different subjects.

**Reader selection must happen on data disjoint from your evaluation cohort.** Tuning and
threshold fitting on the cohort is fine. Choosing *which attributes to keep* on cohort numbers is
selection on the test set. We cut 8 readers at F1 < 0.55 measured on a partition-0 slice that
never enters the cohort — see `selection_metrics.json`.

**Readers run untuned (`logit > 0`)** except the attribute being intervened on, which uses an
F1-optimal threshold. CC counts positive readings, so a permissive reader inflates it: our Bald
reader at the balanced-accuracy threshold predicted bald at 13% against a 6% base rate.

---

## 5. Mistakes made here, so you can skip them

**A domain gap that did not exist.** Reported for days that the readers lose ~5 points on generated
images, and built two rounds of augmentation around fixing it. It came from comparing the
mixed-sex test split against the all-male cohort — a cohort shift. Measured on the *same* subjects,
source vs reconstruction is 0.8473 vs 0.8462, gap −0.001, 0 of 35 attributes losing >5 points.
**Always compare the same subjects.**

**Selecting readers on cohort numbers.** Wearing_Lipstick scores F1 0.367 on the all-male cohort
and 0.950 on a clean slice — it has 22 positives among 2,048 men. Cutting on cohort numbers
discards good instruments.

**A saturated reader hides drift.** Our Blurry reader had balanced accuracy 0.760 and F1 0.033 —
it fired on 25% of images at a 0.4% base rate, so it never changed between reconstruction and
counterfactual and reported FC ≈ 1.0. Balanced accuracy does not catch this; compare prediction
rate against base rate.

**FC has a floor.** Running the pipeline with `do(A = its existing value)` gives FC of 0.92–0.96,
not 1.0 — guidance perturbs the image anyway and readers disagree with themselves on near-identical
images. Normalise as `FC / floor`, **not** `(FC − floor)/(1 − floor)`; the floor is a ceiling on
achievable preservation, not a baseline to subtract. Run this control for your baseline.

**`Male` inflates FC_obs** on an all-male cohort: it reads "preserved" for every subject.

**Size equality is not proof of a complete upload.** A `tar` that hits a full disk uploads and
"verifies" perfectly. Read archives back and count members.

---

## 6. HDAE reference numbers

From `reference_hdae.json`. Target-only nulling, 2,048 held-out males, T=50 DDIM, FC_unobs over
28 readers.

| attribute | best g | CC | FC_obs | FC_unobs | CF1 |
|---|---|---|---|---|---|
| Beard | 2 | 0.9458 | 0.9207 | 0.8161 | 0.8788 |
| Male  | 5 | 0.9194 | 0.7686 | 0.7445 | 0.8220 |
| Young | 5 | 0.8379 | 0.8406 | 0.7515 | 0.7920 |
| Bald  | 6 | 0.8662 | 0.7601 | 0.7117 | 0.7832 |

For a diffusion baseline using classifier-free guidance, the **null set** matters as much as the
scale. Nulling only the edited attribute beat nulling all four everywhere we tested
(ΔCF1: Beard −0.008, Bald −0.025, Young −0.065, Male −0.116). Record which convention you use;
results are not comparable across conventions.

---

## 7. What to deliver

1. `results.json` per baseline from `score_counterfactuals.py`, strength swept per attribute.
2. The null-intervention control for each baseline (`do(A = existing value)`), so FC is normalisable.
3. A note on which null-set / edit convention each baseline used.
4. Reader accuracy on your own reconstructions if you want a per-model instrument check — ground
   truth is in the npz. Our measurement says this is not needed, but it is cheap.

Upload to `s3://najibi-research-7f2a/hdae-handoff/baselines/<model_name>/`.

---

## 8. Open questions in this thread

- The null-intervention control predates per-attribute FC storage, so its floor is computed over an
  earlier 22-reader set rather than the current 28. Re-run it if you need them on one basis.
- `Young`'s optimal guidance moved from 3 to 5 when the reader set changed; its CF1 curve is nearly
  flat, so treat that one as weakly determined.
- The person running this thread flagged the CF1 definition as wrong. The formula used is the
  harmonic mean of CC and FC, matching `aggregate_metrics.py`. The objection was never resolved —
  ask before relying on CF1 rankings.
