# CheXpert — ABLE protocol v2 and the k=1 arm

**Status: paused and fully archived, 2026-11-09.** Nothing here needs to be retrained or
rebuilt. Everything below is in S3 and verified by size against the local copies before the
disk was cleared.

- **Git branch:** `chexpert-k1-archive` (this file's branch), forked from `chexpert-v0`
- **S3 root:** `s3://najibi-research-7f2a/able/chexpert/v2/`
- **Returned to:** `celebahq-ladder` (4525f6c), the last branch before shapes3d and chexpert

---

## 1. Where things stand

### The protocol (frozen, v2)

Four modelled attributes, chosen because each clears a gate on 200 radiologist-labelled
images — AUC ≥ .85 with a bootstrap CI lower bound ≥ .80. A counterfactual score cannot
exceed what the instrument reading it can see, which is why the gate exists.

| attribute | role | radiologist_val AUC |
|---|---|---|
| sex | modelled, root | .9945 |
| age_bin (3-class) | modelled, root | .9523 [.930, .972] |
| effusion | modelled, child of age | .9049 [.858, .944] |
| edema | modelled, child of age | .9111 [.859, .944] |
| ap_pa, lung_opacity, consolidation | eval-only | 1.0000 / .9027 / .9126 |

Graph: `age_bin → effusion`, `age_bin → edema`, `sex` an independent root.
Cardiomegaly was the second modelled finding in v0/v1 and was **replaced by edema**: it reads
.8080 with a TPR ceiling of .515, so `do(cardiomegaly)` would have been capped by measurement
rather than by the generator. Three independent readers agree (XRV .8063, CheXzero .8544 with
CI lo .797, ours .8080).

### The model

DiffAE hierarchical autoencoder, **k=1** (single tap at the bottleneck, 512 dims, all 18
decoder blocks on one level), fine-tuned from `ffhq256_autoenc`.

| | |
|---|---|
| init | 93.2% of params pretrained (706/827 tensors); 12.35M random = FiLM, attr embedding, tap heads |
| resolution | 256 (memmap is 224, upsampled; at img_size 224 the UNet grows an extra decoder ResBlock the checkpoint cannot fill) |
| batch | 4 × accum 16 = 64, lr 1e-4, ema .9999, T 1000 / T_eval 100 |
| trained | **12,496 of 15,625 steps (80%)**, ≈800k of 1M samples |
| crashes / non-finite | 0 |

Stopped deliberately at 80%, not by failure. **Resumable** — see §4.

### Results

**Reconstruction works.** L1 vs source 0.0084. The frozen predictor reads attributes back off
reconstructions at sex 16/16, effusion 16/16, edema 14/16, age_bin 12/16.

**Conditioning works for three of four attributes.** At g=5, on 16 held-out sources
(`do(X)` real flips = reading moved off the reconstruction's value and landed on target):

| intervention | hit | real flips | freebies | wrong | **CC** |
|---|---|---|---|---|---|
| do(age_bin) | 9/16 | 7/16 | 2 | 5 | **+0.300** |
| do(sex) | 9/16 | **9/16** | 0 | 0 | −0.400 |
| do(effusion) | 6/16 | 6/16 | 0 | 0 | −0.250 |
| do(edema) | 4/16 | 2/16 | 2 | 0 | −1.400 |

The ordering tracks how gross the visual signature is: sex (global anatomy) > effusion (focal,
high-contrast) > age (diffuse, 3-way) > edema (diffuse, low-contrast texture). Preservation is
clean — `sex` stays 16/16 under every other intervention, `effusion` 16/16 under `do(sex)` and
`do(edema)`. `do(age_bin)` correctly propagated to both children (effusion 14–15/16 matched the
SCM's target).

**CC is negative for sex and effusion only because n=16 makes the constant-predictor baseline
brutal** (it scores .6875 on an 11/5 split). Real-flip counts are the honest read at this n.
The protocol cohort is 512 sources / 2,048 counterfactuals and will give stable numbers.

**Guidance: g=5 is optimal.** g=3 is safe, g=1 is too weak, and **g ≥ 8 produces artifacts, not
findings** — at g=8 the three readers report `neg→pos` as 9/11, 3/11 and 0/11 on *identical
pixels*, while preservation collapses (FC sex 1.000 → 0.400 at g=8 → 0.000 at g=12). A real
edit moves all readers together. Report reader agreement beside CC.

**do(edema) is flat across training** — real flips 3 → 4 → 2 at steps 6,248 / 7,810 / 12,496.
It is also directionally asymmetric: clearing edema works ~60%, creating it ~18%, because ~74%
of training images are edema-negative and the diffusion prior favours the dominant mode.

### Known issues, carried forward

1. **The graph is misspecified.** It asserts `effusion ⫫ edema | age`; the data rejects this —
   φ stays +0.11 / +0.08 / +0.13 within each age stratum (overall +0.1219, n=161,891). The real
   common cause is cardiac filling pressure, whose best proxy (cardiomegaly, φ +0.1917 with
   edema) was the attribute the measurement gate forced out. `r_k(a)` discounts the expected
   co-movement (do(edema) → effusion +0.1375) but it is a patch, not a fix.
2. **The AP/PA confound is the largest correction in the protocol**: `r_ap_pa(edema) = +0.1742`
   (AP rate 0.2286 → 0.0544). Edema patients are imaged AP at the bedside.
3. **Every hyperparameter was tuned on MorphoMNIST and CelebA-HQ**, not radiographs. One is
   already known wrong: the inherited `cfg_guidance_scale: 3.0` should be 5. `attr_dropout_prob
   0.12` is the next candidate.
4. **n=16 everywhere in the counterfactual results.** Nothing here is cohort-scale.

---

## 2. Where everything lives

All under `s3://najibi-research-7f2a/able/chexpert/v2/` (228 objects, 32.4 GB across all
versions). Local copies were deleted after size verification.

| path | what |
|---|---|
| `images_224.npy` | **8.6 GB memmap, 171,996 × 224 × 224 uint8.** The one artifact that cannot be regenerated without re-downloading the 11.5 GB Kaggle archive. |
| `chexpert_meta.csv` | the frozen row set, sha256 `025b913e…5e34e28` |
| `label_uncertainty.csv` | `-1`/blank flags for 13 labels, row-aligned with the meta |
| `split.npy`, `chexpert_split.json` | patient-level split, 154,256 train / 17,740 held-out |
| `scm_params.json` | fitted saturated mechanisms (edema on certain rows only) |
| `eval_cohort.json` | 512 sources × 24 cells → 2,048 counterfactuals |
| `radiologist_val_224.npy`, `radiologist_val_meta.csv` | the 200-image gate set |
| `source_csv/` | original CheXpert `train.csv` / `valid.csv` |
| `predictors/` | `best.ckpt` (10-head, frozen v2 instrument) + report |
| `SHA256SUMS`, `SHA256SUMS_v2`, `MANIFEST_v2.md`, `PROVENANCE.md` | integrity and provenance |
| `hdae/k1/` | **8 training checkpoints**, steps 1,562 → 12,496, 2.9 GB each, plus `train.log` and `janitor.log` |
| `results/cf/` | counterfactual pixels, scores, figures (see §3) |
| `results/readers_all/`, `results/readers/` | three-reader comparison, probabilities included |
| `results/noise/` | the label-noise audit (S0–S6) |
| `results/views/` | dataset contact sheets |
| `readers_20261108/`, `readers_all_20261108/`, `views_20261108/` | earlier mirrors of the same |

Earlier protocol versions are preserved at `able/chexpert/v0/` and `able/chexpert/v1/`.

### Saved counterfactual pixels

`results/cf/` holds the generated images as uint8 arrays, so **any new reader or question can be
scored without a GPU and without regenerating**:

| directory | checkpoint | contents |
|---|---|---|
| `px_step7810_g358/` | 7,810 | source, recon, do(edema) at g=3,5,8 |
| `px_latest_allg/` | 10,934 | source, recon, do(edema) at g=1,3,5,8,12 |
| `px_multivar/` | 12,496 | source, recon, do(sex/effusion/age_bin/edema) at g=3,5 |

Each has a `manifest.json` with the 16 source row indices, the factual attributes and the SCM's
post-intervention targets.

---

## 3. Code (this branch)

| file | purpose |
|---|---|
| `build_meta.py`, `build_split.py`, `build_images.py` | §3 filter, patient split, memmap. **Frozen at v0** — the row set they define is what everything is aligned to. |
| `scm_exact.py` | the graph + exact Gumbel-max abduction |
| `fit_scm.py`, `build_cohort.py` | mechanism fit, 512-source cohort |
| `train_predictors.py`, `train_head11.py`, `calibrate_predictors.py` | the measurement instruments |
| `regate_v1.py`, `regate_v2.py` | the radiologist-gate amendments |
| `audit_label_noise.py`, `noise_report*.py` | the label-noise audit |
| `chexzero_eval.py`, `chexzero_full.py`, `readers_*.py`, `xrv_score_pixels.py` | the three-reader comparison |
| `datamodule.py` | **training data.** No flip, held-out never trained on, uncertain edema dropped. |
| `causal_graph_chexpert.yaml` | model-side graph (`load_cond_specs` reads this) |
| `cf_render_save.py`, `cf_render_multivar.py` | generate counterfactuals and **save the pixels** |
| `cf_score_pixels.py`, `cf_score_multivar.py`, `cf_gsweep.py`, `cf_images.py` | scoring, flip taxonomy, guidance sweeps, figures |
| `supervise_k1.sh` | unattended crash/non-finite supervisor |
| `tests_protocol.py` | §7 integrity tests — 7 passed, 0 failed |

Config: `experiments/hdae/configs/chexpert256_k1.yaml`.
Trainer wiring: `experiments/hdae/scripts/train.py` (chexpert branch) and
`experiments/hdae/hdae/config_io.py` (`data_dir` key).

---

## 4. How to pick this back up

```bash
# 1. data
mkdir -p ~/able_data/chexpert
aws s3 sync s3://najibi-research-7f2a/able/chexpert/v2/ ~/able_data/chexpert/ \
    --exclude "hdae/*" --exclude "results/*"

# 2. resume training from 12,496 -> 15,625
aws s3 cp s3://najibi-research-7f2a/able/chexpert/v2/hdae/k1/epoch=5-step=12496.ckpt \
    experiments/hdae/outputs/chexpert256_k1/checkpoints/last.ckpt
python experiments/hdae/scripts/train.py --config experiments/hdae/configs/chexpert256_k1.yaml
# train.py resumes from the newest checkpoint by global_step. max_steps is ABSOLUTE.

# 3. or just re-score the saved pixels, no GPU training needed
aws s3 sync s3://najibi-research-7f2a/able/chexpert/v2/results/cf outputs/chexpert_cf
python able/datasets/chexpert/cf_score_multivar.py --px outputs/chexpert_cf/px_multivar
```

Checkpoints are 2.9 GB each; a long run needs `ckpt_janitor.sh` alongside it
(`bash experiments/hdae/scripts/ckpt_janitor.sh <ckpt-dir> able/chexpert/v2/hdae/k1 0`),
which uploads, ETag-verifies and then prunes local copies.

### If resuming, change these first

1. `cfg_guidance_scale: 3.0 → 5.0` (measured, not guessed).
2. Raise the budget. 1M samples was a signal check; CelebA-HQ used 2M for phase 1 alone.
3. Evaluate on the 512-source cohort, not 16 sources.
4. Report **reader agreement** beside CC, and report CC **split by direction** — pooling hides
   that the model deletes findings far more readily than it creates them.
5. Consider whether `effusion ⫫ edema | age` is defensible, or whether the graph needs an edge.
