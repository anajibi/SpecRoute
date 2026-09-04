# Reproduction ledger

Every result in this project, tied to the branch + commit that produces it and the exact
artefacts it was computed from. To reproduce a row: check out the commit, pull the listed
checkpoints from S3, run the listed command.

    git clone git@github.com:anajibi/SpecRoute.git && cd SpecRoute
    git checkout <commit>

**Branch:** `wip-from-826a88e` (forked from upstream commit `826a88e`)
**Remote:** `git@github.com:anajibi/SpecRoute.git`
**S3 root:** `s3://najibi-research-7f2a/wip-from-826a88e/`

Checkpoint integrity is verified by **multipart ETag**, not size. To check a download:

```python
import hashlib
m = []
with open(path, "rb") as f:
    while (b := f.read(8 * 1024 * 1024)):
        m.append(hashlib.md5(b).digest())
print(hashlib.md5(b"".join(m)).hexdigest() + f"-{len(m)}")
```

---

## Tags

| tag | commit | date | what it marks |
|---|---|---|---|
| `c3di-v1-20260825` | `5f57874` | 2026-08-25 | predictors + SCM + conditioning-ablation checkpoints |
| `c3di-v2-validated-20260827` | `08d2a21` | 2026-08-27 | k=1 and k=11 full runs finished and evaluated |

---

## Results, newest first

### Policy: epoch 50 is the canonical checkpoint for every model

k=1, k=5 and k=11 are all released at **epoch 50**, giving one matched-epoch protocol across the
depth ladder. Everything past epoch 50 has been deleted; intermediates at or below 50 are kept.

This is a deliberate choice made against the evidence below, which shows both measured models
peak EARLIER than 50 on counterfactual quality. Matched-epoch comparability was preferred over
per-model best-checkpoint selection. The per-epoch numbers are preserved so a best-checkpoint
protocol can be reconstructed later without re-running anything.

### `2c43fe6`+ — k=11 saturates at epoch 37; the extension bought nothing

Six checkpoints, epochs 12-59, all at one fixed guidance setting
(`class=8 pos_spl=1.5 pos_obj=2.5 rot_obj=3`), n=256, T=50:

| epoch | CC | FC obs | FC unobs | CF1 obs | pos_spl | pos_obj | rot_obj |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.9407 | 0.9640 | 0.9134 | 0.9522 | 0.0227 | 0.0359 | 0.1097 |
| 25 | 0.9517 | 0.9678 | 0.9382 | 0.9597 | 0.0211 | 0.0304 | 0.0841 |
| **37** | **0.9534** | 0.9672 | **0.9565** | **0.9603** | **0.0208** | **0.0301** | 0.0800 |
| 50 | 0.9521 | 0.9670 | 0.9557 | 0.9595 | 0.0213 | 0.0310 | **0.0752** |
| 53 | 0.9516 | 0.9681 | 0.9555 | 0.9597 | 0.0220 | 0.0311 | 0.0757 |
| 59 | 0.9497 | 0.9672 | 0.9557 | 0.9583 | 0.0227 | 0.0312 | 0.0775 |

`pos_spl` at epoch 59 is 0.0227 -- exactly its epoch-12 value. Meanwhile the stitched loss fit
over epochs 0.2-65.8 gives `L(e) = 0.0026914 * e^-0.3017 + 0.000489`, claiming 61.4% of the loss
is still reducible at a rate of 0.279%/epoch. **The loss-plateau criterion points the wrong way on
both models** and should not be used for this question again.

```bash
bash <driver>   # cfg_sweep_c3di.py per checkpoint, --attr-g class=8 pos_spl=1.5 pos_obj=2.5 rot_obj=3
python experiments/hdae/scripts/k11_epoch_curves.py
```

Outputs: `experiments/hdae/outputs/k11_epoch_curves.json`, `k11_convergence.png`,
`cfg_sweep/sweep_k11ep{12,25,37,50,53,59}.json` + matching `persample_*.npz`, and the extension's
TensorBoard history under `outputs/c3di_k11_ext75/logs/`. The epoch-53 and 59 WEIGHTS were deleted
under the epoch-50 policy; the measurements they produced are all retained.

### `c2454f7` — k=1 is over-trained at 50 epochs

The finding that changed the plan: all four counterfactual metrics degrade monotonically
across k=1's four checkpoints while reconstruction stays flat and training loss keeps falling.

| epoch | class | pos_spl | pos_obj | rot_obj | recon (pos_obj) |
|---:|---:|---:|---:|---:|---:|
| 8 | **97.66%** | **0.0291** | **0.1422** | **0.1505** | 0.0128 |
| 22 | 96.09% | 0.0345 | 0.1662 | 0.1485 | 0.0126 |
| 36 | 92.19% | 0.0364 | 0.1828 | 0.1611 | 0.0127 |
| 50 | 87.11% | 0.0390 | 0.1976 | 0.1767 | 0.0125 |

Reproduce (n=256, T=50, each intervention at k=1's best g):

```bash
python experiments/hdae/scripts/cfg_sweep_c3di.py \
  --config experiments/hdae/configs/c3di_hier_k1_final.yaml \
  --ckpt <epoch-N checkpoint> --label ep_k1_eN --weights ema \
  --n 256 --sample-bs 64 --grid-rows 0 --seed 0 --T 50 \
  --attr-g class=8 pos_spl=2 pos_obj=8 rot_obj=8
```

Inputs: the four `c3di_k1_final` checkpoints below (epochs 8/22/36 under `intermediates/`,
epoch 50 as `c3di_k1_final-step217936.ckpt`).

---

### `9a4b5c6` — loss-plateau projection, and the 75-epoch extensions

Power-law fits `L(e) = a*e^-b + c` on the first 50 epochs of each run:

| model | fit | irreducible floor | %/epoch at ep 50 | reaches 0.25%/epoch |
|---|---|---:|---:|---:|
| k=1 | `0.0038384 * e^-0.5026 + 0.00073749` | 0.000737 | 0.424% | epoch 75 |
| k=11 | `0.0027124 * e^-0.2982 + 0.00047204` | 0.000472 | 0.383% | epoch 73 |

k=11 has the lower floor but the shallower exponent — slower improvement toward a better
asymptote. Extrapolates 50% beyond observed range; `c` is the least identifiable parameter.

```bash
python experiments/hdae/scripts/loss_convergence.py          # projection from 50 epochs
python experiments/hdae/scripts/convergence_report.py        # stitched curve + fit + plot
```

Outcome: k=1's extension was **stopped at epoch 51** once the table above came in. k=11's
reached epoch 59 before a disk-full crash; epochs 53 and 59 are archived.

---

### `178ae5c` — k=11 `do(class)` is 99.32%, not 100%

The 256-image cohort produced a bootstrap CI of exactly [1.0000, 1.0000] — zero width, which
signalled a cohort too small to contain a failure rather than certainty. At n=1024 there are
7 failures at g=3, 5 at g=5, 2 at g=8. `do(class)` does **not** saturate at g=3.

```bash
python experiments/hdae/scripts/cfg_sweep_c3di.py \
  --config experiments/hdae/configs/c3di_hier_k11_final.yaml \
  --ckpt <c3di_k11_final-step192936.ckpt> --label k11_n1024 \
  --strengths 1 1.5 2 2.5 3 5 8 --weights ema --n 1024 --sample-bs 64 --seed 0 --T 50
python experiments/hdae/scripts/best_g_analysis.py --labels k1_n1024 k11_n1024
```

**T=50 is the operating point.** Verified equivalent to T=100 on an identical cohort
(k=11, g=3): class 100.00% vs 100.00%, pos_spl 0.0276 vs 0.0275, pos_obj 0.0326 vs 0.0321.
T=25 was tested and rejected (pos_obj +49%). `--fp16` buys 0% and `--compile` emits NaNs.

---

### `95ab75b` — metric definitions settled

```
CC = 1 - (E|pred-target| - floor) / (E|source-target| - floor)
FC = 1 - max(0, E|pred-source| - floor) / (best-constant-predictor error - floor)
CF1 = harmonic mean, clipped to [0,1]
```

Raw accuracy and MAE are retained beside every derived score. FC baseline is the best
**constant** predictor (`E|y-ybar|`; mode for `class`), exact over all 25,200 test rows.

```bash
python experiments/hdae/scripts/aggregate_metrics.py    # CC / FC / CF1
python experiments/hdae/scripts/metric_landscape.py     # every variant of every axis
```

---

### `dd2ac85` / tag `c3di-v2-validated-20260827` — the depth result

Each model at its own best guidance strength, 256-image cohort, EMA, T=100:

| intervention | k=1 best g | k=1 | k=11 best g | k=11 |
|---|---:|---:|---:|---:|
| do(class) | 8 | 88.28% | 3 | 100.00%* |
| do(pos_spl) | 2 | 0.0394 | 2 | **0.0214** |
| do(pos_obj) | 8 | 0.1933 | 2 | **0.0300** |
| do(rot_obj) | 8 | 0.1751 | 3 | **0.0790** |

\* superseded by `178ae5c` — the true figure is 99.32%.

**Carried caveat:** k=1 trained at lr 1e-4 (its resume restored that optimiser state), k=11 at
the configured 2e-4. Sample count (12,348,000 each), batch (64), seed (42), data and schedule
are identical. Neither model had an LR scheduler.

---

## Checkpoints in S3

All under `s3://najibi-research-7f2a/wip-from-826a88e/checkpoints/`.

### `c3di-v2-validated-20260827` — the full runs

| object | size | ETag (md5-of-parts, 8 MB chunks) |
|---|---:|---|
| `c3di-v2-validated-20260827/hdae/c3di_k11_final-step192936.ckpt` | 2,085,985,884 | `59ccfe1e53ddd2faf6267d3f2c3d5f50-249` |
| `c3di-v2-validated-20260827/hdae/c3di_k1_final-step217936.ckpt` | 2,146,025,892 | `a33046e5954dca978cd45a55003be143-256` |

### `c3di-v2-validated-20260827/intermediates` — the metric-vs-epoch curve

| object | size | ETag (md5-of-parts, 8 MB chunks) |
|---|---:|---|
| `c3di-v2-validated-20260827/hdae/intermediates/c3di_k11_final-epoch=12-step=48234.ckpt` | 2,085,985,756 | `99d432a68b1e55edd1922f1ef2bbf92c-249` |
| `c3di-v2-validated-20260827/hdae/intermediates/c3di_k11_final-epoch=25-step=96468.ckpt` | 2,085,985,884 | `a67be9d8027007486b4d874d348b602e-249` |
| `c3di-v2-validated-20260827/hdae/intermediates/c3di_k11_final-epoch=37-step=144702.ckpt` | 2,085,985,884 | `bf41192d4f7d011900e6d100f51ed0e3-249` |
| `c3di-v2-validated-20260827/hdae/intermediates/c3di_k1_final-epoch=22-step=108968.ckpt` | 2,146,025,892 | `9f17a91ddd909f0cbdb37f36015de2c7-256` |
| `c3di-v2-validated-20260827/hdae/intermediates/c3di_k1_final-epoch=36-step=163452.ckpt` | 2,146,025,892 | `3f7276ee99a91a841dc246dcab2e2a4a-256` |
| `c3di-v2-validated-20260827/hdae/intermediates/c3di_k1_final-epoch=8-step=54484.ckpt` | 2,146,025,828 | `cfd9348f3a6c31f7d20a8381b2ee0bd1-256` |

### `c3di-v2-validated-20260827/ext75_*` — the extension runs

| object | size | ETag (md5-of-parts, 8 MB chunks) |
|---|---:|---|
| `c3di-v2-validated-20260827/hdae/ext75_k1/epoch=53-step=231480.ckpt` | 2,146,025,828 | `64047ac37cf980e75dd0533697604ed0-256` |
| `c3di-v2-validated-20260827/hdae/ext75_k11/epoch=53-step=208332.ckpt` | 2,085,993,820 | `07b0f1165f4999026233c981e6809872-249` |
| `c3di-v2-validated-20260827/hdae/ext75_k11/epoch=59-step=231480.ckpt` | 2,085,993,884 | `1205254a6e738c3bb217aefcc04a3352-249` |

### `c3di-v1-20260825` — predictors, SCM, conditioning ablation

| object | size | ETag (md5-of-parts, 8 MB chunks) |
|---|---:|---|
| `c3di-v1-20260825/hdae/c3di_k1_both-step50000.ckpt` | 2,146,018,340 | `ca89747878e944e3bbf40fa6269bb606-256` |
| `c3di-v1-20260825/hdae/c3di_k1_fourier_b16-step50000.ckpt` | 2,146,010,296 | `fbcf127422d08a37c72ed75efb5c5c71-256` |
| `c3di-v1-20260825/hdae/c3di_k1_rmsnorm_b16-step42000.ckpt` | 2,145,903,652 | `459c3c2c1f3367e4824bc236cd6e2529-256` |
| `c3di-v1-20260825/predictors/class.pt` | 111,356,626 | `439efccbe8e1d59151430a6f3701bb39-14` |
| `c3di-v1-20260825/predictors/hue_bg.pt` | 111,343,436 | `3489b4b6f4aa6a4e041efa894b6aa2db-14` |
| `c3di-v1-20260825/predictors/hue_obj.pt` | 111,349,382 | `067606609d0d89e2198ec0adcfec8892-14` |
| `c3di-v1-20260825/predictors/hue_spl.pt` | 111,349,382 | `9919ab5dd584258be9eac6f722f0a192-14` |
| `c3di-v1-20260825/predictors/pos_obj.pt` | 111,355,654 | `fd43aedc44d1a7fcb56a843f660990a8-14` |
| `c3di-v1-20260825/predictors/pos_spl.pt` | 111,349,382 | `8730b78e63ab857186e3e42ded72e336-14` |
| `c3di-v1-20260825/predictors/rot_obj.pt` | 111,355,654 | `0baf94099dbd91351ab14b9b01e8c533-14` |
| `c3di-v1-20260825/scm/causal3dident_scm.pt` | 25,603 | `54add80049db321678d36b562749301b` |
| `c3di-v1-20260825/scm/causal3dident_scm_spline.pt` | 152,932 | `a67b5a1f386c73f98cdb5f75f7a267ec` |

---

## Non-checkpoint artefacts

| file | produced by | holds |
|---|---|---|
| `experiments/hdae/outputs/cfg_sweep/sweep_*.json` | `cfg_sweep_c3di.py` | every cell, raw values, cohort |
| `experiments/hdae/outputs/cfg_sweep/persample_*.npz` | `cfg_sweep_c3di.py` | **per-sample raw errors** — any aggregation is post-hoc arithmetic on these, no re-rendering |
| `experiments/hdae/outputs/cfg_sweep/best_g.json` | `best_g_analysis.py` | optima + paired-bootstrap tied sets |
| `experiments/hdae/outputs/cfg_sweep/aggregate_metrics.json` | `aggregate_metrics.py` | CC / FC / CF1 with raw values |
| `experiments/hdae/outputs/cfg_sweep/metric_landscape.json` | `metric_landscape.py` | every metric-design variant |
| `experiments/hdae/outputs/attr_predictors_c3di/*.pt` | `train_causal3dident_attr_predictors.py` | the 7 measurement instruments |

Predictor state dicts carry a `net.` prefix from their wrapper class and must be stripped
before loading into a bare `convnext_tiny` — see `load_predictor` in `cfg_sweep_c3di.py`.

---

## Published pages

| page | covers |
|---|---|
| https://claude.ai/code/artifact/98521430-a4cb-47df-9e86-637e9341cb70 | metric design space, every variant |
| https://claude.ai/code/artifact/36b064b7-4322-48fd-b2b5-2bb18895101e | best guidance per attribute |
| https://claude.ai/code/artifact/642090d4-e3d6-42f7-b505-a1384f3f8814 | k=1 vs k=11 comparison |
| https://claude.ai/code/artifact/de75f677-0713-4b1d-9c92-dc95fa128554 | k=11 guidance sweep |
| https://claude.ai/code/artifact/494f4045-133b-4070-9818-db853aa7c945 | k=1 guidance sweep |

---

## Environment

Two A100-SXM4-40GB nodes; see `cluster.yaml` for hosts and the bootstrap recipe.
torch 2.2.2+cu121, torchvision 0.17.2, pytorch-lightning 1.9.5, Python 3.11 via `uv`.
Dataset: Causal3DIdent, Zenodo 10.5281/zenodo.4784282, packed to 128x128 HDF5 by
`scripts/build_causal3dident.py` (252,000 train / 25,200 test).

---

## Complete results bundle — Causal3DIdent depth ladder

`s3://najibi-research-7f2a/wip-from-826a88e/results/c3di-ladder-20260904/`

Self-contained: `RESULTS.md` carries the full metric definitions (CC, FC_observed,
FC_unobserved, CF1, AUC), the protocol, the headline tables and the caveats. `metrics_all.csv`
is the flat table — 84 rows, one per (model, intervention, guidance strength), with CC, FC in
all three pools, all three CF1 variants, and the raw predictor reading plus role for each of the
seven attributes. `raw/persample_k*_n1024.npz` holds the per-sample errors, so any change to a
metric definition is arithmetic on those files with nothing re-rendered.

Headline: depth saturates at five taps. Mean CF1 at each model's own best g is 0.8399 (k=1),
0.9507 (k=5), 0.9545 (k=11) — one tap to five is worth +0.1108, five to eleven +0.0038.
