# Checkpoint release `c3di-v1-20260825`

Everything needed to reproduce or reuse the Causal3DIdent work as of 2026-08-25.
Git commit: `4199e504ab45` on branch `wip-from-826a88e` (also tagged `c3di-v1-20260825`).

S3: `s3://najibi-research-7f2a/wip-from-826a88e/checkpoints/c3di-v1-20260825/`

## Quick start

```bash
git checkout c3di-v1-20260825
aws s3 sync s3://najibi-research-7f2a/wip-from-826a88e/checkpoints/c3di-v1-20260825/ ./restore/
```

The dataset is NOT in this release (13.6 GB). Rebuild it from Zenodo -- it is deterministic:

```bash
# downloads 9.1 GB, streams straight to 128x128 HDF5, never unpacks the PNGs
curl -sSLo trainset.tar.gz "https://zenodo.org/records/4784282/files/trainset.tar.gz?download=1"
curl -sSLo testset.tar.gz  "https://zenodo.org/records/4784282/files/testset.tar.gz?download=1"
# md5: trainset acd98fda30eee75856dbbc7c54a27e45   testset c5d9d32d3737e241a2b12b968275fcb8
tar xzf testset.tar.gz  --wildcards '*/*.npy'      # the .npy latents must exist before packing
tar xzf trainset.tar.gz --wildcards '*/*.npy'
python experiments/hdae/scripts/build_causal3dident.py
```

---

## 1. Attribute predictors -- `predictors/`

ConvNeXt-Tiny (ImageNet-pretrained), **one model per attribute**, 8 epochs, batch 64,
trained on 246,960 images, metrics below are on the untouched 25,200-image testset.

These are the measurement instruments for CC / FC / CF1. They must be sharper than the
generator being scored: the earlier MorphoMNIST work stalled because its predictor's error
(MAE 0.115) was **2x wider than the tolerance window** (0.055), so model failure and
instrument noise were indistinguishable.

| file | attribute | kind | test metric | R2 | sha256[:16] |
|---|---|---|---|---|---|
| `predictors/class.pt` | class | categorical (7) | accuracy 99.996% | - | `671a2190d9c2cb13` |
| `predictors/hue_bg.pt` | hue_bg | scalar | MAE 0.00894 | 0.99840 | `f7b2560ae9e14863` |
| `predictors/hue_obj.pt` | hue_obj | scalar | MAE 0.07382 | 0.94697 | `32c9770d22aa9252` |
| `predictors/hue_spl.pt` | hue_spl | scalar | MAE 0.04205 | 0.98171 | `bdaa86499d46b877` |
| `predictors/pos_obj.pt` | pos_obj | scalar | MAE 0.01128 | 0.99899 | `86d37d24d7ede77c` |
| `predictors/pos_spl.pt` | pos_spl | scalar | MAE 0.00991 | 0.99941 | `0faac996a40d5a13` |
| `predictors/rot_obj.pt` | rot_obj | scalar | MAE 0.01856 | 0.99388 | `78d1fc60463a673f` |

**Load one:**
```python
import torch
from torchvision.models import convnext_tiny
blob = torch.load("predictors/pos_obj.pt", map_location="cpu")
m = convnext_tiny()
m.classifier[2] = torch.nn.Sequential(torch.nn.Dropout(0.2),
                                      torch.nn.Linear(768, blob["out_dim"]))
m.load_state_dict(blob["state_dict"]); m.eval()
# blob also carries: attr, kind, cols (dataset columns), img_size, val_metrics, test_metrics, args
```
Inputs are 128x128, ImageNet-normalised: `x = (img_in_[-1,1] + 1)/2`, then subtract
mean `[0.485,0.456,0.406]` / divide std `[0.229,0.224,0.225]`.

**Caveat worth carrying forward:** `hue_obj` is the weakest at R2 0.947 (MAE 0.0739, ~14%
of its target sd) because object hue lives in a small number of pixels on a small object,
while `hue_bg`/`hue_spl` paint large regions. Do not treat all seven as equally sharp --
`hue_obj` carries roughly 3x the measurement noise of the position attributes. Retraining
that one at `--img-size 224` is the obvious improvement if it matters.

---

## 2. Attribute SCM -- `scm/`

Structural causal model over the 4 modelled attributes. Graph:
`class -> rot_obj`, `class -> pos_obj`, `pos_spl -> pos_obj`. Roots: `class` (categorical,
7), `pos_spl`. `pos_obj`/`rot_obj` are 3-D vector nodes. Unmodelled: the 3 hues.

| file | mechanism | test NLL | note |
|---|---|---|---|
| `scm/causal3dident_scm_spline.pt` | conditional rational-quadratic spline over a uniform base | +5.2910 | **USE THIS ONE** |
| `scm/causal3dident_scm.pt` | conditional diagonal Gaussian | +5.8832 | superseded, kept for the comparison |

The Gaussian matched every mean and variance but emitted bell shapes against uniform data,
putting **4.85% of samples outside the data's [-1,1] bounds**; 4x the training budget did
not move that, so it was the density family, not undertraining. The spline drives
out-of-bounds to **0.00%** and mean 1-Wasserstein distance from 0.0450 to 0.0056.

```python
import sys; sys.path.insert(0, "experiments/hdae/causal")
from train_scm_causal3dident import SCM, CausalGraph
import torch
b = torch.load("scm/causal3dident_scm_spline.pt", map_location="cpu")
c = b["config"]
scm = SCM(CausalGraph(c["attributes"], c["edges"]), c["nodes"],
          mechanism=b["mechanism"], bins=b["bins"])
scm.load_state_dict(b["state_dict"]); scm.eval()
cf = scm.counterfactual(obs_attrs, {"class": torch.full((1,1), 2.0)})   # descendants propagate
```

---

## 3. HDAE conditioning ablation -- `hdae/`

Three arms, all **k=1**, differing only in how continuous attributes are conditioned.
Motivation: continuous attributes scored CC 0.21-0.34 against categorical 0.65-0.99 at the
same guidance, because a bare `Linear(1,d)` maps an attribute's whole range onto a
near-affine curve so the CFG delta `e(target)-e(null)` is tiny. Measured at init on this
attribute set, the embedding displacement for a class flip vs a small `pos_spl` change
differed by **~1208x**; Fourier alone closes it to ~14x, Fourier+RMSNorm to ~1.7x.

| file | arm | fourier / norm | steps | batch / lr | PSNR | LPIPS | dL_all | cfg_delta |
|---|---|---|---|---|---|---|---|---|
| `hdae/c3di_k1_fourier_b16-step50000.ckpt` | A | 16 / off | 50,000 | 16 / 5e-5 | 32.57 | 0.0226 | 0.00078 | 0.0155 |
| `hdae/c3di_k1_rmsnorm_b16-step42000.ckpt` | B | 0 / on | 42,000 | 16 / 5e-5 | 30.38 | 0.0204 | 0.00173 | 0.0265 |
| `hdae/c3di_k1_both-step50000.ckpt` | C | 16 / on | 50,000 | 32 / 1e-4 | 37.74 | 0.0106 | 0.00363 | 0.0371 |

`dL_all` = L(attrs nulled) - L(true attrs): what the conditioning signal is worth to the
denoiser. `cfg_delta` = ||eps(cond)-eps(null)|| / ||eps(null)||.

**How to read this table.** C wins every axis, but C also saw **2x the images of A** and
2.4x of B, so its reconstruction lead is substantially training volume. The load-bearing
result is **A vs B** -- matched batch, lr, seed -- where **B beat A on conditioning by 2.2x
from a 16% step deficit**. RMSNorm carries more conditioning value than Fourier features
do. That is the opposite of what the init-time displacement measurement predicted (it
measured achievable range, not what training exploits). Arm C was selected for the
full-length runs because it contains both.

**Arm B caveat:** trained by two processes concurrently for part of its life (a supervisor
bug launched it twice into the same output directory). Weights were never blended -- each
process held its own model and wrote whole files -- but its intermediate checkpoints are of
ambiguous provenance. The step-42000 file archived here was written by the surviving
process and loads cleanly.

```python
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
cfg = load_hdae_config("experiments/hdae/configs/c3di_hier_k1_both.yaml", require_data=False)
m = HDAELitModule.load_from_checkpoint("hdae/c3di_k1_both-step50000.ckpt",
                                       conf=cfg.train_conf, map_location="cpu")
# m.ema_model is what you want for sampling; m.model is the raw training weights
```

Each `hdae/*.ckpt` pairs with the config named in its filename, under
`experiments/hdae/configs/`. Configs are in git at this tag -- do not guess them, the
`fourier_freqs`/`attr_norm` fields change the module shapes and a mismatch fails to load.

---

## 4. Configs -- `configs/`

Copied here so the release stands alone even if git is unavailable. Identical to the
repo at commit `4199e504ab45`.

## Gotchas

* `max_steps` must be RAISED above a resumed checkpoint's `global_step`, or `train.py`
  restores, sees the limit already met, and exits without training a step.
* torch 2.2.2+cu121 supports up to sm_90. A newer GPU needs a newer torch, which
  pytorch-lightning 1.9.5 may not tolerate.
* `torch.compile` is applied to a side handle, never to `self.model` -- see
  `HDAELitModule.enable_compile`. Compiling the module directly puts `_orig_mod.` into
  every state_dict key (measured: 760 of 2281), breaking both EMA and checkpoint reload.
* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is required at batch 64 on a 40 GB A100.

