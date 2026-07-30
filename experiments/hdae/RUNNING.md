# RUNNING — environment, commands, and reproducing current results

Practical companion to `AGENDA.md` (architecture/orientation). This file is
the "what do I actually type" reference, and the reproducibility record for
whoever checks out this repo later.

## 0. If you just checked out this commit and don't know where to start

1. Read `AGENDA.md` first — architecture, what's live vs. dead code, current
   in-flight state.
2. **Trained checkpoints, `experiments/hdae/data/packed/*`, and everything
   under `experiments/hdae/outputs/` are NOT in git** (`.gitignore` excludes
   `outputs/`, `data/**/*`, `*.ckpt`; `checkpoints/` at repo root is also
   gitignored — see §3). They only exist on the machine that produced them.
   If you're on that machine, §3 tells you where to find them. If you're
   anywhere else, or they're gone, §2 tells you how to regenerate everything
   from scratch — the whole pipeline is designed to be restartable and to
   skip stages whose outputs already exist.
3. `diffae_latent_probe/data/` (raw CelebA-HQ images/labels, ~19 GB) is the
   one input nothing here can regenerate for you — it has to already be on
   disk at that path (or every config in `experiments/hdae/configs/`
   repointed at wherever you put it) before step 1 of the pipeline can run.
4. `TODO-List` (repo root) has the planned next steps (model-agnostic CF
   contract, causal-graph normalizing flows, new datasets, CF1 decomposition)
   with a plan under each item — that's the actual roadmap, not this file.

## 1. Environment

```bash
# repo root venv already has upstream DiffAE deps; this adds what HDAE needs
pip install -r experiments/hdae/requirements.txt   # PyYAML, lmdb, Pillow
```

Upstream DiffAE (`diffae_upstream/`) must already be importable — HDAE code
does `sys.path.append("/home/anajibi/HDM/diffae_upstream")` and imports
`templates`, `choices`, `experiment`, `model.*` directly from it. That path is
hardcoded in `hdae/config_io.py`; update it if the repo ever moves.

## 2. Full pipeline, one command

```bash
python experiments/hdae/scripts/run_full_pipeline.py \
  --config experiments/hdae/configs/hier_k5.yaml \
  --attribute Smiling --pcf-guidance-scale 8.0
```

Runs, in order, skipping any stage whose declared outputs already exist
(`--force` to redo a stage): preprocess -> train -> reconstruct -> extract
latents -> train linear probes -> analyze probes -> swap/null grid -> abduct
grid -> build cohorts -> counterfactual eval -> PCF eval. Swap `hier_k5.yaml`
for `hier_k1.yaml` / `hier_k11.yaml` for the other trained hierarchy sizes
(K = 1 / 5 / 11 levels). **Do not use `celeba64_*.yaml` configs — they no
longer load with the current code, see `AGENDA.md` §4.**

**The PCF stage always re-runs** (`run_full_pipeline.py` passes `force=True`
for it specifically) — re-invoking this on `hier_k5`/`hier_k11` overwrites
`pcf_per_intervention.csv`/`pcf_aggregate.csv` and will currently crash
before finishing (see §3, the `FC_success` `KeyError`). Fix
`save_frontier_plot()` first, or expect a non-zero exit after the CSVs are
already (over)written.

## 3. Current trained checkpoints (this machine, as of 2026-07-30)

| Config | Checkpoint | Status |
|---|---|---|
| `hier_k1.yaml` (K=1, flat) | `experiments/hdae/outputs/hier_k1/checkpoints/last.ckpt` | trained; reconstruction/latent-probing/PCF outputs complete |
| `hier_k5.yaml` (K=5) | `experiments/hdae/outputs/hier_k5/checkpoints/last.ckpt` | trained; last PCF run 2026-07-16 via `playground.bash` (`run_k5.log`) — `pcf_aggregate.csv`/`pcf_per_intervention.csv` written, then crashed in `save_frontier_plot()`: `KeyError: 'FC_success'` (stale key from the raw/corrected refactor, see AGENDA.md §8). Not currently running (nothing in `ps aux`). |
| `hier_k11.yaml` (K=11) | `experiments/hdae/outputs/hier_k11/checkpoints/last.ckpt` | trained; same 2026-07-16 run/crash, `run_k11.log`. |

`checkpoints/celeba64d2c_autoenc/` at the repo root (gitignored) is an unrelated upstream
pretrained-DiffAE demo download (sample grids only, no weights) — not used by
HDAE training (`scripts/train.py` never loads it; `base_template:
celeba64d2c_autoenc` in the YAML configs only selects an upstream
architecture/hyperparameter template function, not a checkpoint — verified in
`diffae_upstream/templates.py::celeba64d2c_autoenc`, which never sets
`conf.pretrain`).

## 4. Individual pipeline stages

```bash
# 0. preprocess ONCE per dataset: resize CelebA-HQ -> packed LMDB + aligned attrs
python experiments/hdae/scripts/preprocess_data.py --config experiments/hdae/configs/hier_k5.yaml

# 1. train (resumes automatically from outputs/<name>/checkpoints/last.ckpt if present)
python experiments/hdae/scripts/train.py --config experiments/hdae/configs/hier_k5.yaml

# 2. reconstruction eval
python experiments/hdae/scripts/reconstruct.py --config experiments/hdae/configs/hier_k5.yaml --ckpt <path/to/last.ckpt>
```

### Latent probing

```bash
# extract per-level semantic latents from a checkpoint
python experiments/hdae/latent_probing/extract_latents.py \
  --config experiments/hdae/configs/hier_k5.yaml --ckpt <ckpt> \
  --output experiments/hdae/outputs/hier_k5/latent_probing/latents.npz

# one linear binary classifier per (latent level, CelebA attribute) pair
# K levels x 40 attributes -> K*40 classifiers, one metrics row each
python experiments/hdae/latent_probing/train_linear_probes.py \
  --latents experiments/hdae/outputs/hier_k5/latent_probing/latents.npz \
  --output-dir experiments/hdae/outputs/hier_k5/latent_probing/probes
```
Outputs: `latents.npz` (`z_level_0..K-1`, `attrs`, `partitions`, `indices`,
`attribute_names`), `probe_metrics.csv` (val/test accuracy + balanced
accuracy per classifier), `weights/*.pt` (one serialized linear classifier
per level/attribute + standardization stats), `summary.json`. Falls back to a
deterministic 80/10/10 split if the packed dataset's partition labels are
incomplete.

```bash
# heatmaps + best-level summaries from probe_metrics.csv
python experiments/hdae/latent_probing/analyze_probe_results.py \
  --probe-metrics experiments/hdae/outputs/hier_k5/latent_probing/probes/probe_metrics.csv \
  --output-dir experiments/hdae/outputs/hier_k5/latent_probing/analysis
```
Writes `probe_heatmap.png`, `best_level_counts.png`, `best_level_by_attribute.csv`, `analysis_summary.json`.

### Learned null-token ablations

Each HDAE latent level owns a learned null token (model parameter, saved in
checkpoints); training randomly substitutes it per-level with probability
`conditioning.latent_drop_prob`. At eval time, force specific levels to null
to see what the decoder does without them:

```bash
python experiments/hdae/latent_probing/reconstruct_with_nulls.py \
  --config experiments/hdae/configs/hier_k5.yaml --ckpt <ckpt> \
  --null-levels 0,2 \
  --output experiments/hdae/outputs/hier_k5/null_levels_0_2.png
```

### Swap / null diagnostic grid and abduction reveal grid

```bash
# source/donor rows, every single-level swap, every adjacent-pair swap, every single-level null
python experiments/hdae/latent_probing/swap_null_grid.py \
  --config experiments/hdae/configs/hier_k5.yaml --ckpt <ckpt> \
  --output experiments/hdae/outputs/hier_k5/latent_probing/swap_null_grid.png

# abduct Z and x_T once; decode all-null, forward-cumulative (Z0, Z0+Z1, ...),
# and reverse-cumulative (Z-1, Z-1+Z-2, ...) rows, unrevealed levels forced to null
python experiments/hdae/latent_probing/abduct_xt_z_grid.py \
  --config experiments/hdae/configs/hier_k5.yaml --ckpt <ckpt> \
  --output experiments/hdae/outputs/hier_k5/latent_probing/abduct_xt_z_grid.png
```
Both work for any configured K.

### Counterfactual / PCF evaluation

```bash
# model-agnostic fixed image cohorts (pos/neg per attribute) so every model is scored
# on identical images — cohorts.yaml controls num_images/seed, shared across all configs
python experiments/hdae/build_cohorts.py \
  --attr-npz experiments/hdae/data/packed/celebahq_64_attrs.npz \
  --attributes Smiling,Eyeglasses,Male,Young \
  --num-images 1024 --seed 0 \
  --output experiments/hdae/outputs/shared_cohorts/celeba_hq_conditioning_cohorts.json

# CC / FC / PCF eval (raw + prevalence-corrected variants, see AGENDA.md §8-9 for metric definitions)
python experiments/hdae/counterfactuals/run_pcf_eval.py \
  --config experiments/hdae/configs/hier_k5.yaml --ckpt <ckpt> \
  --attr-classifier experiments/hdae/outputs/finetuned_attr_classifier.pt \
  --cohorts experiments/hdae/outputs/shared_cohorts/celeba_hq_conditioning_cohorts.json \
  --output-dir experiments/hdae/outputs/hier_k5/counterfactuals/pcf \
  --model-name hier_k5 --guidance-scale 8.0
```
Writes `pcf_per_intervention.csv`, `pcf_aggregate.csv` (macro/micro/weighted x
raw/corrected `PCF`), `pcf_experiments_grid.png`, `correlation_baseline.json`
(cached natural co-occurrence baseline, reused across guidance-scale sweeps
unless `--baseline-cache` points elsewhere).

**Known-broken right now:** `save_frontier_plot()` in this file still reads
`r["FC_success"]`, a key the raw/corrected refactor renamed to
`FC_success_raw`/`FC_success_corr`. The two CSVs above are written first and
are valid; the frontier PNG and the process exit code are not — confirmed by
the 2026-07-16 crash in `run_k5.log`/`run_k11.log` (§3). Fix the key
reference (pick `_raw` or `_corr`, or plot both) before trusting this path
end-to-end.

**Known-broken, don't run:** `run_cf_consistency.py` and
`diagnose_xt_dominance.py` both `import
experiments.hdae.counterfactuals.attribute_classifier`, a module that doesn't
exist (the live one is `attr_classifier.py`). Unmaintained since a rename;
fix the import or delete before trusting either script.

## 5. Config schema (only the live `hier_k*.yaml` shape — see AGENDA.md §4 for why `celeba64_*.yaml` is dead)

| Field | Meaning |
|---|---|
| `base_template` | Upstream `templates.py` function selecting base diffusion/network hyperparameters (image size, channel mult, etc.) — not a checkpoint. |
| `data.image_dir`, `attr_path`, `partition_path` | Raw CelebA-HQ image dir + attribute/split label files (currently under `diffae_latent_probe/data/`, see AGENDA.md §5). |
| `data.image_size` | Packed resolution and model resolution. |
| `data.lmdb_path`, `attr_npz` | Packed-LMDB output dir and aligned attribute array output (both build artifacts, regenerated by `preprocess_data.py`). |
| `data.flip_aug` | Train-only random horizontal flip. |
| `data.resize_filter` | `bicubic` or `lanczos`, used only while packing. |
| `encoder.type` | `hierarchical` (the only value the current code path exercises). |
| `encoder.hier_tap_block_ids` | Encoder block indices (or `"mid"`) tapped for each latent level, coarse-to-fine. |
| `encoder.hier_level_dims` | Latent width per level, one-to-one with taps. |
| `encoder.hier_block_to_level` | Maps each decoder output block to which latent level styles it. |
| `encoder.n_decoder_output_blocks` | Must match the decoder's actual output-block count. |
| `encoder.n_attributes` | Number of conditioning attributes (must equal `len(conditioning_attrs)`). |
| `encoder.conditioning_attrs` | CelebA attribute names used as explicit conditioning input, e.g. `[Smiling, Eyeglasses, Male, Young]`. |
| `encoder.attr_embed_dim` | Attribute embedding width. |
| `encoder.attr_dropout_prob` | Per-attribute dropout probability during attribute embedding. |
| `encoder.attr_input_range` | `pm1` or `auto` — raw attribute value range, converted to index space by `hdae/attr_utils.py::to_index_space`. |
| `encoder.hier_proj` | `linear` (head type on top of pooled taps). |
| `conditioning.strategy` | `per_block_attr` (the only path `HierarchicalAutoencModel` builds — no branch on this value). |
| `conditioning.style_ch` | Decoder semantic conditioning width. |
| `conditioning.latent_drop_prob` | Per-sample, per-level probability of substituting the learned null token during training. |
| `conditioning.cfg_drop_prob` | Per-sample probability of dropping all modeled attributes to the null attribute token, for classifier-free guidance training. |
| `conditioning.cfg_guidance_scale` | Default attribute-CFG inference scale used by PCF eval when no `--guidance-scale` CLI override is given. |
| `train.batch_size_per_gpu`, `total_batch_size` | Local/global batch (global must equal local x devices). |
| `train.lr`, `ema_decay`, `T`, `T_eval` | LR, EMA decay, train diffusion steps, DDIM eval steps. |
| `train.max_steps`, `precision`, `grad_clip`, `num_workers` | Trainer/runtime settings. |
| `train.compile` | Reserved; false by default. |
| `lightning.*` | Passed to the Lightning `Trainer`; production configs use 2-GPU DDP. |
| `seed`, `output_dir` | Reproducibility seed and artifact root (everything under `output_dir` is gitignored — see §0). |

## 6. Tests

There is no test suite for `experiments/hdae` right now (no `tests/`
directory, no `smoke_test.py`). The archived predecessor's tests
(`archive/diffae_latent_probe/tests/`) test that project's code, not this
one.
