# AGENDA — orientation for agents working on `experiments/hdae`

Read this before touching the code. It exists so a fresh agent doesn't have to
re-derive repo structure, stale-vs-live scripts, and the current experiment
state from scratch. Update it when you learn something a cold agent would
otherwise waste time rediscovering — this file rots fast if left alone.

## 1. What this experiment is

HDAE = **H**ierarchical **D**iffusion **A**uto**E**ncoder. It is a from-scratch
retrain of DiffAE where the semantic encoder is replaced by a small BeatGANs
down-path with taps at several resolutions, producing `K` separate semantic
latent chunks `z_0 .. z_{K-1}` (coarse → fine) instead of DiffAE's single
`z_sem`. The DDPM decoder, diffusion objective, DDIM sampler, EMA, and
optimizer are all unchanged from upstream. The scientific question is whether
splitting the semantic code into a resolution-ordered hierarchy makes
individual attributes more linearly separable/editable at some levels than
others, and whether editing one level lets you change one attribute while
preserving the rest better than editing a flat `z_sem`.

This branch (`new-arch`) also adds **class/attribute conditioning**: the
model is trained to take 4 CelebA attributes (`Smiling, Eyeglasses, Male,
Young` in current configs) as an explicit conditioning signal via
`per_block_attr` styling plus classifier-free guidance (CFG), on top of the
hierarchical latents. An earlier "phase 1" `concat_proj` design (no attribute
conditioning) existed before this but its config schema and merger module are
now dead code — see §4, don't try to run it.

Predecessor project: **`archive/diffae_latent_probe/`** (see §7). It probed a
*frozen, pretrained, non-hierarchical* DiffAE checkpoint (single `z_sem` +
`x_T`, linear-probe attribute directions, `z_edit = z_sem + alpha * w`). HDAE
generalizes the same probing/pseudo-counterfactual methodology to a
hierarchical latent that is trained (not frozen) from scratch.

## 2. Directory map

```
experiments/hdae/
  hdae/                  # model + training package
    hier_encoder.py        HierarchicalSemanticEncoder: BeatGANs down-path + taps -> [z_0..z_{K-1}]
    hier_autoenc.py         HierarchicalAutoencModel(BeatGANsAutoencModel): wires encoder + per-block styling into the (frozen-architecture) decoder
    hier_config.py          Typed dataclasses: EncoderHierarchyConfig, ConditioningConfig, HDAEConfig (the only schema `config_io.py` can load, matches hier_k*.yaml)
    conditioning.py         ConcatProjectionMerger: learned per-level null tokens + latent_drop_prob. DEAD CODE — not imported anywhere else in experiments/hdae. Leftover from the pre-attribute-conditioning "concat_proj" design; see §4.
    attr_conditioner.py      AttributeEmbedding, PerBlockStyle: CFG-droppable attribute embedding + per-decoder-block style injection (the only conditioning path `HierarchicalAutoencModel` builds)
    attr_utils.py            to_index_space / observed_unique — raw attribute value <-> model index-space conversion (pm1 vs "auto" ranges)
    lit_module.py            HDAELitModule(experiment.LitModel): training_step wires cond={"zs","y_idx"} into upstream diffusion losses
    grid_utils.py            save_labeled_grid — shared image-grid rendering used by nearly every visualization script
    null_tokens.py, config_io.py
  data/                   CelebAHQPacked dataset, attribute loading, LMDB packing (data/packed/*.lmdb, *_attrs.npz — build artifacts, not source data)
  configs/                YAML experiment configs — see §4: `hier_k*.yaml` is live, `celeba64_*.yaml` no longer loads
  scripts/                train.py, reconstruct.py, preprocess_data.py, run_full_pipeline.py (orchestrator), evaluate_architectures.py, compare_configs.py
  latent_probing/         extract_latents.py, train_linear_probes.py, analyze_probe_results.py, swap_null_grid.py, abduct_xt_z_grid.py, reconstruct_with_nulls.py — commands in RUNNING.md §4
  counterfactuals/        run_pcf_eval.py (current, active), run_counterfactual_eval.py, run_preservation_sweep.py, attr_classifier.py, finetune_attr_classifier.py, run_swap_eval.py
  build_cohorts.py        model-agnostic fixed image cohorts (pos/neg per attribute) so all models are scored on identical images
  cf_aggregate.py         aggregates run_cf_consistency.py CSVs into a canonical table
  run_cf_consistency.py, diagnose_xt_dominance.py   STALE — see §6, broken import
  outputs/                per-config run artifacts (gitignored): outputs/<config-name>/{checkpoints,reconstruction,latent_probing,counterfactuals,logs}
  AGENDA.md               this file
  RUNNING.md              environment setup, exact commands per pipeline stage, current checkpoint inventory, config schema table
```

`diffae_upstream/` (repo root) is the vendored original DiffAE codebase
(unchanged). HDAE imports from it directly (`from diffae_upstream.model...`,
`from experiment import LitModel`, `from choices import TrainMode`, etc.) —
don't fork it, extend via the `hdae/` package instead.

## 3. One-command pipeline

```bash
python experiments/hdae/scripts/run_full_pipeline.py \
  --config experiments/hdae/configs/hier_k5.yaml \
  --attribute Smiling --pcf-guidance-scale 8.0
```

`run_full_pipeline.py` is a transparent orchestrator: preprocess → train →
reconstruct → extract latents → train linear probes → analyze probes →
swap/null grid → abduct grid → build cohorts → counterfactual eval → PCF
eval. Every stage is skipped if its declared output files already exist
(pass `--force` to redo a stage), so it's safe to re-invoke after a crash or
when only editing a downstream stage.

## 4. Config schema — `celeba64_*.yaml` configs are dead, don't run them

`celeba64_flat.yaml`, `celeba64_hier_k3.yaml`, `celeba64_hier_k5*.yaml` use
an **older** schema (`encoder.type/tap_resolutions/level_dims/pool/proj`,
`conditioning.strategy: concat_proj`). This is not a second live path — it no
longer loads. `hdae/config_io.py::load_hdae_config` does
`EncoderHierarchyConfig(**raw["encoder"])`, and that dataclass (in
`hier_config.py`) has no `tap_resolutions`/`level_dims`/`pool`/`proj` fields
at all (it has `hier_tap_block_ids`/`hier_level_dims`/`hier_block_to_level`
instead) — passing a `celeba64_*.yaml` file raises `TypeError: unexpected
keyword argument`. Confirmed by grep: `ConcatProjectionMerger`
(`conditioning.py`) is defined and has a factory method but is never
imported or constructed anywhere else in `experiments/hdae/`. Treat
`celeba64_*.yaml` and `hdae/conditioning.py` as dead-together legacy from
before attribute conditioning was added.

The **only schema the code can load** is `hier_k1.yaml` / `hier_k5.yaml` /
`hier_k11.yaml` (K = number of hierarchy levels): `encoder.hier_tap_block_ids
/hier_level_dims/hier_block_to_level/conditioning_attrs/attr_embed_dim` and
`conditioning.strategy: per_block_attr`, matching `hdae/hier_config.py`'s
dataclasses field-for-field (`HierarchicalAutoencModel.__init__` builds
`PerBlockStyle` unconditionally — there's no branch on `strategy`). Full
field-by-field reference: `RUNNING.md` §5.

If you add a new config, copy `hier_k5.yaml`, not `celeba64_hier_k3.yaml`.

## 5. Data path dependency (do not break this)

`data.image_dir` / `data.attr_path` / `data.partition_path` in **every**
`hier_k*.yaml` / `celeba64_*.yaml` config point at
`diffae_latent_probe/data/...` — raw CelebA-HQ images and label files
(~19 GB, untracked/gitignored). This directory was **not** moved when
`diffae_latent_probe`'s code was archived (see §7); it still lives at
`diffae_latent_probe/data/`. If you ever relocate or rename it, update every
config in `experiments/hdae/configs/`. `data/packed/*.lmdb` and
`*_attrs.npz` are derived build artifacts produced by
`scripts/preprocess_data.py` from that raw data — safe to delete/regenerate.

## 6. Known-stale / broken scripts

- `run_cf_consistency.py` and `diagnose_xt_dominance.py` both
  `import experiments.hdae.counterfactuals.attribute_classifier`, but that
  module doesn't exist — the live module is `attr_classifier.py`
  (`load_classifier`). These two scripts will `ImportError` as committed;
  treat them as unmaintained/pre-rename until someone fixes or removes them.
  `run_pcf_eval.py` (the live counterfactual eval path) correctly imports
  `attr_classifier`.
- No test suite exists for this experiment (no `tests/`, no `smoke_test.py`
  under `experiments/hdae/`; RUNNING.md §6).

## 7. `diffae_latent_probe` was archived

The predecessor experiment's **code** (`diffae_tools/`, `src/`, `scripts/`,
`experiments/*.py`, `tests/`, `configs/experiments/`,
`PRESENTATION_CONTEXT.md`, `requirements_extra.txt`) was moved to
`archive/diffae_latent_probe/` at the repo root because nothing under
`experiments/hdae/` imports it and it operated on a frozen pretrained DiffAE
checkpoint rather than training one. `archive/diffae_latent_probe/
PRESENTATION_CONTEXT.md` is a thorough writeup of that project if you need
the earlier methodology for comparison. **Its `data/` directory was left in
place** at `diffae_latent_probe/data/` — see §5, it's still the live raw-data
source for `experiments/hdae`.

## 8. Current in-flight state (as of 2026-07-30, branch `new-arch`)

Uncommitted working-tree changes at the time this file was written:

- `experiments/hdae/counterfactuals/run_pcf_eval.py`: PCF metric is being
  split into **raw** vs **prevalence-corrected** variants
  (`FC_success_raw/corr`, `PCF_raw/corr`, `macro/micro/weighted_PCF_raw/corr`)
  — corrected subtracts each unmodeled attribute's expected baseline flip
  rate (`correlation_baseline.json`, built by `compute_baselines_and_weights`)
  before scoring factual consistency, so attributes that are *correlated*
  with the intervened one don't unfairly tank the preservation score. Also
  hoisted `torch.compile(module.ema_model)` out of the per-batch
  loop (was recompiling every batch — likely a real perf bug fix, not WIP).
  **This refactor is currently broken**: `save_frontier_plot()` (line ~195)
  still indexes the pre-refactor key `r["FC_success"]`, which no longer
  exists (renamed to `FC_success_raw`/`FC_success_corr`). Confirmed by
  `run_k5.log`/`run_k11.log` (repo root, from the 2026-07-16 run described
  below): both runs crashed with `KeyError: 'FC_success'` at that line.
  `pcf_per_intervention.csv` and `pcf_aggregate.csv` are written *before*
  that call and are valid/complete; only the frontier PNG and the
  `run_full_pipeline.py` process's exit code are affected. Fix before
  relying on the frontier plot or on `run_full_pipeline.py` exiting 0 for
  `hier_k5`/`hier_k11`.
- `experiments/hdae/configs/cohorts.yaml`: `num_images` raised 256 → 1024
  (bigger fixed eval cohort per attribute/direction).
- `playground.bash`: launches `run_full_pipeline.py` for `hier_k5.yaml` and
  `hier_k11.yaml` in parallel in the background (`run_k5.log`,
  `run_k11.log` at repo root), with a `hier_k1` invocation commented out
  (already has a completed checkpoint under `outputs/hier_k1/`). **Not
  currently running** — both logs are from 2026-07-16 and ended in the
  crash above; nothing matches `run_full_pipeline` in `ps aux` right now.
  Re-running as-is will hit the same crash after writing valid CSVs.
- `TODO-List` (repo root, staged): near-term direction is (1) a
  model-agnostic contract for counterfactual-image generation so many
  architectures can be evaluated uniformly, (2) adding a Normalizing-Flow
  causal-graph model on top of/alongside HDAE, (3) bringing in MorphoMNIST
  and Causal3DIdent datasets, (4) splitting "CF1" into observed/unobserved/
  aggregated variants, (5) adding CheXpert. None of this is implemented yet
  — HDAE + PCF eval is still the whole implemented pipeline.
- `checkpoints/` at repo root (untracked) — a `celeba64d2c_autoenc` upstream
  pretrained checkpoint download, unrelated to HDAE's own
  `outputs/<config>/checkpoints/`.

Trained/evaluated so far: `hier_k1`, `hier_k5`, `hier_k11` (K=1/5/11 levels)
all have checkpoints and completed reconstruction/latent-probing/
counterfactual/PCF outputs under `experiments/hdae/outputs/<name>/`; the
`celeba64_*` (old-schema) configs cannot be run with the current code (§4).

## 9. Confirmed upstream (`diffae_upstream/`) integration points

Read this before touching model internals — it's the exact contract this
branch relies on staying stable in the vendored upstream code:

- Autoencoder model/config: `BeatGANsAutoencModel` and
  `BeatGANsAutoencConfig` in `model/unet_autoenc.py`.
- Semantic encoder model/config: `BeatGANsEncoderModel` and
  `BeatGANsEncoderConfig` in `model/unet.py`. Builds `input_blocks`,
  `middle_block`, then the `adaptivenonzero` pool/projection `out`.
- Semantic dimension: `BeatGANsAutoencConfig.enc_out_channels`, populated
  from `TrainConfig.style_ch` by `TrainConfig.make_model_conf()`.
- Decoder conditioning: `BeatGANsAutoencModel.forward()` obtains `cond`,
  passes it through `TimeStyleSeperateEmbed`, then threads `cond_emb`
  through the U-Net blocks. Each conditioned `ResBlock` applies
  `cond_emb_layers`; `apply_conditions()` combines its scale/shift with the
  timestep scale/shift.
- Training: upstream already uses Lightning (`experiment.LitModel`).
  `training_step()` samples timesteps with `T_sampler.sample`, calls
  `sampler.training_losses(model, x_start, t)`, and averages `loss`.
  `on_train_batch_end()` performs EMA; `configure_optimizers()` preserves the
  configured Adam/AdamW and optional warmup schedule.

## 10. Where to look next

- Environment setup, exact commands, current checkpoint inventory,
  reproducibility notes: `RUNNING.md`.
- Metric definitions (read `counterfactuals/run_pcf_eval.py::main` /
  `evaluate_intervention`, verified against source): for a given
  `(attribute, direction)`, restrict to source images not already on the
  target side (`valid`); **CC** = fraction of those whose classifier
  prediction flips to the target side after the CFG edit
  (`success.sum()/valid.sum()`). **FC** = 1 − mean flip rate of the other 36
  (unmodeled) attributes' classifier predictions among the successful edits;
  `FC_raw` uses the observed flip rate directly, `FC_corr` first subtracts
  each unmodeled attribute's natural co-occurrence baseline flip rate
  (`compute_baselines_and_weights`, cached in `correlation_baseline.json`) so
  attributes that are simply correlated with the intervened one aren't
  penalized (see §8). **PCF** = harmonic mean (F1-style) of CC and FC,
  reported as macro/micro/weighted × raw/corrected.
