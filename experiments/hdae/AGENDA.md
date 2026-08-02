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
  counterfactuals/        run_cf1_eval.py (current, active; formerly run_pcf_eval.py — PCF metric renamed CF1), cf_contract.py (model-agnostic CFModelAdapter interface), hdae_adapter.py, diffae_adapter.py, train_diffae_directions.py, run_counterfactual_eval.py, run_preservation_sweep.py, attr_classifier.py, finetune_attr_classifier.py, run_swap_eval.py
  causal/                 SCM/normalizing-flow causal graph (TODO item 2): graph.py (CausalGraph), normalize.py (logit transform), scm.py (SCM: fit/abduct/propagate), train_scm.py, verify_scm.py (correctness check on a toy non-empty DAG — the shipped graph is edgeless, doesn't exercise propagation itself)
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
  --attribute Smiling --cf1-edit-strength 8.0
```

`run_full_pipeline.py` is a transparent orchestrator: preprocess → train →
reconstruct → extract latents → train linear probes → analyze probes →
swap/null grid → abduct grid → build cohorts → counterfactual eval → CF1
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
  `run_cf1_eval.py` (the live counterfactual eval path) correctly imports
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

- `experiments/hdae/counterfactuals/run_pcf_eval.py` was the PCF eval
  script as of 2026-07-16: PCF split into raw/prevalence-corrected variants,
  but `save_frontier_plot()` still indexed the pre-refactor `FC_success` key
  and the final log line read a nonexistent `agg["micro_macro_gap"]` key —
  both `KeyError`s, confirmed by `run_k5.log`/`run_k11.log` (repo root).
  **Superseded 2026-07-31** by TODO item 1's `CFModelAdapter` contract: the
  file is now `run_cf1_eval.py`, model-agnostic, PCF renamed CF1, both
  `KeyError`s fixed. See §9 for what changed and how it was verified.
- `experiments/hdae/configs/cohorts.yaml`: `num_images` raised 256 → 1024
  (bigger fixed eval cohort per attribute/direction).
- `playground.bash`: launches `run_full_pipeline.py` for `hier_k5.yaml` and
  `hier_k11.yaml` in parallel in the background (`run_k5.log`,
  `run_k11.log` at repo root), with a `hier_k1` invocation commented out
  (already has a completed checkpoint under `outputs/hier_k1/`). Flag
  renamed `--pcf-guidance-scale` → `--cf1-edit-strength` alongside the §9
  rename. **Not currently running** — both logs are from 2026-07-16 and
  predate the fix; nothing matches `run_full_pipeline` in `ps aux` right now.
- `TODO-List` (repo root): near-term direction is (1) a model-agnostic
  contract for counterfactual-image generation so many architectures can be
  evaluated uniformly — **done, see §9** — (2) adding a Normalizing-Flow
  causal-graph model on top of/alongside HDAE, (3) bringing in MorphoMNIST
  and Causal3DIdent datasets, (4) splitting "CF1" into observed/unobserved/
  aggregated variants, (5) adding CheXpert. Items 2-5 not implemented yet.
- `checkpoints/` at repo root (untracked) — a `celeba64d2c_autoenc` upstream
  pretrained-checkpoint *download*, but only training-log sample PNGs, no
  actual model weights (no `.ckpt`/`.pt` under it) — unrelated to HDAE's own
  `outputs/<config>/checkpoints/`, and not usable as a frozen DiffAE
  baseline (see §9's resolution-mismatch note; `diffae_probe` uses
  `diffae_upstream/checkpoints/ffhq256_autoenc/last.ckpt` instead, the only
  real frozen checkpoint on this machine).

Trained/evaluated so far: `hier_k1`, `hier_k5`, `hier_k11` (K=1/5/11 levels)
all have checkpoints and completed reconstruction/latent-probing/
counterfactual/CF1 outputs under `experiments/hdae/outputs/<name>/`; the
`celeba64_*` (old-schema) configs cannot be run with the current code (§4).

## 9. Counterfactual-evaluation contract (`CFModelAdapter`, TODO item 1)

`run_cf1_eval.py` (formerly `run_pcf_eval.py`; the PCF metric was renamed
CF1) is model-agnostic: it drives whatever's registered under `--model-type`
through `counterfactuals/cf_contract.py`'s `CFModelAdapter` interface
(`load`/`encode`/`intervene`/`render`, plus a `modeled_attrs` list) instead
of importing `HDAELitModule` directly. Two adapters are registered today:

- `hdae` (`hdae_adapter.py`) — the trained per_block_attr HDAE, with the
  `AttributeCFGWrapper`/DDIM-abduction/attribute-CFG-render logic that used
  to live inline in `run_pcf_eval.py`. `edit_strength` = attribute-CFG
  guidance scale (>= 1.0).
- `diffae_probe` (`diffae_adapter.py`) — a frozen, pretrained DiffAE
  (`ffhq256_autoenc`; see below) edited via `z_sem + alpha * w` logistic-
  regression probe directions (`train_diffae_directions.py` fits `w` against
  this repo's packed CelebA-HQ data, saved to `directions.pt`). This is the
  TODO item 1 acceptance test — a genuinely different model (single
  `z_sem`, no hierarchy, no attribute conditioning/CFG) behind the same
  contract, not a refactor of HDAE. `edit_strength` = alpha.

**Resolution/domain mismatch, by necessity, not oversight:** the shared
packed pipeline (dataset + attribute classifier) everything runs on is
64x64 CelebA-HQ. The only *real* frozen DiffAE checkpoint on this machine is
`diffae_upstream/checkpoints/ffhq256_autoenc/last.ckpt` (256x256, FFHQ) —
`checkpoints/celeba64d2c_autoenc/` at repo root (matching resolution) has no
actual weights, only training-log sample PNGs (confirmed: no `.ckpt`/`.pt`
under it). `diffae_adapter.py` resizes 64->256 on encode and 256->64 on
render so CC/FC scoring stays on identical images across adapters; this
trades edit quality (upsampled input, FFHQ/CelebA domain gap) for
genericity. Don't read `diffae_probe` CF1 numbers as "DiffAE is worse than
HDAE at this task" — it's confounded by checkpoint availability, not
architecture.

Both adapters were smoke-tested end-to-end on GPU through the same
`run_cf1_eval.py` invocation (tiny hand-built cohorts, small `--max-images`,
reduced `--T`) and produce valid `cf1_per_intervention.csv`/
`cf1_aggregate.csv`. The renamed macro/micro/weighted/frontier-area
aggregation math was regression-checked bit-identical against the old
(pre-rename) `pcf_aggregate.csv` numbers for `hier_k5` by feeding the old
`pcf_per_intervention.csv` rows through the new aggregation code directly.
The refactor also fixed two latent bugs found in `run_pcf_eval.py` while
doing this: `save_frontier_plot()` read the stale `FC_success` key (the
crash AGENDA previously documented at §8), and the final log line read a
nonexistent `agg["micro_macro_gap"]` key (`_raw`/`_corr` variants exist, not
the bare name) — both would have reproduced under the new code unchanged,
so both are fixed as part of the rename, not left as follow-up.

**Model-path parity, not just aggregation-math parity:** the pre-rename
`run_pcf_eval.py` (retrieved via `git show`, since it's deleted from the
working tree by this change) and `run_cf1_eval.py --model-type hdae` were
run on the
identical smoke cohort at identical settings (`--T 5`, guidance/edit-strength
4.0). `CC` and `n` matched exactly on all 8 (attribute, direction) rows —
the extraction into `HDAEAdapter` is faithful on the metric that matters
most. Two `FC_*` values differed by ~1 flipped prediction out of 144; running
the *new* script twice in a row (nothing changed) reproduced the same
magnitude of `FC`/`n` variation between the two new-script runs, confirming
it's `cudnn.benchmark`-driven run-to-run GPU nondeterminism near classifier
decision boundaries — present in both old and new code, not a refactor
regression.

`edit_strength` is a declared `CFModelAdapter` field (not duck-typed via
`getattr`), so a future third adapter that names its knob something else
gets a clear `AttributeError` instead of a silent `nan` in the
`model_type`/`edit_strength` comparability columns.

Explicitly out of scope for this refactor (not folded into the adapter):
`run_counterfactual_eval.py`, `run_preservation_sweep.py`,
`run_swap_eval.py` still call `HDAELitModule` directly. They duplicate some
per-model logic with `hdae_adapter.py` but none is the item-1 acceptance
test; folding them in would have tripled the diff for no genericity payoff.

## 10. Causal SCM / normalizing flows (`causal/`, TODO item 2) and the CF1 observed/unobserved split (TODO item 4)

Both landed together (2026-07-31), same commit range as §9's adapter work.

**Causal graph config** (`configs/causal_graph.yaml`, shared/model-agnostic like
`cohorts.yaml`): `attributes` (currently the 4 HDAE conditioning attrs), `edges`
(`[parent, child]` pairs), `logit_smoothing_eps`, `scm_checkpoint`. **`edges: []` right
now — your explicit call** (fully independent, "no edges for now"). Add real edges here
later; nothing else needs to change.

**`causal/graph.py`** — `CausalGraph`: `parents`/`children`/transitive `descendants`
(BFS from children), `topological_order` (Kahn's algorithm, raises on a cycle).

**`causal/normalize.py`** — CelebA attributes are binary; flows need a continuous
target, so each attribute is mapped through a logit-of-smoothed-probability transform
(`p = y*(1-2*eps) + eps`, then `logit(p)`) before/after the SCM. `eps` lives in
`causal_graph.yaml`, not hardcoded — a real modeling choice (resolves TODO item 2's
"binary attributes" open decision), kept visible.

**`causal/scm.py`** — `SCM`: one node per attribute. Each node's "flow" is a
conditional diagonal Gaussian in logit space (`nflows.ConditionalDiagonalNormal`, mean/
log-std from a small MLP over the node's parents' logit values, identity transform
stack) — a scalar target can't support a coupling/autoregressive transform, so this is
the simplest valid instance of the "Deep SCM" pattern (Pawlowski et al.); a future
multi-dimensional node could drop in a richer transform without touching the rest.
Parentless (root) nodes get a constant context, making them a *learnable* unconditional
Gaussian (not fixed N(0,1)). Implements Pearl's three-step counterfactual recipe:
`abduct` (recover each node's exogenous noise from real observed values, in whatever
order — abduction only reads real values, order-independent), `propagate` (walk
`topological_order()`, forcing intervened nodes and reconstructing everything else from
its preserved noise — order matters here), `counterfactual_binary` (the full
abduct->intervene->predict->binarize pipeline other code calls).

**`causal/train_scm.py`** — fits by maximum likelihood (Adam on `-log_prob`) on the
packed attribute table (`data/packed/*_attrs.npz`), nothing image-specific — genuinely
dataset-agnostic, ready for item 3's datasets. **Leakage note:** fits on the same images
later drawn into CF1 eval cohorts (a real form of train/eval leakage for the SCM
specifically — HDAE and the attribute classifier have their own separate splits and
aren't affected). Low-stakes with today's edgeless graph since no propagation actually
happens; worth a held-out split if real edges get added and the observed-FC numbers need
to be leakage-free.

**`causal/verify_scm.py` — the actual acceptance test**, not the module itself. The
shipped edgeless graph never exercises `propagate()` (every descendant set is empty), so
this fits a *toy*, non-empty 2-hop chain (`Male -> Young -> Smiling`) on the real packed
data and asserts, on real fitted parameters: (1) round-trip identity — abduct then
propagate with no intervention reproduces every node's original binarized value exactly;
(2) intervening on `Male` shifts `Young`'s (direct child) and `Smiling`'s (2-hop
grandchild) propagated probability; (3) the non-descendant `Eyeglasses` is *exactly*
untouched (max shift `0.00e+00`). All three pass — run it yourself:
`python experiments/hdae/causal/verify_scm.py`.

**Contract integration** — `CFModelAdapter.intervene()` (§9) gained a required
`cf_attrs: Dict[str, Tensor]` parameter: the SCM's full propagated counterfactual value
for every `modeled_attrs` entry, keyed by name (not just the one flipped attribute).
`HDAEAdapter.intervene` sets the *entire* `y_idx` row from it (the TODO item 2
integration point: replacing `y_cf[:, target_col] = 1/0` with the whole vector).
`DiffAEProbeAdapter.intervene` accepts and ignores it (no attribute-vector
conditioning). No `None`/legacy-fallback code path — `run_cf1_eval.py` always computes
`cf_attrs` via `scm.counterfactual_binary()`; if `--scm-checkpoint` is missing,
`SCM.load()`'s `torch.load()` fails loudly rather than silently degrading.

**CF1 metric redefinition (TODO item 4), the actual numbers that changed:**
- **CC (Counterfactual Consistency)** — redefined to "intervened node and its
  descendants" (your instruction): pooled success rate over {intervened attribute
  flipped to target} union {each causal descendant's post-edit classifier prediction
  matches the SCM's propagated value}. With the shipped edgeless graph, `descendants`
  is `[]` for every attribute by construction, so `cc_num = success.sum()` and
  `cc_den = valid.sum() * 1` reduce algebraically to exactly §9's `CC` — not an
  observation, a structural guarantee of the empty-graph case (a GPU smoke run's `CC`
  values matched item-1's row-for-row too, consistent with but weaker evidence than the
  algebra, since separate runs vary slightly from `cudnn.benchmark` noise, §9). You
  explicitly declined a same-turn proxy for descendant scoring ("first
  implement the flows... then this part becomes trivial") — descendant success is
  scored against the *real, verified* SCM prediction, not a placeholder.
- **FC split, redefined** (this is *not* my original TODO draft — you corrected it):
  **observed** = the graph's other conditioning attributes that are *not* descendants of
  the intervened one (today: the other 3 — the causal graph "observes"/declares their
  position, expected exactly fixed, scored strictly). **unobserved** = the 36
  non-conditioning CelebA attributes, entirely outside the declared graph (no causal
  claim; same flip-rate computation the pre-item-2 `FC` used).
- **Dropped, not kept alongside the new split:** item 1's `FC_raw`/`FC_corr` /
  `CF1_raw`/`CF1_corr` and `compute_baselines_and_weights`/`correlation_baseline.json`
  are gone from `run_cf1_eval.py`. Your three-metric ask (cf1-observed, cf1-unobserved,
  cf1-aggregate) reads as replacing the correlational-adjustment axis with the causal
  one, and TODO item 4's own text says this item "replaces" the correlational baseline
  — stated here so it isn't a silently-buried behavior change.
- **Pool-size asymmetry is real, not a bug:** `CF1_observed` is currently scored over a
  3-attribute pool, `CF1_unobserved` over 36 — very different statistical power (visible
  directly via `cf1_per_intervention.csv`'s new `n_observed_attrs`/`n_unobserved_attrs`
  columns, plus `n_descendants`). Don't read the two CF1 numbers as directly comparable
  without checking those columns.
- `cf1_aggregate.csv` extends the same macro/micro/weighted/frontier-area pattern §9
  built for raw/corr, now for observed/unobserved.

## 11. Confirmed upstream (`diffae_upstream/`) integration points

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

## 12. Where to look next

- Environment setup, exact commands, current checkpoint inventory,
  reproducibility notes: `RUNNING.md`.
- Metric definitions (read `counterfactuals/run_cf1_eval.py::main` /
  `evaluate_intervention`, verified against source; full derivation in §10):
  for a given `(attribute, direction)`, restrict to source images not
  already on the target side (`valid`). **CC** = pooled success rate over
  {intervened attribute flips to target side} union {each causal descendant
  (per `causal_graph.yaml`) whose post-edit classifier prediction matches
  the SCM's propagated counterfactual value} — with today's edgeless graph
  this has zero descendants everywhere, so it's numerically the plain
  target-flip rate. **FC** is reported over two attribute pools: **observed**
  = the graph's other conditioning attributes that are not descendants of
  the intervened one (today: the other 3, strict — no correlational
  softening); **unobserved** = the 36 non-conditioning CelebA attributes
  outside the graph entirely (no causal claim). Both are `1 − mean flip
  rate` of that pool's classifier predictions among the successful edits —
  the item-1 raw/corr correlational-baseline split is gone, superseded by
  this causal one. **CF1** (formerly PCF) = harmonic mean (F1-style) of CC
  and each FC pool (`CF1_observed`, `CF1_unobserved`), reported as
  macro/micro/weighted, comparable across models via the
  `model_type`/`edit_strength` columns in `cf1_aggregate.csv` (see §9); pool
  sizes (`n_observed_attrs`/`n_unobserved_attrs`) are written per-row so the
  two CF1 numbers are never read as directly comparable by accident.
