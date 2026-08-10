# EXP_NOTES — Morpho-MNIST++ experiment, reconciled against the real repo

**Status (2026-08-10): Phases 0-2 done** (dataset generator, normalization,
causal graph — the scope actually asked for this pass). Phase 3 (ABLE
model/adapter) explicitly not started per instruction. See `TODO-List` item
3 for the itemized done/not-done list and verification results.

Written per the pasted plan's Phase 0 instruction ("reconcile names with the
real tree; adjust paths throughout"). The plan's assumed paths (`src/data/`,
`src/models/able/`, `src/sec/tables/`, `configs/causal_graph/`) belong to a
different/aspirational repo layout — this repo is `experiments/hdae/`. Real
mapping below; use these paths, not the plan's literal ones.

## Real entry points (plan's assumed name -> actual)

| Plan assumes | Actually is |
|---|---|
| "normalizing-flow SCM (nflows), abduct/intervene/predict" | `experiments/hdae/causal/scm.py::SCM` |
| causal graph config | `experiments/hdae/configs/causal_graph.yaml` (CelebA); new `configs/causal_graph_morpho.yaml` for this experiment |
| dataset dispatch (`data.dataset_name` / registry) | **does not exist yet** — current pipeline (`data/celeba_hq.py`, `data/datamodule.py`, `scripts/preprocess_data.py`) is CelebA-HQ-only. Morpho gets its own loader (`data/morphomnist.py`) and its own build script, not routed through a shared dispatcher — no such dispatcher exists to hook into, and building one is out of scope for this pass (nothing currently needs it beyond this one addition). |
| `CFModelAdapter` | `experiments/hdae/counterfactuals/cf_contract.py` |
| CC/FC/CF1 driver | `experiments/hdae/counterfactuals/run_cf1_eval.py` |
| "src/" | `experiments/hdae/` |
| the ABLE model | **does not exist** — current model is HDAE (`experiments/hdae/hdae/`). Not built this pass — explicit instruction: "you should not change the models at all." Everything below is dataset + causal-graph/metric infrastructure only. |

`SimpleSCMMorphoMNIST/report/7_experiments.tex` (elsewhere on this machine)
is almost certainly the paper this plan's theorem labels
(`thm:matched-budget`, `prop:localized-amplification`) come from — **not
consulted and nothing from other directories on this machine is reused**,
per explicit instruction. MorphoMNIST++ here is built from scratch: raw
digits from torchvision's public MNIST download only; all
perturbation/rendering code is new, written for this repo.

## Decisions made this pass (confirmed with you, not guessed)

- **Hue: modeled (option A).** 4th SCM node alongside digit/thickness/intensity.
- **Digit: a real categorical SCM node** (not conditioning-only), per your
  doc's Phase 2 (`do(digit)` exposed). This is the harder of the two
  decisions — the existing `SCM` only supported scalar continuous nodes
  (one conditional Gaussian per node); digit needs a genuinely different
  mechanism. See "SCM generalization" below for the scope this was built at.
- **Factor list reduced from the doc's ~13 to 10, explicitly:** built —
  digit, thickness, intensity (real perturbations, not synthetic), hue,
  slant, rotation, scale, translation, background field, texture seed.
  **Deferred, not built:** local swelling, fractures, stroke-width
  modulation — these need skeleton-based morphology (thinning + targeted
  redraw), materially more engineering than the others, and aren't
  load-bearing for this phase's gate (determinism/round-trip, not factor
  count). Noted here so a later pass doesn't assume they exist.
- **No LMDB packing.** MorphoMNIST++ is small (70k images at 32x32x3) —
  everything fits in memory / a single `.npz`. The LMDB machinery in
  `data/celeba_hq.py`/`preprocess.py` exists specifically for CelebA-HQ's
  ~19GB of raw images; using it here would be unwarranted complexity for a
  dataset three orders of magnitude smaller.

## SCM generalization scope (touches shared code — CelebA depends on it too)

`causal/normalize.py` was binary-only (logit-of-smoothed-probability,
{0,1} -> real line). `causal/scm.py` built one scalar conditional Gaussian
per node, all binary. Generalized to a per-node `kind`:
- `binary` (unchanged, existing logit-space Gaussian) — CelebA's default,
  nothing in `causal_graph.yaml` needs to change for CelebA to keep working.
- `continuous` (new) — min-max normalization to a bounded range (declared
  per-node in the graph config, e.g. thickness's empirical min/max) instead
  of the logit transform, otherwise the same conditional-Gaussian-in-
  transformed-space mechanism as binary nodes.
- `categorical` (new) — an MLP(context) -> softmax head, fit by
  cross-entropy, **implemented for root nodes only** (no parents). `digit`
  is a root in this graph (no edges into it), so this is sufficient here.
  A parent-conditioned categorical node would need Gumbel-max-style
  abduction (recovering the noise that made the observed class win) — not
  implemented; raises `NotImplementedError` with this explanation if ever
  hit, rather than silently doing something wrong.

**Regression risk:** this is shared code the working CelebA item-1/item-2
pipeline depends on. `causal/verify_scm.py` (the toy Male->Young->Smiling
DAG) and a CelebA smoke `run_cf1_eval.py` run are re-verified after this
change, before any Morpho-specific work — see task tracking for the pass/fail.

## Open item for a later phase (not solved now, scope is dataset+graph only)

Phase 5 (CF1 eval, not this pass) needs a way to *measure* whether a
rendered counterfactual's unobserved factors (rotation, background field,
etc.) match the source image's logged values — CelebA does this via a
trained attribute classifier (a proxy). The plan's text ("measure against
the ground-truth generator, not a proxy") is aspirational for factors we
can cheaply measure directly from pixels (e.g. rotation via image moments);
others will likely still need a learned regressor. Left unresolved here —
first solve dataset + causal graph, decide the measurement mechanism when
eval is actually being built.
