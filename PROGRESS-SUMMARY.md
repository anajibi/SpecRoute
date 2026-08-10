aright# TODO-List progress summary

(See `TODO-List` at repo root for full detail per item; this is the short version.)

## Done

- **Item 1 — model-agnostic CF contract.** Built `CFModelAdapter`
  (`experiments/hdae/counterfactuals/cf_contract.py`): any model implements
  `encode`/`intervene`/`render`, scored by the same `run_cf1_eval.py` (renamed
  from `run_pcf_eval.py`, PCF renamed CF1). Two adapters exist: `hdae` (the
  trained HDAE) and `diffae_probe` (a frozen DiffAE, the acceptance test for
  genericity). Two pre-existing crashes in the eval script were fixed and the
  metric math was regression-checked against the old outputs.
- **Item 2 — normalizing-flow causal graph.** Built a small SCM
  (`experiments/hdae/causal/`) over the 4 conditioning attributes, following
  the abduct-intervene-predict recipe, fit with `nflows`. The declared DAG
  (`configs/causal_graph.yaml`) currently has **no edges** (your call) — the
  propagation machinery is real and verified (`causal/verify_scm.py`) but
  dormant until edges are added.
- **Item 4 — CF1 decomposition.** `CC` now covers the intervened attribute
  plus its causal descendants; `FC` is split into `observed` (non-descendant
  conditioning attributes, strict) and `unobserved` (the other 36 CelebA
  attributes) — this replaces item 1's correlation-baseline raw/corr split
  rather than sitting alongside it. Landed together with item 2 since the
  redefinition needed the causal graph to mean anything.

## Not done

- **Item 2, follow-up:** the causal DAG itself is still empty — add real
  `[parent, child]` edges to `causal_graph.yaml` and re-run `train_scm.py`
  whenever you're ready to make CC/FC-observed non-trivial.
- **Item 3 — MorphoMNIST and Causal3DIdent datasets.** Not started. Flagged
  as high-value to pull forward since both ship a *known* ground-truth causal
  graph, which CelebA can't offer to validate item 2's DAG against.
- **Item 5 — CheXpert.** Explicitly parked by you until 1–4 land; still
  parked.
