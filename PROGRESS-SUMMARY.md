# TODO-List progress summary

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

- **Item 3 — MorphoMNIST dataset, causal graph, and HDAE training.** MorphoMNIST++
  dataset (140k images, digit/thickness/intensity/hue modeled + 10 unobserved
  factors logged) and its causal graph/SCM (`causal_graph_morpho.yaml`,
  `morpho_scm.pt`) built in an earlier pass. **This pass (2026-08-11):**
  generalized the HDAE conditioning path (previously binary-only) to support
  MorphoMNIST's mixed categorical (digit) + continuous (thickness/intensity/hue)
  attributes — new `MixedAttributeEmbedding`, mask-based CFG null (fixed a real
  bug: the old binary "null" trick silently corrupted continuous conditioning),
  a new MorphoMNIST datamodule, and three new `morpho_hier_k{1,5,11}.yaml`
  configs. Training launched for all three k-variants; CF1 (item 4's
  CelebA-specific eval harness) was **not** ported — judged out of proportion
  to the time available, see `TODO-List` item 3 for the reasoning and the
  lighter substitute (`morpho_cf_smoketest.py`) built instead. Full decisions
  list and status are in `TODO-List` item 3's second-pass section — this file
  only tracks top-level done/not-done, not the reasoning.

- **Item 3, continued (2026-08-11 to 2026-08-14).** Dataset finalized as
  `morphomnist_70k.h5` (rotation narrowed to 5-bin ±45°, slant dropped, hue
  changed continuous → categorical 10-bin); all attribute predictors
  retrained, top-k accuracy added. Found and fixed a real bug (categorical
  attributes whose raw storage isn't already the class index — hue —
  getting truncated instead of binned) in three places: the trained
  embedding module, the SCM's likelihood/counterfactual math, and three eval
  scripts. **Found a fourth instance of the same bug on 2026-08-14, in the
  training data path itself (`attr_utils.to_cond_values`) — fixed same day**,
  after a bounded repo-wide audit confirmed it was the only remaining
  instance (see `TODO-List` item 3's fifth-pass section for the full sweep
  and what was deliberately left alone). Every hue-conditioned checkpoint
  trained so far (`k1_v2`, `k11_v2`, `k11_v3`) still has invalid hue results
  — the fix only affects *future* training, it doesn't retroactively repair
  an already-trained embedding table — so a retrain is still needed and
  hasn't been launched; digit/thickness/intensity are unaffected throughout.
  Also implemented: `attr_fusion: concat_film`, an opt-in alternative to
  summed attribute embeddings that gives each attribute a protected slice
  and FiLM-modulates the style vector instead of concatenating into it, plus
  independent per-attribute dropout (`attr_dropout_prob`). Full state
  mirrored to `s3://najibi-research-7f2a/hdae-handoff/`.

- **Item 3, retrain (2026-08-14 to 2026-08-17).** Deleted the three
  pre-fix checkpoints (`k1_v2`/`k11_v2`/`k11_v3`) and retrained `k11_v3` on
  the fixed code, distributed across both GPUs (batch 128 global, lr 4e-4,
  30k steps). **The fix works**: hue CC went from 0.000 (pre-fix) to
  **0.971**, confirmed both numerically and by eye in a labeled sweep grid
  (every sampled hue intervention now renders the correct target color,
  previously either inert or corrupted noise). Global CC 0.316→0.504, macro
  CF1 0.397→0.497, digit CC also improved (0.706→0.755). Real tradeoff, not
  free: hue's FC_obs dropped to 0.29 — intervening on hue now visibly pulls
  thickness along with it some of the time. Not a controlled ablation of
  `concat_film`/`attr_dropout` in isolation (would need a `sum`/no-dropout
  run on the same fixed code to isolate that), but the actual blocker — the
  `to_cond_values` bug — is confirmed fixed. Full detail in `TODO-List`
  item 3's sixth-pass section.

- **Item 3, k=1/5/11 comparison (2026-08-17 to 2026-08-18).** Trained
  `k1_v3`/`k5_v3` with the same recipe as `k11_v3` for a matched 3-way
  comparison. `k1_v3` performs best overall (global CC 0.558, hue CC 0.990).
  **`k5_v3` is a real anomaly**: hue CC only 0.182 (vs 0.99/0.97 for k1/k11)
  despite identical code/data/batch/lr — confirmed by eye, several hue
  interventions render as corrupted color blobs. Root cause not
  investigated (candidates: unlucky seed, or a genuine k=5-specific
  interaction with FiLM) — flagged, not pursued, since training all three
  fast was the actual ask. Full table in `TODO-List` item 3's seventh-pass
  section.

## Not done

- **Item 2, follow-up:** the causal DAG itself is still empty — add real
  `[parent, child]` edges to `causal_graph.yaml` and re-run `train_scm.py`
  whenever you're ready to make CC/FC-observed non-trivial. (MorphoMNIST's
  *separate* graph, `causal_graph_morpho.yaml`, already has a real edge,
  thickness -> intensity — this note is about the original CelebA graph only.)
  Causal3DIdent (item 3): still not started.
- **Item 3, follow-up:** CF1 eval harness not ported to MorphoMNIST (see
  above); a batch-size-matched rerun to cleanly separate the k=1 reconstruction
  confound from a true k effect (not needed for the counterfactual-editing
  result, which points the opposite way from the confound — see `TODO-List`).
  All three k-variants finished training (150,000 steps each) and were
  evaluated: k=1 wins reconstruction but that's confounded with 2x the
  training images; k=5/k=11 clearly win on counterfactual editing quality
  (digit-swap accuracy 0.27 -> 0.53 -> 0.56 for k=1/5/11) and leakage control,
  which is not explained by the confound. Full numbers in
  `experiments/hdae/outputs/morpho_hier_k_comparison_report.md`.
- **Item 5 — CheXpert.** Explicitly parked by you until 1–4 land; still
  parked.
