# ABLE — one entry point for every trained model

    able list                  what is registered
    able show   <model>        config, checkpoint, attributes, graph
    able verify <model>|--all  does the config still build the claimed architecture
    able eval   <model>        run the evaluation protocol
    able score  --persample    re-pool an existing run under the shared metrics

## What was actually wrong

Training was already shared — `scripts/train.py` dispatches on `data.type` and every model is the
same `HierarchicalAutoencModel`. The Aug-28 Causal3DIdent checkpoint loads into today's model with
a 736-key exact match, and `hier_autoenc.py` has not changed since 25 August. There was no
multi-architecture problem to solve.

Two things were wrong:

**Provenance was not written down.** Nothing recorded which config reproduces which checkpoint. A
MorphoMNIST checkpoint filed under `ladder/k5_s42/` turned out to be reproduced by
`adv_k5_s42.yaml`, and the only way to find that out was to build each candidate and compare
state_dict keys. Getting it wrong does not raise — PyTorch reports missing and unexpected keys and
carries on with a partly random model. `registry.py` now records the pairing and the key count, and
`able verify` checks it.

**Evaluation was two pipelines.** Causal3DIdent and CelebA-HQ had separate CLIs, separate metric
code and separate output schemas. That is how they ended up combining CC with different
preservation terms without anyone deciding to.

## Layout

    registry.py          every released model: checkpoint, config, verified key count
    adapters/base.py     what a dataset adapter must answer
    adapters/c3di.py     one categorical factor and six continuous, normalised by trivial error
    adapters/celebahq.py binary attributes, two causal graphs, the inversion guard
    shared/metrics.py    CC and FC components; CF1 combiners, no default
    shared/schema.py     one result shape
    cli.py

The adapter boundary is where the datasets genuinely differ: Causal3DIdent measures factors in
their own units, so "did it change" is a distance that must be normalised before it can sit beside
an accuracy; CelebA-HQ measures binary labels, so it is a flip and the reader's error rate bounds
what can be claimed. Everything downstream — guidance sweep, null-intervention control, CC/FC
arithmetic, output schema — is shared.

## CF1 is deliberately undecided

`shared/metrics.py` computes the components and offers the combiners but picks no default, because
the evaluation protocol is still being defined. `aggregate()` emits a CF1 **per FC term**, named, so
a table can never be ambiguous about which preservation set it used.

One thing is pinned: CF1 is the mean of per-intervention harmonic means, not the harmonic mean of
pooled rates. Those differ — 0.7717 against 0.7756 for one real cell — and the pooled form lets a
frequent intervention dominate a rare one. The original `aggregate_metrics.py` docstring claimed
the pooled form while the code did the per-intervention one; `cf1_pooled` exists only so a test can
show the two apart.

## Tests

    pytest -m "not slow and not gpu"   56 tests, CPU, ~2 min
    pytest -m slow                     adds S3 existence and exact key-match for all 8 (~20 GB)
    pytest -m gpu                      one optimizer step per dataset; needs a CUDA device

The GPU tier is skipped while the driver is down. It exists for the `accum_batches`/EMA trap:
`conf.accum_batches` gates the EMA update, so setting it on the Trainer alone advances the EMA eight
times too fast, which corrupts every evaluation while the loss curve still looks normal.

## Not covered

MorphoMNIST is out of scope — those models are being retrained. The generation half of `eval`
(DDIM inversion, guided sampling) still lives in the per-dataset scripts and needs a GPU.
