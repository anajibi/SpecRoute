"""ABLE — one entry point for every trained model.

Training was already shared: `scripts/train.py` dispatches on `data.type` and every model is the
same `HierarchicalAutoencModel`. Evaluation was not — Causal3DIdent and CelebA-HQ had separate
CLIs, separate metric code and separate output schemas, which is how the two ended up combining CC
with different preservation terms without anyone deciding to.

This package keeps the one genuine difference in an adapter and shares everything else:

    registry.py      every released model: checkpoint key, config, verified key count
    adapters/        the per-dataset part -- what an attribute IS and how to read it back
    shared/metrics.py   CC / FC components, dataset-agnostic
    shared/schema.py    one result shape
    cli.py           able eval | able show
"""
__version__ = "0.1.0"
