#!/usr/bin/env python
"""Fit the per-attribute causal SCM (normalizing flows) on packed attribute data.

Run once per dataset, independent of and prior to HDAE training/inference
(TODO item 2). Dataset-agnostic by construction: only needs a labeled
attribute table (columns matching ``causal_graph.yaml``'s ``attributes``)
and the declared DAG -- no image-specific or CelebA-specific code.

NOTE: fits on the full attribute table, which includes images later drawn
into the CF1 eval cohorts -- a form of train/eval leakage for the SCM
specifically (the HDAE model and attribute classifier have their own,
separate splits and are unaffected). Low-stakes right now since the shipped
causal_graph.yaml is edgeless (no propagation happens), but worth revisiting
with a held-out split if/when real edges are added and CF1 numbers need to
be leakage-free.
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from experiments.hdae.causal.attr_io import load_attr_table
from experiments.hdae.causal.graph import CausalGraph
from experiments.hdae.causal.scm import SCM, node_specs_from_config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--causal-graph", default="experiments/hdae/configs/causal_graph.yaml")
    p.add_argument("--attr-npz", required=True, help="path to a .npz (CelebA-HQ) or .h5/.hdf5 (MorphoMNIST++) attribute table")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    with open(args.causal_graph) as f:
        raw = yaml.safe_load(f)
    graph = CausalGraph.from_dict(raw)
    node_specs = node_specs_from_config(raw, graph)
    output = Path(args.output or raw["scm_checkpoint"])
    output.parent.mkdir(parents=True, exist_ok=True)

    raw_attrs, attr_names, _ = load_attr_table(args.attr_npz)
    cols = [attr_names.index(a) for a in graph.attributes]
    attrs_matrix = raw_attrs[:, cols].astype(np.float32)
    for j, node in enumerate(graph.attributes):
        if node_specs[node].kind == "binary":
            attrs_matrix[:, j] = (attrs_matrix[:, j] > 0).astype(np.float32)
    attrs = torch.from_numpy(attrs_matrix)
    attr_index = {name: i for i, name in enumerate(graph.attributes)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scm = SCM(graph, node_specs).to(device)
    opt = torch.optim.Adam(scm.parameters(), lr=args.lr)
    data = attrs.to(device)
    n = data.shape[0]
    logging.info("fitting SCM: attributes=%s edges=%s kinds=%s n_images=%d device=%s",
                 graph.attributes, raw.get("edges", []), {n_: s.kind for n_, s in node_specs.items()}, n, device)
    log_every = max(1, args.epochs // 10)
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            batch = data[idx]
            opt.zero_grad()
            loss = scm.nll(batch, attr_index)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * len(idx)
        if (epoch + 1) % log_every == 0 or epoch == 0:
            logging.info("epoch=%d mean_nll=%.4f", epoch + 1, total_loss / n)

    scm.save(output)
    logging.info("wrote fitted SCM to %s", output)


if __name__ == "__main__":
    main()
