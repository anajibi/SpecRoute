#!/usr/bin/env python
"""Standalone correctness check for the SCM abduct/intervene/propagate path.

The shipped ``causal_graph.yaml`` is edgeless (your explicit choice,
2026-07-31), so ``propagate()`` there is a no-op-equivalent identity by
construction -- nothing in a normal run exercises the propagation logic
itself. This script fits a TOY, non-empty 2-hop chain
(Male -> Young -> Smiling) on the real packed CelebA-HQ attribute table and
checks the three properties abduct/intervene/propagate must satisfy before
the machinery can be trusted with real edges later:

1. Round-trip identity: abduct real attributes, propagate with NO
   intervention -> reconstructs the original binarized value for every
   node exactly (holds by construction -- each node's noise cancels its own
   conditional mean/std when nothing is forced -- but a bug in parent-order
   handling, sign errors, etc. would break it).
2. Intervening on Male changes Young's (and transitively Smiling's)
   propagated probability, and leaves the non-descendant Eyeglasses
   untouched.
3. Topological order is respected: Smiling only moves through Young, i.e.
   the graph is actually being walked in dependency order, not attribute
   declaration order.

Run: python experiments/hdae/causal/verify_scm.py
"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.hdae.causal.graph import CausalGraph
from experiments.hdae.causal.normalize import to_prob
from experiments.hdae.causal.scm import SCM

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

ATTR_NPZ = "experiments/hdae/data/packed/celebahq_64_attrs.npz"


def main():
    arrays = np.load(ATTR_NPZ, allow_pickle=True)
    attr_names = [str(x) for x in arrays["attribute_names"]]
    attributes = ["Male", "Young", "Smiling", "Eyeglasses"]
    edges = [("Male", "Young"), ("Young", "Smiling")]
    graph = CausalGraph(attributes, edges)
    attr_index = {name: i for i, name in enumerate(attributes)}
    cols = [attr_names.index(a) for a in attributes]
    attrs01 = torch.from_numpy((arrays["attrs"][:, cols] > 0).astype(np.float32))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scm = SCM(graph, eps=0.05).to(device)
    opt = torch.optim.Adam(scm.parameters(), lr=1e-2)
    data = attrs01.to(device)
    n = data.shape[0]
    logging.info("fitting toy SCM (Male -> Young -> Smiling) on %d real images", n)
    for epoch in range(300):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, 4096):
            idx = perm[start:start + 4096]
            opt.zero_grad()
            loss = scm.nll(data[idx], attr_index)
            loss.backward()
            opt.step()
    with torch.no_grad():
        final_nll = float(scm.nll(data, attr_index))
    logging.info("fit done, final mean_nll=%.4f", final_nll)

    batch = data[:256]

    # 1. round-trip identity
    eps_by_node = scm.abduct(batch, attr_index)
    z_recon = scm.propagate(eps_by_node, interventions={})
    for node in attributes:
        recon_bin = (to_prob(z_recon[node]) >= 0.5).float().squeeze(-1)
        real_bin = batch[:, [attr_index[node]]].squeeze(-1)
        n_mismatch = int((recon_bin != real_bin).sum())
        assert n_mismatch == 0, f"round-trip failed for {node}: {n_mismatch}/{len(real_bin)} mismatches"
    logging.info("PASS: round-trip identity holds for all 4 nodes with no intervention (n=%d)", batch.shape[0])

    # 2 & 3. intervening on Male propagates through Young to Smiling, not to Eyeglasses
    male_flip = 1.0 - batch[:, [attr_index["Male"]]].squeeze(-1)
    z_cf = scm.propagate(eps_by_node, interventions={"Male": male_flip})

    def prob_shift(node):
        return (to_prob(z_cf[node]).squeeze(-1) - to_prob(z_recon[node]).squeeze(-1)).abs()

    eyeglasses_shift = prob_shift("Eyeglasses")
    young_shift = prob_shift("Young")
    smiling_shift = prob_shift("Smiling")

    max_eyeglasses_shift = float(eyeglasses_shift.max())
    assert max_eyeglasses_shift < 1e-6, \
        f"non-descendant Eyeglasses should be untouched by a Male intervention, max shift={max_eyeglasses_shift:.2e}"
    logging.info("PASS: non-descendant Eyeglasses unaffected by Male intervention (max shift=%.2e)",
                 max_eyeglasses_shift)

    logging.info("Young (direct child) probability shift under Male intervention: mean=%.4f max=%.4f",
                 float(young_shift.mean()), float(young_shift.max()))
    logging.info("Smiling (2-hop grandchild) probability shift under Male intervention: mean=%.4f max=%.4f",
                 float(smiling_shift.mean()), float(smiling_shift.max()))

    assert float(young_shift.mean()) > 1e-3, \
        "Young did not move under a Male intervention -- propagation isn't wired correctly, or the graph learned no dependence"
    assert float(smiling_shift.mean()) > 1e-4, \
        "Smiling (2-hop) did not move under a Male intervention -- topological propagation through Young is broken"
    logging.info("PASS: intervention propagates Male -> Young -> Smiling and stops at the non-descendant boundary")

    logging.info("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
