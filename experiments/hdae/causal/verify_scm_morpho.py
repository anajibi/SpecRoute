#!/usr/bin/env python
"""Correctness check for the fitted MorphoMNIST++ SCM (real graph, not a toy).

Unlike CelebA's shipped edgeless graph, this one has a real declared edge
(thickness -> intensity) that's actually exercised by a normal run, so this
checks the *fitted* checkpoint directly rather than a synthetic detour:

1. Round-trip identity for all four nodes, including the categorical
   `digit` node (abduct then propagate with no intervention reproduces the
   original value exactly).
2. do(thickness) moves intensity's propagated value (the causal-graph
   analogue of data/verify_morphomnist.py's pixel-level correlation check,
   now through the fitted SCM) and leaves the unrelated roots digit/hue
   untouched.
3. do(digit) (a categorical intervention) leaves thickness/intensity/hue
   untouched -- digit is an isolated root, nothing should move.

Run: python experiments/hdae/causal/verify_scm_morpho.py
"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.hdae.causal.scm import SCM

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

SCM_CKPT = "experiments/hdae/outputs/scm/morpho_scm.pt"
ATTR_NPZ = "experiments/hdae/data/packed/morphomnist_32.npz"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scm = SCM.load(SCM_CKPT, device=device)
    graph = scm.graph
    logging.info("loaded SCM: attributes=%s edges=%s kinds=%s", graph.attributes, graph.edges,
                 {n: s.kind for n, s in scm.specs.items()})

    arrays = np.load(ATTR_NPZ, allow_pickle=True)
    attr_names = [str(x) for x in arrays["attribute_names"]]
    cols = [attr_names.index(a) for a in graph.attributes]
    attrs = torch.from_numpy(arrays["attrs"][:512, cols].astype(np.float32)).to(device)
    attr_index = {name: i for i, name in enumerate(graph.attributes)}

    # 1. Round-trip identity, all four nodes including categorical `digit`.
    eps_by_node = scm.abduct(attrs, attr_index)
    cf_recon = scm.counterfactual(attrs, attr_index, interventions={})
    for node in graph.attributes:
        real = attrs[:, [attr_index[node]]]
        recon = cf_recon[node]
        if scm.specs[node].kind == "categorical":
            match = torch.equal(recon.round(), real.round())
            assert match, f"round-trip failed for categorical node {node!r}"
        else:
            max_err = float((recon - real).abs().max())
            assert max_err < 1e-2, f"round-trip failed for {node!r}: max_err={max_err:.4f}"
    logging.info("PASS: round-trip identity holds for all 4 nodes (digit, thickness, intensity, hue), n=%d",
                 attrs.shape[0])

    # 2. do(thickness) propagates to intensity, leaves digit/hue untouched.
    new_thickness = torch.full((attrs.shape[0], 1), 6.5, device=device)
    cf_do_thickness = scm.counterfactual(attrs, attr_index, interventions={"thickness": new_thickness})
    intensity_shift = (cf_do_thickness["intensity"] - cf_recon["intensity"]).abs()
    digit_shift = (cf_do_thickness["digit"].round() != cf_recon["digit"].round()).float().mean()
    hue_shift = (cf_do_thickness["hue"] - cf_recon["hue"]).abs().max()
    logging.info("do(thickness=6.5): intensity shift mean=%.2f max=%.2f; digit changed frac=%.4f; hue max shift=%.2e",
                 float(intensity_shift.mean()), float(intensity_shift.max()), float(digit_shift), float(hue_shift))
    assert float(intensity_shift.mean()) > 1.0, "intensity did not move under a thickness intervention"
    assert float(digit_shift) == 0.0, "digit (unrelated root) moved under a thickness intervention"
    assert float(hue_shift) < 1e-4, "hue (unrelated root) moved under a thickness intervention"
    logging.info("PASS: do(thickness) propagates to intensity and leaves digit/hue untouched")

    # 3. do(digit) is a no-op on everything else (digit is an isolated root).
    new_digit = torch.full((attrs.shape[0], 1), 7.0, device=device)
    cf_do_digit = scm.counterfactual(attrs, attr_index, interventions={"digit": new_digit})
    thickness_shift = (cf_do_digit["thickness"] - cf_recon["thickness"]).abs().max()
    intensity_shift2 = (cf_do_digit["intensity"] - cf_recon["intensity"]).abs().max()
    hue_shift2 = (cf_do_digit["hue"] - cf_recon["hue"]).abs().max()
    assert float(thickness_shift) < 1e-4, "thickness moved under a digit intervention"
    assert float(intensity_shift2) < 1e-4, "intensity moved under a digit intervention"
    assert float(hue_shift2) < 1e-4, "hue moved under a digit intervention"
    digit_now = cf_do_digit["digit"].round()
    assert torch.equal(digit_now, new_digit.round()), "do(digit) did not force the intervened value"
    logging.info("PASS: do(digit) forces the new class and leaves thickness/intensity/hue untouched")

    logging.info("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
