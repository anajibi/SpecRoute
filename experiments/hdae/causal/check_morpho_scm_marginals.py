"""Do samples from the fitted MorphoMNIST SCM look like the real attribute data?

The same check that caught the Causal3DIdent SCM. There, the conditional-Gaussian mechanism
produced Gaussian-shaped marginals for attributes that are actually uniform, and had to be
replaced with rational-quadratic spline flows before any counterfactual work was trustworthy.

The MorphoMNIST checkpoint has BOTH `encoders` (mu/log_sigma) and `flows` entries, but the flow
entries contain only `_distribution._context_encoder` -- a conditional diagonal Normal -- with no
transform parameters, so it is not a spline. And `abduct`/`propagate` both read `_dist_params`,
which uses the Gaussian encoders. So this is the mechanism that failed before, and this script
measures whether it fails here too.

Reported per attribute: 1-Wasserstein distance between SCM samples and the real marginal, the
out-of-bounds rate (samples outside the physically valid range), and overlaid histograms.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

from experiments.hdae.causal.graph import CausalGraph
from experiments.hdae.causal.scm import SCM, NodeSpec

CKPT = ROOT / "experiments/hdae/outputs/scm/morpho_scm.pt"
H5 = ROOT / "experiments/hdae/data/morphomnist/morphomnist_70k_v2.h5"
OUT = ROOT / "experiments/hdae/outputs/morpho_scm_marginals.png"
N = 20000


def main():
    b = torch.load(CKPT, map_location="cpu")
    specs = {k: NodeSpec(**v) if not isinstance(v, NodeSpec) else v for k, v in b["node_specs"].items()}
    scm = SCM(CausalGraph(b["attributes"], b["edges"]), specs)
    scm.load_state_dict(b["state_dict"]); scm.eval()
    attrs = b["attributes"]

    with h5py.File(H5, "r") as h:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in h["attribute_names"][:]]
        A = h["attrs"][:].astype(np.float64)
    real = {a: A[:, names.index(a)] for a in attrs}

    # sample: categorical roots from their learned logits, continuous from N(0,1) noise
    eps = {}
    ctx = torch.ones(N, 1)
    for a in attrs:
        s = specs[a]
        if s.kind == "categorical":
            p = torch.softmax(scm.encoders[a](ctx[:1]), dim=-1)[0]
            idx = torch.multinomial(p, N, replacement=True).float()
            # digit stores the class index directly; hue stores its bin CENTRE
            if s.lo is None:
                eps[a] = idx.view(N, 1)
            else:
                eps[a] = (s.lo + (idx + 0.5) * (s.hi - s.lo) / s.num_classes).view(N, 1)
        else:
            eps[a] = torch.randn(N, 1)
    with torch.no_grad():
        z = scm.propagate(eps, {})
    samp = {}
    for a in attrs:
        v = z[a].view(-1)
        # propagate returns z-space for continuous nodes; _to_raw is the inverse of _to_z
        samp[a] = (v if specs[a].kind == "categorical"
                   else scm._to_raw(a, v.view(-1, 1)).view(-1)).numpy()

    print(f"{'attribute':12s} {'kind':12s} {'W1':>10s} {'real range':>22s} {'sample range':>22s} {'OOB':>8s}")
    fig, ax = plt.subplots(1, len(attrs), figsize=(4.6 * len(attrs), 3.9))
    for i, a in enumerate(attrs):
        r, s_ = real[a], samp[a]
        sp = specs[a]
        lo, hi = (sp.lo, sp.hi) if sp.lo is not None else (r.min(), r.max())
        oob = float(((s_ < lo) | (s_ > hi)).mean()) if sp.kind != "categorical" else 0.0
        w1 = wasserstein_distance(r, s_)
        print(f"{a:12s} {sp.kind:12s} {w1:10.4f} "
              f"{f'[{r.min():.3f}, {r.max():.3f}]':>22s} {f'[{s_.min():.3f}, {s_.max():.3f}]':>22s} "
              f"{oob*100:7.2f}%")
        bins = 40 if sp.kind != "categorical" else np.arange(-0.5, (sp.num_classes or 10) + .5)
        if sp.kind == "categorical" and sp.lo is not None:
            conv = lambda v: np.round((v - sp.lo) / (sp.hi - sp.lo) * sp.num_classes - 0.5)
            r, s_ = conv(r), conv(s_)
        ax[i].hist(r, bins=bins, density=True, alpha=.55, color="#2F6FB0", label="data")
        ax[i].hist(s_, bins=bins, density=True, alpha=.55, color="#C0553B", label="SCM samples")
        ax[i].set_title(f"{a}   W1={w1:.4f}", fontsize=11, fontweight="bold")
        ax[i].spines[["top", "right"]].set_visible(False); ax[i].grid(alpha=.25, lw=.6)
        if i == 0: ax[i].legend(frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig(OUT, dpi=140, facecolor="white")
    print(f"\nwrote {OUT}")

    # the one declared edge: does thickness -> intensity reproduce?
    rt, ri = real["thickness"], real["intensity"]
    st, si = samp["thickness"], samp["intensity"]
    print(f"\nedge thickness -> intensity")
    print(f"  data    Pearson r = {np.corrcoef(rt, ri)[0,1]:+.4f}")
    print(f"  SCM     Pearson r = {np.corrcoef(st, si)[0,1]:+.4f}")


if __name__ == "__main__":
    main()
