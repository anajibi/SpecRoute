"""Fit + verify the Causal3DIdent attribute SCM. Self-contained: no imports from
experiments.hdae, so it runs without the repo's hardcoded /home/anajibi/HDM sys.path.

Why not causal/scm.py: that SCM is scalar-per-node (see its module docstring).
Causal3DIdent's pos_obj and rot_obj are 3-dimensional, so each node here carries a
`dim` and its mechanism acts on that many dims.

Two mechanism families, selected with --mechanism:

  gaussian  conditional diagonal Gaussian, mean/log-sigma from MLP(parents). This is
            what causal/scm.py uses (an nflows ConditionalDiagonalNormal with an
            identity transform stack), extended to vectors. It matches the mean and
            variance of any target but can only ever emit a bell shape, and puts mass
            outside the data's [-1, 1] bounds.

  spline    conditional monotone rational-quadratic spline on [-1, 1], parameterised
            by MLP(parents), over a Uniform[-1, 1] base. This is a real (if elementwise)
            normalising flow: it can represent uniform, bimodal, or skewed conditionals,
            and because the spline maps [-1, 1] onto itself every sample is in-bounds by
            construction. `nflows` supplies the spline; the repo already depends on it.

Both keep identical counterfactual semantics -- exogenous noise u is recovered by the
inverse mechanism and re-used on the forward pass -- so the abduct / intervene / predict
recipe (Pearl; Pawlowski et al. Deep SCM) is unchanged:
  abduct    u = mech^{-1}(z ; parents)      per node, from real values
  propagate walk topological order, forcing intervened nodes, reconstructing the rest
            from their preserved u

Graph (configs/causal_graph_causal3dident.yaml):
    class -> rot_obj,  class -> pos_obj,  pos_spl -> pos_obj
Roots: class (categorical, 7), pos_spl (continuous, 1-D).

Usage:
    python experiments/hdae/causal/train_scm_causal3dident.py --mechanism spline
    python experiments/hdae/causal/train_scm_causal3dident.py --verify-only
"""
import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from nflows.transforms.splines.rational_quadratic import rational_quadratic_spline

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "experiments/hdae/configs/causal_graph_causal3dident.yaml")
ROOT_CONTEXT_DIM = 1
SPLINE_BINS = 16
BOUND = 1.0
EDGE = 1e-6          # keep inputs strictly inside [-1, 1] for the spline


# ---------------------------------------------------------------- graph

class CausalGraph:
    """Minimal local copy of causal/graph.py (kept here so this file stands alone)."""

    def __init__(self, attributes: List[str], edges):
        self.attributes = list(attributes)
        self.edges = [tuple(e) for e in edges]
        self._children = {a: [] for a in self.attributes}
        self._parents = {a: [] for a in self.attributes}
        for p, c in self.edges:
            if p not in self._children or c not in self._children:
                raise ValueError(f"edge {(p, c)} references an attribute not in {self.attributes}")
            self._children[p].append(c)
            self._parents[c].append(p)
        self._topo = self._topological_order()

    def parents(self, n):
        return list(self._parents[n])

    def children(self, n):
        return list(self._children[n])

    def descendants(self, n):
        seen, stack = set(), list(self._children[n])
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(self._children[x])
        return seen

    def topological_order(self):
        return list(self._topo)

    def _topological_order(self):
        indeg = {a: len(self._parents[a]) for a in self.attributes}
        q = [a for a in self.attributes if indeg[a] == 0]
        order = []
        while q:
            n = q.pop(0)
            order.append(n)
            for c in self._children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    q.append(c)
        if len(order) != len(self.attributes):
            raise ValueError(f"graph over {self.attributes} with edges {self.edges} has a cycle")
        return order


# ---------------------------------------------------------------- data

def load_attributes(cfg: dict, split: str) -> Dict[str, torch.Tensor]:
    """raw_latents_{i}.npy (one file per object class) -> per-attribute tensors.

    Returns every attribute in the config, modeled and unmodeled alike, so the
    unmodeled three are available to callers as the FC_unobserved pool.
    """
    root = cfg["data_root"]
    if not os.path.isabs(root):
        root = os.path.join(REPO_ROOT, root)
    d = os.path.join(root, split)
    n_classes = cfg["nodes"]["class"]["num_classes"]
    per_class = []
    for i in range(n_classes):
        p = os.path.join(d, f"raw_latents_{i}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found -- extract {split}.tar.gz first")
        per_class.append(np.load(p))
    lat = torch.from_numpy(np.concatenate(per_class)).float()
    cls = torch.cat([torch.full((a.shape[0],), i, dtype=torch.long)
                     for i, a in enumerate(per_class)])
    out = {"class": cls.unsqueeze(-1).float()}
    for name, cols in cfg["latent_columns"].items():
        out[name] = lat[:, cols]
    return out


# ---------------------------------------------------------------- scm

class ContextEncoder(nn.Module):
    def __init__(self, context_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(context_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class SCM(nn.Module):
    """One mechanism per node. The single categorical node (class) is a root, so it is
    an unconditional learnable categorical -- no Gumbel-max abduction needed."""

    def __init__(self, graph: CausalGraph, nodes_cfg: dict, mechanism: str = "gaussian",
                 bins: int = SPLINE_BINS):
        super().__init__()
        if mechanism not in ("gaussian", "spline"):
            raise ValueError(f"unknown mechanism {mechanism!r}")
        self.graph = graph
        self.cfg = nodes_cfg
        self.mechanism = mechanism
        self.bins = bins
        self.dims = {n: int(nodes_cfg[n].get("dim", 1)) for n in graph.attributes}
        self.kinds = {n: nodes_cfg[n]["kind"] for n in graph.attributes}
        self.ranges = {n: tuple(nodes_cfg[n].get("range", (-1.0, 1.0))) for n in graph.attributes}

        self.encoders = nn.ModuleDict()
        self.class_logits = None
        per_dim = 2 if mechanism == "gaussian" else (3 * bins + 1)
        for n in graph.attributes:
            if self.kinds[n] == "categorical":
                if graph.parents(n):
                    raise ValueError(
                        f"categorical node {n!r} has parents {graph.parents(n)}; parent-conditioned "
                        "categorical nodes need Gumbel-max abduction, not implemented")
                self.class_logits = nn.Parameter(torch.zeros(int(nodes_cfg[n]["num_classes"])))
            else:
                self.encoders[n] = ContextEncoder(self._context_dim(n), per_dim * self.dims[n])

    def _context_dim(self, node: str) -> int:
        ps = self.graph.parents(node)
        if not ps:
            return ROOT_CONTEXT_DIM
        tot = 0
        for p in ps:
            tot += int(self.cfg[p]["num_classes"]) if self.kinds[p] == "categorical" else self.dims[p]
        return tot

    def _to_z(self, node, raw):
        lo, hi = self.ranges[node]
        return (raw - lo) / (hi - lo) * 2.0 - 1.0

    def _to_raw(self, node, z):
        lo, hi = self.ranges[node]
        return (z + 1.0) / 2.0 * (hi - lo) + lo

    def _context(self, node, z_by_node, batch, device):
        ps = self.graph.parents(node)
        if not ps:
            return torch.ones(batch, ROOT_CONTEXT_DIM, device=device)
        parts = []
        for p in ps:
            if self.kinds[p] == "categorical":
                idx = z_by_node[p].long().squeeze(-1)
                parts.append(torch.nn.functional.one_hot(
                    idx, int(self.cfg[p]["num_classes"])).float())
            else:
                parts.append(z_by_node[p])
        return torch.cat(parts, dim=-1)

    # -- mechanism: gaussian ---------------------------------------

    def _gauss_params(self, node, context):
        out = self.encoders[node](context)
        d = self.dims[node]
        mu, log_sigma = out[:, :d], out[:, d:]
        return mu, log_sigma.clamp(-7.0, 3.0)

    # -- mechanism: spline -----------------------------------------

    def _spline_params(self, node, context):
        d, K = self.dims[node], self.bins
        out = self.encoders[node](context).view(-1, d, 3 * K + 1)
        return out[..., :K], out[..., K:2 * K], out[..., 2 * K:]

    def _spline(self, node, x, context, inverse):
        """Monotone RQ spline mapping [-1,1] -> [-1,1]. Returns (y, logabsdet summed over dims)."""
        w, h, dv = self._spline_params(node, context)
        y, lad = rational_quadratic_spline(
            inputs=x.clamp(-BOUND + EDGE, BOUND - EDGE),
            unnormalized_widths=w, unnormalized_heights=h, unnormalized_derivatives=dv,
            inverse=inverse, left=-BOUND, right=BOUND, bottom=-BOUND, top=BOUND)
        return y, lad.sum(-1)

    # -- shared: noise <-> value -----------------------------------

    def _to_noise(self, node, z, context):
        """z (data space) -> u (exogenous noise) + log|du/dz|."""
        if self.mechanism == "gaussian":
            mu, log_sigma = self._gauss_params(node, context)
            return (z - mu) / log_sigma.exp(), (-log_sigma).sum(-1)
        return self._spline(node, z, context, inverse=True)

    def _from_noise(self, node, u, context):
        """u -> z (data space)."""
        if self.mechanism == "gaussian":
            mu, log_sigma = self._gauss_params(node, context)
            return mu + log_sigma.exp() * u
        z, _ = self._spline(node, u, context, inverse=False)
        return z

    def _base_log_prob(self, node, u):
        d = self.dims[node]
        if self.mechanism == "gaussian":
            return torch.distributions.Normal(0.0, 1.0).log_prob(u).sum(-1)
        # Uniform[-1, 1] per dim
        return torch.full((u.shape[0],), -float(np.log(2.0)) * d, device=u.device)

    def _sample_base(self, node, n, device):
        d = self.dims[node]
        if self.mechanism == "gaussian":
            return torch.randn(n, d, device=device)
        return torch.rand(n, d, device=device) * 2 * BOUND - BOUND

    # -- likelihood ------------------------------------------------

    def log_prob(self, attrs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Joint log-likelihood per row. `attrs` in raw units (class as float index)."""
        any_node = self.graph.attributes[0]
        batch, device = attrs[any_node].shape[0], attrs[any_node].device
        z_by_node = {n: (attrs[n] if self.kinds[n] == "categorical" else self._to_z(n, attrs[n]))
                     for n in self.graph.attributes}
        total = torch.zeros(batch, device=device)
        for n in self.graph.topological_order():
            if self.kinds[n] == "categorical":
                lp = torch.log_softmax(self.class_logits, dim=-1)
                total = total + lp[attrs[n].long().squeeze(-1)]
            else:
                ctx = self._context(n, z_by_node, batch, device)
                u, logabsdet = self._to_noise(n, z_by_node[n], ctx)
                total = total + self._base_log_prob(n, u) + logabsdet
        return total

    # -- counterfactual recipe -------------------------------------

    def abduct(self, attrs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Recover each continuous node's exogenous noise from real observed values.
        Reads only real values, so node order does not matter here."""
        any_node = self.graph.attributes[0]
        batch, device = attrs[any_node].shape[0], attrs[any_node].device
        z_by_node = {n: (attrs[n] if self.kinds[n] == "categorical" else self._to_z(n, attrs[n]))
                     for n in self.graph.attributes}
        u = {}
        for n in self.graph.attributes:
            if self.kinds[n] == "categorical":
                continue
            ctx = self._context(n, z_by_node, batch, device)
            u[n], _ = self._to_noise(n, z_by_node[n], ctx)
        return u

    def propagate(self, u: Dict[str, torch.Tensor], attrs: Dict[str, torch.Tensor],
                  interventions: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Walk topological order, forcing intervened nodes and rebuilding every other
        node from its preserved noise. Order matters here (unlike abduct)."""
        interventions = interventions or {}
        any_node = self.graph.attributes[0]
        batch, device = attrs[any_node].shape[0], attrs[any_node].device
        z_by_node, out = {}, {}
        for n in self.graph.topological_order():
            if n in interventions:
                v = interventions[n]
                if v.dim() == 1:
                    v = v.unsqueeze(-1)
                if v.shape[0] == 1:
                    v = v.expand(batch, -1)
                out[n] = v
                z_by_node[n] = v if self.kinds[n] == "categorical" else self._to_z(n, v)
                continue
            if self.kinds[n] == "categorical":
                out[n] = attrs[n]
                z_by_node[n] = attrs[n]
                continue
            ctx = self._context(n, z_by_node, batch, device)
            z = self._from_noise(n, u[n], ctx)
            z_by_node[n] = z
            out[n] = self._to_raw(n, z)
        return out

    def counterfactual(self, attrs, interventions):
        return self.propagate(self.abduct(attrs), attrs, interventions)

    @torch.no_grad()
    def conditional_mean(self, node: str, attrs: Dict[str, torch.Tensor], n_mc: int = 64) -> torch.Tensor:
        """E[node | its REAL parent values], estimated by averaging the mechanism over
        base noise. Works for both mechanism families (the Gaussian's analytic mean is
        the n_mc -> inf limit of this)."""
        batch, device = attrs[node].shape[0], attrs[node].device
        z_by_node = {n: (attrs[n] if self.kinds[n] == "categorical" else self._to_z(n, attrs[n]))
                     for n in self.graph.attributes}
        ctx = self._context(node, z_by_node, batch, device)
        acc = torch.zeros(batch, self.dims[node], device=device)
        for _ in range(n_mc):
            u = self._sample_base(node, batch, device)
            acc += self._from_noise(node, u, ctx)
        return self._to_raw(node, acc / n_mc)

    @torch.no_grad()
    def sample(self, n: int, device) -> Dict[str, torch.Tensor]:
        """Ancestral sample of the joint (no data involved)."""
        out, z_by_node = {}, {}
        for node in self.graph.topological_order():
            if self.kinds[node] == "categorical":
                probs = torch.softmax(self.class_logits, dim=-1)
                idx = torch.multinomial(probs, n, replacement=True).float().unsqueeze(-1)
                out[node] = idx
                z_by_node[node] = idx
                continue
            ctx = self._context(node, z_by_node, n, device)
            z = self._from_noise(node, self._sample_base(node, n, device), ctx)
            z_by_node[node] = z
            out[node] = self._to_raw(node, z)
        return out


# ---------------------------------------------------------------- train / verify

def batches(attrs, bs, generator):
    n = next(iter(attrs.values())).shape[0]
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n - bs + 1, bs):
        idx = perm[i:i + bs]
        yield {k: v[idx] for k, v in attrs.items()}


def evaluate_nll(scm, attrs, modeled, bs=8192):
    n = next(iter(attrs.values())).shape[0]
    tot = 0.0
    with torch.no_grad():
        for i in range(0, n, bs):
            sl = {k: attrs[k][i:i + bs] for k in modeled}
            tot += -scm.log_prob(sl).sum().item()
    return tot / n


DIMS = [("pos_obj", 0, "pos_obj x"), ("pos_obj", 1, "pos_obj y"), ("pos_obj", 2, "pos_obj z"),
        ("rot_obj", 0, "rot_obj alpha"), ("rot_obj", 1, "rot_obj beta"),
        ("rot_obj", 2, "rot_obj gamma"), ("pos_spl", 0, "pos_spl")]


def marginal_report(scm, real, device, n=25200):
    """Shape fidelity of ancestral samples vs real: sd, out-of-bounds rate, and a
    1-Wasserstein distance (the moment-blind statistic -- sd alone cannot see a
    Gaussian fitted to a uniform)."""
    samp = scm.sample(n, device)
    print(f"\n{'attribute':14s} {'real sd':>9s} {'model sd':>9s} {'OOB%':>7s} {'W1':>8s}")
    rows = []
    for node, d, label in DIMS:
        r = np.sort(real[node][:, d].numpy())
        s = np.sort(samp[node][:, d].cpu().numpy())
        q = np.linspace(0, 1, 2000)
        w1 = float(np.abs(np.quantile(r, q) - np.quantile(s, q)).mean())
        oob = float(np.mean((s < -1) | (s > 1)) * 100)
        print(f"{label:14s} {r.std():9.4f} {s.std():9.4f} {oob:7.2f} {w1:8.4f}")
        rows.append((label, r.std(), s.std(), oob, w1))
    print(f"{'MEAN':14s} {'':9s} {'':9s} {np.mean([x[3] for x in rows]):7.2f} "
          f"{np.mean([x[4] for x in rows]):8.4f}")
    return rows


def verify(scm, attrs, modeled, cfg, device):
    g = scm.graph
    print("\n" + "=" * 72)
    print(f"VERIFICATION  (mechanism = {scm.mechanism})")
    print("=" * 72)
    sub = {k: attrs[k][:4096].to(device) for k in modeled}
    ok = True

    with torch.no_grad():
        rt = scm.counterfactual(sub, {})
    print("\n[1] round-trip (abduct -> propagate, no intervention) reproduces originals")
    for n in g.attributes:
        err = (rt[n] - sub[n]).abs().max().item()
        good = err < 1e-4
        ok &= good
        print(f"    {n:9s} max abs err = {err:.3e}  {'OK' if good else 'FAIL'}")

    TOL = 1e-4
    print(f"\n[2] interventions: descendants shift, non-descendants unchanged (tol {TOL:g})")
    for target, value in [("class", 3), ("pos_spl", 0.8)]:
        desc = g.descendants(target)
        v = torch.full((1, 1), float(value), device=device)
        with torch.no_grad():
            cf = scm.counterfactual(sub, {target: v})
        print(f"    do({target} = {value})   descendants={sorted(desc)}")
        for n in g.attributes:
            if n == target:
                continue
            shift = (cf[n] - sub[n]).abs().mean().item()
            if n in desc:
                good = shift > 1e-3
                print(f"        {n:9s} mean |shift| = {shift:.4e}  (descendant) {'OK' if good else 'FAIL'}")
            else:
                good = shift < TOL
                print(f"        {n:9s} mean |shift| = {shift:.4e}  (non-descendant, must be ~0) "
                      f"{'OK' if good else 'FAIL'}")
            ok &= good

    print("\n[3] learned E[node | class] vs empirical, per class (max abs dev over dims)")
    names = cfg["nodes"]["class"].get("class_names", [])
    cls_all = attrs["class"].squeeze(-1).long()
    worst = 0.0
    for n in ["rot_obj", "pos_obj"]:
        print(f"    {n}:")
        for c in range(cfg["nodes"]["class"]["num_classes"]):
            m = cls_all == c
            emp = attrs[n][m].mean(0)
            s = {k: attrs[k][m].to(device) for k in modeled}
            mean_pred = scm.conditional_mean(n, s).mean(0)
            dev = (mean_pred.cpu() - emp).abs().max().item()
            worst = max(worst, dev)
            nm = names[c] if c < len(names) else str(c)
            print(f"        class {c} ({nm:9s}) empirical={np.array2string(emp.numpy(), precision=3)} "
                  f"model={np.array2string(mean_pred.cpu().numpy(), precision=3)} maxdev={dev:.3f}")
    print(f"    worst conditional-mean deviation across all classes/dims: {worst:.4f}")

    print("\n[4] d E[pos_obj] / d pos_spl, per class (single graph, class is also a parent)")
    for c in range(cfg["nodes"]["class"]["num_classes"]):
        m = cls_all == c
        s = {k: attrs[k][m][:2048].to(device) for k in modeled}
        with torch.no_grad():
            lo = scm.counterfactual(s, {"pos_spl": torch.full((1, 1), -0.8, device=device)})["pos_obj"].mean(0)
            hi = scm.counterfactual(s, {"pos_spl": torch.full((1, 1), 0.8, device=device)})["pos_obj"].mean(0)
        slope = ((hi - lo) / 1.6).cpu().numpy()
        emp = np.corrcoef(attrs["pos_spl"][m].squeeze(-1).numpy(),
                          attrs["pos_obj"][m][:, 0].numpy())[0, 1]
        nm = names[c] if c < len(names) else str(c)
        print(f"        class {c} ({nm:9s}) model slope(x,y,z)="
              f"{np.array2string(slope, precision=3)}  empirical corr(pos_spl,pos_x)={emp:+.3f}")

    print("\n[5] marginal shape fidelity of ancestral samples")
    marginal_report(scm, {k: attrs[k] for k in modeled}, device)

    print("\n" + ("ALL STRUCTURAL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--mechanism", choices=["gaussian", "spline"], default="spline")
    ap.add_argument("--bins", type=int, default=SPLINE_BINS)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    graph = CausalGraph(cfg["attributes"], cfg["edges"])
    modeled = cfg["attributes"]
    device = torch.device(args.device)

    print(f"mechanism  : {args.mechanism}" + (f" ({args.bins} bins)" if args.mechanism == "spline" else ""))
    print(f"graph      : {graph.edges}")
    print(f"topo order : {graph.topological_order()}")
    print(f"modeled    : {modeled}   unmodeled: {cfg['unmodeled']}")

    train = load_attributes(cfg, "trainset")
    test = load_attributes(cfg, "testset")
    print(f"train rows : {train['class'].shape[0]}   test rows: {test['class'].shape[0]}")

    scm = SCM(graph, cfg["nodes"], mechanism=args.mechanism, bins=args.bins).to(device)
    nparam = sum(p.numel() for p in scm.parameters())
    print(f"parameters : {nparam:,}")

    default_out = cfg["scm_checkpoint"]
    if args.mechanism != "gaussian":
        default_out = default_out.replace(".pt", f"_{args.mechanism}.pt")
    out_path = args.out or os.path.join(REPO_ROOT, default_out)

    if args.verify_only:
        sd = torch.load(out_path, map_location=device)
        scm.load_state_dict(sd["state_dict"])
    else:
        opt = torch.optim.Adam(scm.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        gen = torch.Generator().manual_seed(args.seed)
        step = 0
        print(f"\ntraining {args.steps} steps, batch {args.batch_size}, lr {args.lr}, device {device}")
        while step < args.steps:
            for b in batches({k: train[k] for k in modeled}, args.batch_size, gen):
                b = {k: v.to(device) for k, v in b.items()}
                loss = -scm.log_prob(b).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(scm.parameters(), 5.0)
                opt.step()
                sched.step()
                step += 1
                if step % 1000 == 0 or step == 1:
                    print(f"  step {step:5d}  train NLL {loss.item():+.4f}")
                if step >= args.steps:
                    break
        tr = evaluate_nll(scm, {k: train[k].to(device) for k in modeled}, modeled)
        te = evaluate_nll(scm, {k: test[k].to(device) for k in modeled}, modeled)
        print(f"\nfinal joint NLL   train {tr:+.4f}   test {te:+.4f}   gap {te - tr:+.4f}")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save({"state_dict": scm.state_dict(), "config": cfg, "mechanism": args.mechanism,
                    "bins": args.bins, "train_nll": tr, "test_nll": te, "steps": args.steps}, out_path)
        print(f"saved -> {out_path}")

    ok = verify(scm, {k: test[k] for k in modeled}, modeled, cfg, device)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
