"""Per-attribute normalizing-flow SCM: fit/abduct/propagate over a small causal DAG.

Every node is scalar (a single binary/continuous/categorical attribute), so
each node's "flow" is the simplest valid mechanism for its kind, following
the "Deep SCM" pattern (Pawlowski et al.) scoped down to scalars:

- ``binary``/``continuous`` nodes: a conditional diagonal Gaussian in a
  transformed continuous space (mean/log-std from a small MLP over the
  node's parents, ``nflows``'s ``ConditionalDiagonalNormal`` with an
  identity transform stack -- a 1-D target can't support a coupling/
  autoregressive transform). Root nodes (no parents) get a constant
  context, making them a *learnable* unconditional Gaussian, not fixed
  N(0,1). ``causal/normalize.py`` supplies the kind-specific transform
  into/out of that continuous space (logit-of-smoothed-probability for
  binary, min-max for continuous).
- ``categorical`` nodes (e.g. MorphoMNIST's digit): an MLP(context) ->
  softmax head fit by cross-entropy -- **root nodes only**. A root's
  "exogenous noise" under this SCM's abduct/propagate contract is just its
  own observed value (nothing explains it away, so abduction is the
  identity and propagation with no intervention reproduces it exactly,
  regardless of node kind -- true for every root node in this file, not
  just categorical). A *parent-conditioned* categorical node would need
  Gumbel-max-style abduction (recovering the noise that made the observed
  class win); not implemented -- raises ``NotImplementedError`` rather than
  silently doing something wrong.

Counterfactual generation follows Pearl's three-step recipe:
1. abduct: from a real image's observed attribute vector, recover each
   node's exogenous noise.
2. intervene: force the target node(s) to specific values.
3. predict/propagate: walk the graph in topological order, recomputing each
   non-intervened node's value from its (possibly-intervened) parents and
   its own preserved exogenous noise.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform

from .graph import CausalGraph
from .normalize import minmax_to_continuous, minmax_to_raw, to_binary, to_continuous, to_prob

_ROOT_CONTEXT_DIM = 1  # constant input -> learnable unconditional distribution for parentless nodes


class NodeSpec:
    """Per-node kind + kind-specific parameters.

    ``kind`` in {"binary", "continuous", "categorical"}. ``eps`` (binary),
    ``lo``/``hi`` (continuous), ``num_classes`` (categorical) are required
    for their respective kind.
    """

    def __init__(self, kind: str = "binary", eps: float = 0.05, lo: Optional[float] = None,
                 hi: Optional[float] = None, num_classes: Optional[int] = None):
        self.kind = kind
        self.eps = float(eps)
        self.lo = None if lo is None else float(lo)
        self.hi = None if hi is None else float(hi)
        self.num_classes = None if num_classes is None else int(num_classes)
        if kind == "continuous" and (self.lo is None or self.hi is None):
            raise ValueError("continuous node needs lo/hi (a 'range: [lo, hi]' entry)")
        if kind == "categorical" and not self.num_classes:
            raise ValueError("categorical node needs num_classes")
        if kind not in ("binary", "continuous", "categorical"):
            raise ValueError(f"unknown node kind {kind!r}")

    def to_dict(self):
        return {"kind": self.kind, "eps": self.eps, "lo": self.lo, "hi": self.hi, "num_classes": self.num_classes}

    @classmethod
    def from_dict(cls, d: dict, default_eps: float):
        lo, hi = (d["range"] if "range" in d else (None, None))
        return cls(kind=d.get("kind", "binary"), eps=float(d.get("eps", default_eps)), lo=lo, hi=hi,
                   num_classes=d.get("num_classes"))


def default_node_specs(graph: CausalGraph, eps: float) -> Dict[str, NodeSpec]:
    """All-binary specs, matching this module's pre-generalization behavior (CelebA)."""
    return {node: NodeSpec(kind="binary", eps=eps) for node in graph.attributes}


def node_specs_from_config(raw: dict, graph: CausalGraph) -> Dict[str, NodeSpec]:
    """Build per-node specs from a causal_graph.yaml dict.

    Nodes not listed under ``nodes:`` default to ``binary`` with the
    top-level ``logit_smoothing_eps`` -- this is why CelebA's
    ``causal_graph.yaml`` (no ``nodes:`` section) needs no changes.
    """
    default_eps = float(raw.get("logit_smoothing_eps", 0.05))
    per_node = raw.get("nodes", {})
    return {node: NodeSpec.from_dict(per_node.get(node, {}), default_eps) for node in graph.attributes}


class _ContextEncoder(nn.Module):
    def __init__(self, context_dim: int, out_dim: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(context_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, context):
        return self.net(context)


class SCM(nn.Module):
    """Structural causal model over ``graph``'s attributes, one node per attribute."""

    def __init__(self, graph: CausalGraph, node_specs: Dict[str, NodeSpec]):
        super().__init__()
        self.graph = graph
        self.specs = node_specs
        missing = [n for n in graph.attributes if n not in node_specs]
        if missing:
            raise ValueError(f"missing NodeSpec for attributes: {missing}")
        for node in graph.attributes:
            spec = node_specs[node]
            if spec.kind == "categorical" and graph.parents(node):
                raise NotImplementedError(
                    f"categorical node {node!r} has parents {graph.parents(node)}; parent-conditioned "
                    f"categorical nodes need Gumbel-max abduction, not implemented. Only root categorical "
                    f"nodes are supported.")
        self.encoders = nn.ModuleDict()
        self.flows = nn.ModuleDict()
        for node in graph.attributes:
            spec = node_specs[node]
            ctx_dim = max(_ROOT_CONTEXT_DIM, len(graph.parents(node)))
            out_dim = spec.num_classes if spec.kind == "categorical" else 2
            enc = _ContextEncoder(ctx_dim, out_dim)
            self.encoders[node] = enc
            if spec.kind != "categorical":
                self.flows[node] = Flow(CompositeTransform([]),
                                        ConditionalDiagonalNormal(shape=[1], context_encoder=enc))

    def _context(self, node: str, z_by_node: Dict[str, torch.Tensor], batch_size: int, device) -> torch.Tensor:
        parents = self.graph.parents(node)
        if not parents:
            return torch.ones(batch_size, _ROOT_CONTEXT_DIM, device=device)
        return torch.cat([z_by_node[p] for p in parents], dim=-1)

    def _to_z(self, node: str, raw: torch.Tensor) -> torch.Tensor:
        spec = self.specs[node]
        if spec.kind == "binary":
            return to_continuous(raw, spec.eps)
        if spec.kind == "continuous":
            return minmax_to_continuous(raw, spec.lo, spec.hi)
        raise ValueError(f"_to_z not defined for categorical node {node!r}")

    def _to_raw(self, node: str, z: torch.Tensor) -> torch.Tensor:
        spec = self.specs[node]
        if spec.kind == "binary":
            return to_binary(to_prob(z))
        if spec.kind == "continuous":
            return minmax_to_raw(z, spec.lo, spec.hi)
        raise ValueError(f"_to_raw not defined for categorical node {node!r}")

    def _dist_params(self, node: str, context: torch.Tensor):
        mu_log_sigma = self.encoders[node](context)
        mu, log_sigma = mu_log_sigma[:, :1], mu_log_sigma[:, 1:]
        return mu, log_sigma.clamp(-5.0, 5.0)

    def nll(self, attrs: torch.Tensor, attr_index: Dict[str, int]) -> torch.Tensor:
        """Mean joint negative log-likelihood of a batch of real attribute rows.

        ``attrs`` columns are in each node's raw units (0/1 for binary, the
        attribute's native scale for continuous, the integer class index
        -- stored as a float -- for categorical).
        """
        batch_size, device = attrs.shape[0], attrs.device
        z_by_node = {node: (self._to_z(node, attrs[:, [attr_index[node]]])
                            if self.specs[node].kind != "categorical" else None)
                    for node in self.graph.attributes}
        total = torch.zeros((), device=device)
        for node in self.graph.attributes:
            spec = self.specs[node]
            context = self._context(node, z_by_node, batch_size, device)
            if spec.kind == "categorical":
                logits = self.encoders[node](context)
                target = attrs[:, attr_index[node]].long()
                total = total + F.cross_entropy(logits, target, reduction="mean")
            else:
                total = total - self.flows[node].log_prob(z_by_node[node], context=context).mean()
        return total

    @torch.no_grad()
    def abduct(self, attrs: torch.Tensor, attr_index: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """Exogenous noise per node from a real (batch, len(graph.attributes)) raw-units tensor."""
        batch_size, device = attrs.shape[0], attrs.device
        z_by_node = {node: (self._to_z(node, attrs[:, [attr_index[node]]])
                            if self.specs[node].kind != "categorical" else None)
                    for node in self.graph.attributes}
        eps_by_node = {}
        for node in self.graph.attributes:
            spec = self.specs[node]
            if spec.kind == "categorical":
                # Root-only (enforced in __init__): the noise IS the observed class.
                eps_by_node[node] = attrs[:, [attr_index[node]]].clone()
                continue
            context = self._context(node, z_by_node, batch_size, device)
            mu, log_sigma = self._dist_params(node, context)
            eps_by_node[node] = (z_by_node[node] - mu) / log_sigma.exp()
        return eps_by_node

    @torch.no_grad()
    def propagate(self, eps_by_node: Dict[str, torch.Tensor],
                  interventions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass in topological order: interventions forced, others reconstructed from noise."""
        batch_size = next(iter(eps_by_node.values())).shape[0]
        device = next(iter(eps_by_node.values())).device
        z_by_node: Dict[str, torch.Tensor] = {}
        for node in self.graph.topological_order():
            spec = self.specs[node]
            if spec.kind == "categorical":
                if node in interventions:
                    z_by_node[node] = interventions[node].to(device=device, dtype=torch.float32).view(batch_size, 1)
                else:
                    z_by_node[node] = eps_by_node[node]
                continue
            if node in interventions:
                forced = interventions[node].to(device=device, dtype=torch.float32).view(batch_size, 1)
                z_by_node[node] = self._to_z(node, forced)
                continue
            context = self._context(node, z_by_node, batch_size, device)
            mu, log_sigma = self._dist_params(node, context)
            z_by_node[node] = mu + eps_by_node[node] * log_sigma.exp()
        return z_by_node

    def counterfactual(self, attrs: torch.Tensor, attr_index: Dict[str, int],
                       interventions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Full abduct -> intervene -> predict: raw-units counterfactual value per node, shape (batch, 1)."""
        eps_by_node = self.abduct(attrs, attr_index)
        z_cf = self.propagate(eps_by_node, interventions)
        return {node: (z if self.specs[node].kind == "categorical" else self._to_raw(node, z))
               for node, z in z_cf.items()}

    def counterfactual_binary(self, attrs01: torch.Tensor, attr_index: Dict[str, int],
                              interventions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Back-compat alias for the all-binary case (item 1/2's CelebA path)."""
        return self.counterfactual(attrs01, attr_index, interventions)

    def save(self, path):
        torch.save({"state_dict": self.state_dict(), "attributes": self.graph.attributes,
                   "edges": self.graph.edges, "node_specs": {n: s.to_dict() for n, s in self.specs.items()}}, path)

    @classmethod
    def load(cls, path, device="cpu") -> "SCM":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        graph = CausalGraph(ckpt["attributes"], ckpt["edges"])
        if "node_specs" in ckpt:
            specs = {n: NodeSpec(**d) for n, d in ckpt["node_specs"].items()}
        else:
            specs = default_node_specs(graph, ckpt["eps"])  # pre-generalization checkpoints
        scm = cls(graph, specs).to(device)
        scm.load_state_dict(ckpt["state_dict"])
        scm.eval()
        return scm
