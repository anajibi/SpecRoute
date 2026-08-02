"""Per-attribute normalizing-flow SCM: fit/abduct/propagate over a small causal DAG.

Each attribute node is a scalar (CelebA attributes are binary), so its
"flow" is the simplest valid instance of a conditional normalizing flow: a
conditional diagonal Gaussian in logit space, mean/log-std produced by a
small MLP over the node's parents' logit-space values (``nflows``'s
``ConditionalDiagonalNormal`` with an identity transform stack -- there's no
room for a coupling/autoregressive transform on a 1-D target). This is the
"Deep SCM" pattern (Pawlowski et al.) scoped to scalar attribute nodes;
richer per-node transforms are a natural drop-in if a future node represents
a multi-dimensional attribute group. Root nodes (no parents) get a constant
context, making them a learnable *unconditional* Gaussian -- not fixed
N(0,1).

Counterfactual generation follows Pearl's three-step recipe:
1. abduct: from a real image's observed attribute vector, recover each
   node's exogenous noise (the part of its value not explained by its
   parents).
2. intervene: force the target node(s) to specific values.
3. predict/propagate: walk the graph in topological order, recomputing each
   non-intervened node's value from its (possibly-intervened) parents and
   its own preserved exogenous noise.
"""
from typing import Dict

import torch
import torch.nn as nn
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform

from .graph import CausalGraph
from .normalize import to_binary, to_continuous, to_prob

_ROOT_CONTEXT_DIM = 1  # constant input -> learnable unconditional Gaussian for parentless nodes


class _ContextEncoder(nn.Module):
    def __init__(self, context_dim: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(context_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, context):
        return self.net(context)


class SCM(nn.Module):
    """Structural causal model over ``graph``'s attributes, in logit space."""

    def __init__(self, graph: CausalGraph, eps: float):
        super().__init__()
        self.graph = graph
        self.eps = float(eps)
        self.encoders = nn.ModuleDict()
        self.flows = nn.ModuleDict()
        for node in graph.attributes:
            ctx_dim = max(_ROOT_CONTEXT_DIM, len(graph.parents(node)))
            enc = _ContextEncoder(ctx_dim)
            self.encoders[node] = enc
            self.flows[node] = Flow(CompositeTransform([]), ConditionalDiagonalNormal(shape=[1], context_encoder=enc))

    def _context(self, node: str, z_by_node: Dict[str, torch.Tensor], batch_size: int, device) -> torch.Tensor:
        parents = self.graph.parents(node)
        if not parents:
            return torch.ones(batch_size, _ROOT_CONTEXT_DIM, device=device)
        return torch.cat([z_by_node[p] for p in parents], dim=-1)

    def _dist_params(self, node: str, context: torch.Tensor):
        mu_log_sigma = self.encoders[node](context)
        mu, log_sigma = mu_log_sigma[:, :1], mu_log_sigma[:, 1:]
        return mu, log_sigma.clamp(-5.0, 5.0)

    def nll(self, attrs01: torch.Tensor, attr_index: Dict[str, int]) -> torch.Tensor:
        """Mean negative log-likelihood of a batch of real (0/1) attribute rows."""
        z_by_node = {node: to_continuous(attrs01[:, [attr_index[node]]], self.eps) for node in self.graph.attributes}
        total = 0.0
        for node in self.graph.attributes:
            context = self._context(node, z_by_node, attrs01.shape[0], attrs01.device)
            total = total - self.flows[node].log_prob(z_by_node[node], context=context)
        return total.mean()

    @torch.no_grad()
    def abduct(self, attrs01: torch.Tensor, attr_index: Dict[str, int]) -> Dict[str, torch.Tensor]:
        """Exogenous noise per node from a real (batch, len(graph.attributes)) 0/1 tensor."""
        z_by_node = {node: to_continuous(attrs01[:, [attr_index[node]]], self.eps) for node in self.graph.attributes}
        eps_by_node = {}
        for node in self.graph.attributes:
            context = self._context(node, z_by_node, attrs01.shape[0], attrs01.device)
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
            if node in interventions:
                forced = interventions[node].to(device=device, dtype=torch.float32).view(batch_size, 1)
                z_by_node[node] = to_continuous(forced, self.eps)
                continue
            context = self._context(node, z_by_node, batch_size, device)
            mu, log_sigma = self._dist_params(node, context)
            z_by_node[node] = mu + eps_by_node[node] * log_sigma.exp()
        return z_by_node

    def counterfactual_binary(self, attrs01: torch.Tensor, attr_index: Dict[str, int],
                              interventions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Full abduct -> intervene -> predict: binary {0,1} counterfactual value per node, shape (batch, 1)."""
        eps_by_node = self.abduct(attrs01, attr_index)
        z_cf = self.propagate(eps_by_node, interventions)
        return {node: to_binary(to_prob(z)) for node, z in z_cf.items()}

    def save(self, path):
        torch.save({"state_dict": self.state_dict(), "attributes": self.graph.attributes,
                   "edges": self.graph.edges, "eps": self.eps}, path)

    @classmethod
    def load(cls, path, device="cpu") -> "SCM":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        graph = CausalGraph(ckpt["attributes"], ckpt["edges"])
        scm = cls(graph, ckpt["eps"]).to(device)
        scm.load_state_dict(ckpt["state_dict"])
        scm.eval()
        return scm
