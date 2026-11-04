"""CelebA-HQ: binary annotations, read off per-attribute classifiers.

Two causal graphs over the same images. Which attributes count as conditioned INVERTS between them,
and that inversion is the thing most likely to be got wrong: Eyeglasses and Smiling sit inside graph
A's unobserved set, so scoring a graph-B model with graph A's basis counts the intervened attributes
as collateral damage and penalises an edit for working.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from .base import AttrSpec

GRAPHS = {
    "A": {"conditioned": ("Male", "Young", "Beard", "Bald"),
          "edges": (("Male", "Beard"), ("Male", "Bald"),
                    ("Young", "Beard"), ("Young", "Bald"))},
    "B": {"conditioned": ("Eyeglasses", "Smiling"), "edges": ()},
}


class CelebAHQAdapter:
    dataset = "celebahq"

    def __init__(self, graph: str = "A", unobs: Sequence[str] = ()):
        if graph not in GRAPHS:
            raise KeyError(f"unknown graph {graph!r}; have {', '.join(GRAPHS)}")
        self.graph = graph
        self.conditioned = GRAPHS[graph]["conditioned"]
        self.edges = GRAPHS[graph]["edges"]
        # The screened reader set, which differs per graph. Passed in rather than hardcoded so the
        # caller loads reader_keep.json / reader_keep_graphB.json and cannot mix them up silently.
        self.unobs = tuple(a for a in unobs if a not in self.conditioned)
        bad = [a for a in unobs if a in self.conditioned]
        if bad:
            raise ValueError(
                f"{bad} are conditioned under graph {graph} and cannot also be FC_unobs; "
                "this is the graph-A/graph-B inversion -- use the matching reader_keep file")

    def attrs(self) -> Sequence[AttrSpec]:
        return tuple(AttrSpec(a, "binary", True) for a in self.conditioned) + \
               tuple(AttrSpec(a, "binary", False) for a in self.unobs)

    def descendants(self, name: str) -> set:
        out, frontier = set(), [name]
        while frontier:
            cur = frontier.pop()
            for a, b in self.edges:
                if a == cur and b not in out:
                    out.add(b)
                    frontier.append(b)
        return out

    def distance(self, spec: AttrSpec, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (np.asarray(a) != np.asarray(b)).astype(float)

    def trivial_error(self, spec: AttrSpec, y: np.ndarray) -> float:
        y = np.asarray(y)
        _, counts = np.unique(y, return_counts=True)
        return 1.0 - counts.max() / len(y)

    @staticmethod
    def preservation_rate(a: np.ndarray, b: np.ndarray) -> float:
        """Fraction of samples whose label is unchanged.

        Binary attributes are reported as an agreement rate, not as a normalised error -- the
        constant-predictor normalisation C3DI needs would make a 2.4%-prevalence attribute look
        perfectly preserved by a reader that always answers 'no'.
        """
        return float((np.asarray(a) == np.asarray(b)).mean())
