"""Causal3DIdent: one categorical factor, six continuous.

The graph conditions on four (class, spotlight position, object position, object rotation). The
three hue factors are measured but never conditioned, which makes them a ready-made FC_unobs set --
anything that moves there is collateral damage by construction.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from .base import AttrSpec

# Ranges are the rendered factor ranges; rotation and the hues are angular in [-pi, pi].
_PI = float(np.pi)
_SPECS = (
    AttrSpec("class",    "categorical", True,  n_classes=7),
    AttrSpec("pos_spl",  "continuous",  True,  lo=-1.0, hi=1.0),
    AttrSpec("pos_obj",  "continuous",  True,  lo=-1.0, hi=1.0),
    AttrSpec("rot_obj",  "continuous",  True,  lo=-_PI, hi=_PI),
    AttrSpec("hue_obj",  "continuous",  False, lo=-_PI, hi=_PI),
    AttrSpec("hue_spl",  "continuous",  False, lo=-_PI, hi=_PI),
    AttrSpec("hue_bg",   "continuous",  False, lo=-_PI, hi=_PI),
)

# class -> rot_obj, class -> pos_obj, pos_spl -> pos_obj
EDGES = (("class", "rot_obj"), ("class", "pos_obj"), ("pos_spl", "pos_obj"))


class C3DIAdapter:
    dataset = "c3di"

    def attrs(self) -> Sequence[AttrSpec]:
        return _SPECS

    def descendants(self, name: str) -> set:
        """Attributes an intervention on `name` is licensed to move.

        Excluded from preservation rather than counted as failures -- a descendant that did not move
        would be the bug.
        """
        out, frontier = set(), [name]
        while frontier:
            cur = frontier.pop()
            for a, b in EDGES:
                if a == cur and b not in out:
                    out.add(b)
                    frontier.append(b)
        return out

    def distance(self, spec: AttrSpec, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = np.asarray(a), np.asarray(b)
        if spec.kind in ("binary", "categorical"):
            return (a != b).astype(float)
        d = np.abs(a - b)
        if spec.name.startswith(("rot", "hue")):
            # angular: 350 deg and 10 deg are 20 apart, not 340
            span = spec.hi - spec.lo
            d = np.minimum(d, span - d)
        return d if d.ndim == 1 else d.mean(axis=tuple(range(1, d.ndim)))

    def trivial_error(self, spec: AttrSpec, y: np.ndarray) -> float:
        y = np.asarray(y)
        if spec.kind in ("binary", "categorical"):
            _, counts = np.unique(y, return_counts=True)
            return 1.0 - counts.max() / len(y)
        return float(np.abs(y - y.mean()).mean())
