"""What a dataset adapter has to answer.

Everything downstream of these questions -- the guidance sweep, the null-intervention control, the
CC/FC arithmetic, the output schema -- is shared. The adapter exists only because the two datasets
disagree on what an attribute *is*:

  Causal3DIdent  one categorical factor and six continuous ones, read off regressors, so "did it
                 change" is a distance in the factor's own units and has to be normalised before it
                 can sit beside an accuracy.
  CelebA-HQ      binary annotations read off classifiers, so "did it change" is a label flip, and
                 the reader's own error rate bounds what can be claimed.

Mixing those two notions is what produced metrics that could not be compared across datasets.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np


@dataclass(frozen=True)
class AttrSpec:
    """One attribute, and how to decide whether it moved."""
    name: str
    kind: str              # "binary" | "categorical" | "continuous"
    conditioned: bool      # in the causal graph the model conditions on
    n_classes: int = 0     # categorical only
    lo: float = 0.0        # continuous range, for normalisation
    hi: float = 1.0


class Adapter(Protocol):
    dataset: str

    def attrs(self) -> Sequence[AttrSpec]:
        """Every attribute this dataset can measure, conditioned or not."""

    def distance(self, spec: AttrSpec, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Per-sample disagreement between two readings of the same attribute.

        0 means identical. Binary and categorical give 0/1; continuous gives absolute error in the
        attribute's own units, which `trivial_error` then normalises.
        """

    def trivial_error(self, spec: AttrSpec, y: np.ndarray) -> float:
        """Error of the best constant predictor, the scale that makes attributes comparable.

        For a continuous factor that is E|y - mean(y)|; for a categorical one it is 1 - mode rate.
        Dividing a measured error by this is what lets an accuracy and an MAE be averaged without
        the result being meaningless. Without it, an attribute measured in radians dominates one
        measured in [0,1] for no reason but its units.
        """


def cc_from_error(err: float, trivial: float) -> float:
    """Turn a residual error into 'did the edit land', on [0, 1].

    1.0 is a perfect edit; 0.0 is no better than the constant predictor. Negative values mean worse
    than constant and are left unclipped here so they stay visible -- the clip happens in `hm`,
    where a negative would otherwise make the harmonic mean nonsense.
    """
    if trivial <= 0:
        raise ValueError(f"trivial error must be positive, got {trivial}")
    return 1.0 - err / trivial


def fc_from_error(err: float, trivial: float) -> float:
    """Same normalisation for preservation: 1.0 means nothing moved."""
    return cc_from_error(err, trivial)
