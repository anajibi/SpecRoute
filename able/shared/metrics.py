"""CC and FC, and the arithmetic for combining them.

Deliberately split into two layers:

  COMPONENTS (cc, fc) -- settled. How well an edit landed, and how much survived it.
  COMBINATION (CF1)   -- NOT settled. The protocol is still being defined, so this module offers
                         the combiners and refuses to pick a default.

That split exists because the two datasets silently diverged: both computed `hm(CC, FC)`, but one
fed it FC over all measured attributes and the other FC over the unconditioned ones, and the
difference never appeared in any single place a reader would look. Here the FC term is always named
in the output, so a table can never be ambiguous about which one it reports.

The other thing this pins down: CF1 is the MEAN OF PER-INTERVENTION harmonic means, not the
harmonic mean of pooled rates. Those differ by Jensen's inequality -- for one real cell, 0.7717
against 0.7756 -- and the pooled form lets a frequent intervention dominate a rare one. The
docstring in the original aggregate_metrics.py claimed the pooled form while the code did the
per-intervention one; `cf1_pooled` is provided only so a test can show the two apart.
"""
from __future__ import annotations
import math
from typing import Mapping, Sequence


def hm(a: float, b: float) -> float:
    """Harmonic mean, clipped to [0, 1] first.

    Clipping matters for Causal3DIdent: CC there is 1 - (error / trivial-baseline error), which goes
    negative when a model does worse than always answering the mean. A negative input makes the
    harmonic mean meaningless rather than merely bad.
    """
    a = min(max(a, 0.0), 1.0)
    b = min(max(b, 0.0), 1.0)
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def cf1_per_intervention(per: Mapping[str, Mapping[str, float]], fc_term: str = "fc_unobs") -> float:
    """Mean over interventions of each intervention's hm(cc, fc). The convention in use."""
    if not per:
        raise ValueError("no interventions to aggregate")
    missing = [k for k, v in per.items() if "cc" not in v or fc_term not in v]
    if missing:
        raise KeyError(f"interventions {missing} lack 'cc' or {fc_term!r}")
    return sum(hm(v["cc"], v[fc_term]) for v in per.values()) / len(per)


def cf1_pooled(per: Mapping[str, Mapping[str, float]], fc_term: str = "fc_unobs") -> float:
    """hm of the pooled means. NOT the convention -- kept so the difference is testable."""
    if not per:
        raise ValueError("no interventions to aggregate")
    n = len(per)
    return hm(sum(v["cc"] for v in per.values()) / n,
              sum(v[fc_term] for v in per.values()) / n)


def normalise_to_floor(measured: float, floor: float) -> float:
    """Express preservation against the ceiling a null intervention actually reaches.

    Preservation under do(A = its current value) is not 1.0 -- inversion, guidance and reader noise
    all cost something before any edit is requested. So preservation is reported as a RATIO to that
    measured ceiling. The difference form (measured - floor) / (1 - floor) was tried and discarded:
    it goes negative whenever a model lands below the floor, which happens.
    """
    if floor <= 0:
        raise ValueError(f"floor must be positive, got {floor}")
    return measured / floor


def aggregate(per: Mapping[str, Mapping[str, float]],
              fc_terms: Sequence[str] = ("fc_obs", "fc_unobs")) -> dict:
    """Pool per-intervention components into one record.

    Reports every FC term present and a CF1 for each, named so the FC term is never implicit. No
    single number is promoted to 'the' CF1 -- that choice belongs to the evaluation protocol.
    """
    if not per:
        raise ValueError("no interventions to aggregate")
    n = len(per)
    out = {"n_interventions": n,
           "cc": sum(v["cc"] for v in per.values()) / n,
           "per_intervention": dict(per)}
    for term in fc_terms:
        if all(term in v for v in per.values()):
            out[term] = sum(v[term] for v in per.values()) / n
            out[f"cf1_{term.removeprefix('fc_')}"] = cf1_per_intervention(per, term)
    return out
