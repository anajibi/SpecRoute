"""Tier 3 -- the metric arithmetic, against cases worked by hand.

The point of these is to pin the definitions that drifted: which FC term pairs with CC, whether CF1
is per-intervention or pooled, and how preservation is normalised against the composition floor.
"""
import math
import pytest
from able.shared.metrics import (hm, cf1_per_intervention, cf1_pooled,
                                 normalise_to_floor, aggregate)

# One real cell: Causal3DIdent K=1 at g=1, whose stored CF1_obs is 0.7717.
REAL = {
    "class":   {"cc": 0.5166, "fc_obs": 0.9371},
    "pos_spl": {"cc": 0.7849, "fc_obs": 0.9739},
    "pos_obj": {"cc": 0.6816, "fc_obs": 0.8986},
    "rot_obj": {"cc": 0.6883, "fc_obs": 0.8899},
}


def test_harmonic_mean_basics():
    assert hm(1.0, 1.0) == 1.0
    assert hm(0.0, 1.0) == 0.0
    assert hm(0.5, 0.5) == pytest.approx(0.5)
    assert hm(0.6, 0.9) == pytest.approx(2 * 0.6 * 0.9 / 1.5)


def test_harmonic_mean_clips_negative_cc():
    """C3DI's CC is 1 - err/trivial and goes negative when a model loses to the constant
    predictor. Unclipped, that makes the harmonic mean meaningless rather than merely bad."""
    assert hm(-0.3, 0.9) == 0.0
    assert hm(1.4, 0.8) == pytest.approx(hm(1.0, 0.8))


def test_cf1_is_per_intervention_not_pooled():
    """The convention in use, and the one the stale docstring got wrong."""
    assert cf1_per_intervention(REAL, "fc_obs") == pytest.approx(0.7717, abs=1e-4)
    assert cf1_pooled(REAL, "fc_obs") == pytest.approx(0.7756, abs=1e-4)
    assert cf1_per_intervention(REAL, "fc_obs") != pytest.approx(cf1_pooled(REAL, "fc_obs"), abs=1e-4)


def test_pooled_lets_a_frequent_intervention_dominate():
    """Why the per-intervention form is the right one: with one easy and one hard intervention,
    pooling hides the hard one."""
    per = {"easy": {"cc": 0.99, "fc_obs": 0.99}, "hard": {"cc": 0.20, "fc_obs": 0.99}}
    assert cf1_per_intervention(per, "fc_obs") < cf1_pooled(per, "fc_obs")


def test_cf1_weights_interventions_equally():
    per = {"a": {"cc": 0.9, "fc_obs": 0.9}, "b": {"cc": 0.3, "fc_obs": 0.9}}
    expect = (hm(0.9, 0.9) + hm(0.3, 0.9)) / 2
    assert cf1_per_intervention(per, "fc_obs") == pytest.approx(expect)


def test_floor_normalisation_is_a_ratio_not_a_difference():
    """Preservation under a null intervention is below 1, so scores are read against that measured
    ceiling. The difference form (m - f)/(1 - f) was tried and discarded: it goes negative whenever
    a model lands under the floor, which happens."""
    assert normalise_to_floor(0.9232, 0.9232) == pytest.approx(1.0)
    assert normalise_to_floor(0.7117, 0.9232) == pytest.approx(0.7709, abs=1e-4)
    below = normalise_to_floor(0.80, 0.92)
    assert 0 < below < 1, "a score under the floor must stay positive and interpretable"
    bad = (0.80 - 0.92) / (1 - 0.92)
    assert bad < 0, "the discarded difference form goes negative, which is why it was discarded"


def test_aggregate_names_every_fc_term():
    """No single CF1 is promoted: the FC term is always in the key, so a table cannot be
    ambiguous about which preservation set it used."""
    per = {k: {**v, "fc_unobs": v["fc_obs"] - 0.05} for k, v in REAL.items()}
    out = aggregate(per)
    assert "cf1_obs" in out and "cf1_unobs" in out
    assert "cf1" not in out, "aggregate must not pick a default CF1; that is a protocol decision"
    assert out["n_interventions"] == 4
    assert out["cc"] == pytest.approx(sum(v["cc"] for v in per.values()) / 4)
    assert out["cf1_obs"] == pytest.approx(0.7717, abs=1e-4)


def test_aggregate_rejects_empty_and_missing_terms():
    with pytest.raises(ValueError):
        aggregate({})
    with pytest.raises(KeyError):
        cf1_per_intervention({"a": {"cc": 0.5}}, "fc_unobs")


def test_floor_must_be_positive():
    with pytest.raises(ValueError):
        normalise_to_floor(0.5, 0.0)
