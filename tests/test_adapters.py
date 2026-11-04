"""Tier 3 (cont.) -- the dataset adapters.

The guard worth having tests for is the graph-A/graph-B inversion: the attributes graph B conditions
on sit inside graph A's unobserved set, so reusing graph A's reader basis for a graph-B model counts
the intervened attributes as collateral damage and penalises an edit for succeeding.
"""
import numpy as np
import pytest
from able.adapters import for_dataset
from able.adapters.base import cc_from_error, fc_from_error
from able.adapters.celebahq import GRAPHS


def test_c3di_descendants_follow_the_graph():
    a = for_dataset("c3di")()
    assert a.descendants("class") == {"rot_obj", "pos_obj"}
    assert a.descendants("pos_spl") == {"pos_obj"}
    assert a.descendants("hue_bg") == set()
    assert a.descendants("pos_obj") == set(), "pos_obj is a sink"


def test_c3di_conditioned_vs_measured():
    """Four conditioned, three measured-only -- the three hues are the built-in FC_unobs set."""
    specs = for_dataset("c3di")().attrs()
    assert sum(s.conditioned for s in specs) == 4
    assert [s.name for s in specs if not s.conditioned] == ["hue_obj", "hue_spl", "hue_bg"]


def test_c3di_angular_distance_wraps():
    """A hue at +pi and one at -pi are identical, not two pi apart."""
    a = for_dataset("c3di")()
    spec = next(s for s in a.attrs() if s.name == "hue_bg")
    d = a.distance(spec, np.array([np.pi - 0.01]), np.array([-np.pi + 0.01]))
    assert d[0] == pytest.approx(0.02, abs=1e-6)


def test_c3di_categorical_distance_is_a_flip():
    a = for_dataset("c3di")()
    spec = next(s for s in a.attrs() if s.name == "class")
    d = a.distance(spec, np.array([0, 1, 2]), np.array([0, 2, 2]))
    assert list(d) == [0.0, 1.0, 0.0]


def test_trivial_error_sets_the_scale():
    """Without this normalisation an attribute in radians dominates one in [0,1] for no reason
    but its units."""
    a = for_dataset("c3di")()
    cont = next(s for s in a.attrs() if s.name == "pos_spl")
    y = np.array([0.0, 1.0, 2.0, 3.0])
    assert a.trivial_error(cont, y) == pytest.approx(1.0)  # mean |y - 1.5|
    cat = next(s for s in a.attrs() if s.name == "class")
    assert a.trivial_error(cat, np.array([0, 0, 0, 1])) == pytest.approx(0.25)


def test_cc_is_one_when_perfect_and_zero_at_the_constant_predictor():
    assert cc_from_error(0.0, 0.5) == pytest.approx(1.0)
    assert cc_from_error(0.5, 0.5) == pytest.approx(0.0)
    assert cc_from_error(0.75, 0.5) == pytest.approx(-0.5), "worse than constant stays visible"
    assert fc_from_error(0.0, 0.5) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cc_from_error(0.1, 0.0)


def test_celebahq_graphs_are_what_the_configs_say():
    assert GRAPHS["A"]["conditioned"] == ("Male", "Young", "Beard", "Bald")
    assert GRAPHS["B"]["conditioned"] == ("Eyeglasses", "Smiling")
    assert GRAPHS["B"]["edges"] == (), "graph B is deliberately edgeless"


def test_celebahq_graph_a_descendants():
    a = for_dataset("celebahq")("A", unobs=["Smiling"])
    assert a.descendants("Male") == {"Beard", "Bald"}
    assert a.descendants("Young") == {"Beard", "Bald"}
    assert a.descendants("Beard") == set()


def test_celebahq_graph_b_has_nothing_to_propagate():
    """With no edges every intervention is licensed to change exactly one thing, which is the
    point of the graph."""
    b = for_dataset("celebahq")("B", unobs=["Male", "Young"])
    assert b.descendants("Eyeglasses") == set()
    assert b.descendants("Smiling") == set()


def test_the_graph_inversion_is_rejected():
    """Scoring graph B with graph A's basis would count the intervened attributes as damage."""
    with pytest.raises(ValueError, match="conditioned under graph B"):
        for_dataset("celebahq")("B", unobs=["Eyeglasses", "Male"])
    with pytest.raises(ValueError, match="conditioned under graph A"):
        for_dataset("celebahq")("A", unobs=["Male", "Smiling"])


def test_celebahq_unobs_excludes_conditioned_silently_only_when_disjoint():
    a = for_dataset("celebahq")("A", unobs=["Smiling", "Eyeglasses", "Bangs"])
    assert a.unobs == ("Smiling", "Eyeglasses", "Bangs")
    assert all(x not in a.conditioned for x in a.unobs)


def test_unknown_graph_and_dataset_raise():
    with pytest.raises(KeyError):
        for_dataset("celebahq")("C")
    with pytest.raises(KeyError):
        for_dataset("morphomnist")


def test_binary_preservation_is_an_agreement_rate():
    """Not a constant-predictor normalisation: Bald at 2.4% prevalence would look perfectly
    preserved by a reader that always answers 'no'."""
    A = for_dataset("celebahq")
    y = np.array([1, 1, 0, 0])
    assert A.preservation_rate(y, np.array([1, 1, 0, 0])) == 1.0
    assert A.preservation_rate(y, np.array([1, 0, 0, 0])) == pytest.approx(0.75)
