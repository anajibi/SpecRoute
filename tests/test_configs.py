"""Tier 1 -- every registered config builds, and the ladder invariants hold.

These are the invariants that, when violated, do not raise: a level-dims list that does not sum to
512 silently changes the latent budget, and a block_to_level that skips a decoder block leaves that
block unmodulated. Both would train happily and produce a model that is not the one intended.
"""
import pytest
from able import registry as R


@pytest.mark.parametrize("name", sorted(R.MODELS))
def test_config_builds_claimed_architecture(name):
    r = R.verify(R.get(name))
    assert r["ok"], (
        f"{name}: config produces {r['n_keys']} keys, registry claims {r['expected']}. "
        "Either the config changed after the model was trained, or the registry is wrong. "
        "A load would half-initialise the model without raising.")


@pytest.mark.parametrize("name", sorted(R.MODELS))
def test_ladder_invariants(name):
    m = R.get(name)
    _, cfg = R.build(m)
    e = cfg.raw["encoder"]
    dims, b2l = e["hier_level_dims"], e["hier_block_to_level"]
    k = len(e["hier_tap_block_ids"])

    assert len(dims) == k, f"{name}: {len(dims)} level dims for {k} taps"
    assert sum(dims) == 512, f"{name}: level dims sum to {sum(dims)}, not the fixed 512 budget"
    assert len(b2l) == e["n_decoder_output_blocks"], (
        f"{name}: block_to_level has {len(b2l)} entries for "
        f"{e['n_decoder_output_blocks']} decoder blocks")
    assert max(b2l) == k - 1, f"{name}: top level {max(b2l)} but {k} taps -- a tap feeds nothing"
    assert set(b2l) == set(range(k)), f"{name}: levels {sorted(set(b2l))} skip one of 0..{k-1}"
    assert all(b2l[i] <= b2l[i + 1] for i in range(len(b2l) - 1)), (
        f"{name}: block_to_level is not monotone; taps must enter in depth order")


@pytest.mark.parametrize("name", sorted(R.MODELS))
def test_attr_embed_divides_by_attribute_count(name):
    """concat_film splits attr_embed_dim per attribute, so an indivisible value truncates."""
    m = R.get(name)
    _, cfg = R.build(m)
    e = cfg.raw["encoder"]
    d, n = e["attr_embed_dim"], e["n_attributes"]
    assert d % n == 0, (
        f"{name}: attr_embed_dim {d} is not divisible by {n} attributes; "
        "concat_film splits it per attribute")
    assert n == len(m.attrs), f"{name}: config says {n} attributes, registry lists {len(m.attrs)}"


# Per-attribute width differs BY DATASET -- C3DI uses 32, CelebA-HQ 96 -- because the CelebA
# protocol tuned it later. What has to hold is that it is constant WITHIN a dataset, or the rungs
# of that ladder differ by conditioning capacity as well as by K and the comparison is confounded.
PER_ATTR = {"c3di": 32, "celebahq": 96}


@pytest.mark.parametrize("dataset,expected", sorted(PER_ATTR.items()))
def test_per_attribute_width_is_constant_within_a_dataset(dataset, expected):
    widths = {}
    for m in R.by_dataset(dataset):
        _, cfg = R.build(m)
        e = cfg.raw["encoder"]
        widths[m.name] = e["attr_embed_dim"] // e["n_attributes"]
    assert set(widths.values()) == {expected}, (
        f"{dataset}: per-attribute width varies across the ladder -- {widths}. "
        "Rungs would differ by conditioning capacity as well as K.")


def test_graph_b_narrows_the_embedding_to_hold_96_per_attribute():
    """Graph B has two attributes, so it uses 192 rather than graph A's 384. Leaving it at 384
    would have silently given each attribute a 192-dim slice -- double graph A's."""
    a = R.build(R.get("celebahq_a_k11"))[1].raw["encoder"]
    b = R.build(R.get("celebahq_b_k11"))[1].raw["encoder"]
    assert (a["attr_embed_dim"], a["n_attributes"]) == (384, 4)
    assert (b["attr_embed_dim"], b["n_attributes"]) == (192, 2)
    assert a["attr_embed_dim"] // a["n_attributes"] == b["attr_embed_dim"] // b["n_attributes"] == 96
