"""3DShapes Graph A: config, exact SCM, and the frozen layout's own invariants."""
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from able.datasets.shapes3d import scm_exact as sx

CFG_DIR = os.path.join(ROOT, "able/datasets/shapes3d/configs")
ARMS = {"shapes3d_a_k1_d256": (1, 256), "shapes3d_a_k5_d256": (5, 256),
        "shapes3d_a_k11_d256": (11, 256), "shapes3d_a_k11_d128": (11, 128)}
LAYOUT_DIR = "/home/exouser/able_data/shapes3d/graph_a/v1"
H5 = "/home/exouser/able_data/src_3dshapes/3dshapes.h5"
N = 16_000_000


def _raw(name):
    with open(os.path.join(CFG_DIR, name + ".yaml")) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ configs


@pytest.mark.parametrize("name", sorted(ARMS))
def test_ladder_invariants(name):
    k, budget = ARMS[name]
    e = _raw(name)["encoder"]
    assert len(e["hier_level_dims"]) == k
    assert sum(e["hier_level_dims"]) == budget
    assert len(e["hier_tap_block_ids"]) == k
    assert len(e["hier_block_to_level"]) == e["n_decoder_output_blocks"] == 12
    assert max(e["hier_block_to_level"]) == k - 1
    assert sorted(set(e["hier_block_to_level"])) == list(range(k)), "every level must be used"
    assert e["n_attributes"] == 4 == len(e["conditioning_attrs"])
    assert list(e["conditioning_attrs"]) == list(sx.MODELED)


def test_k5_taps_are_nested_in_k11():
    k5 = [t for t in _raw("shapes3d_a_k5_d256")["encoder"]["hier_tap_block_ids"] if t != "mid"]
    k11 = [t for t in _raw("shapes3d_a_k11_d256")["encoder"]["hier_tap_block_ids"] if t != "mid"]
    assert set(k5) <= set(k11), "the ladder must be nested or depth is not the only variable"


@pytest.mark.parametrize("name", sorted(ARMS))
def test_one_pass_budget(name):
    """max_steps must be exactly one pass, or the arm is not comparable to the others."""
    t = _raw(name)["train"]
    eff = t["batch_size_per_gpu"] * int(t.get("accum_batches", 1))
    assert t["total_batch_size"] == eff
    assert t["max_steps"] * eff == N, (t["max_steps"], eff)


@pytest.mark.parametrize("name", sorted(ARMS))
def test_recipe_is_identical_across_arms(name):
    """Every training hyperparameter must match, or depth/budget is confounded."""
    ref = _raw("shapes3d_a_k1_d256")
    got = _raw(name)
    for key in ("batch_size_per_gpu", "lr", "ema_decay", "T", "T_eval", "max_steps",
                "precision", "grad_clip", "checkpoint_every_n_steps"):
        assert got["train"][key] == ref["train"][key], key
    for key in ("attr_embed_dim", "attr_dropout_prob", "fourier_freqs", "attr_norm"):
        assert got["encoder"][key] == ref["encoder"][key], key
    assert got["conditioning"] == ref["conditioning"]
    assert got["seed"] == ref["seed"]


@pytest.mark.parametrize("name", sorted(ARMS))
def test_no_adversary(name):
    """The adversarial path must be absent, not merely disabled."""
    e = _raw(name)["encoder"]
    assert not [k for k in e if k.startswith("adv_tap")], "adv_tap_* keys present"


def test_graph_nodes_all_categorical_without_range():
    """Categorical nodes whose raw storage IS the class index must carry no `range`, or
    load_cond_specs takes the binning path and mis-bins an index."""
    g = yaml.safe_load(open(os.path.join(ROOT, "able/datasets/shapes3d/"
                                               "causal_graph_shapes3d_a.yaml")))
    assert list(g["attributes"]) == list(sx.MODELED)
    for name, node in g["nodes"].items():
        assert node["kind"] == "categorical", name
        assert node["num_classes"] == sx.CARDINALITY[name], name
        assert "range" not in node, f"{name}: range would mis-bin an already-integer index"
    assert {tuple(e) for e in g["edges"]} == {("shape", "scale"), ("scale", "floor_hue"),
                                              ("object_hue", "floor_hue")}


# ------------------------------------------------------------------ exact SCM


def test_index_roundtrip_whole_grid():
    idx = np.arange(sx.N_GRID)
    assert np.array_equal(sx.grid_index(*sx.decode_index(idx)), idx)


def test_grid_index_matches_frozen_generator():
    """The frozen generator keeps its own copy of the index formula on purpose (it must stay
    self-contained). This is the test that stops the two drifting apart."""
    sys.path.insert(0, os.path.join(ROOT, "able/datasets/shapes3d"))
    import build_graph_a as gen
    rng = np.random.RandomState(0)
    f = [rng.randint(0, s, 5000) for s in sx.SIZES]
    assert np.array_equal(sx.grid_index(*f), gen.grid_index(*f))


def test_abduce_forward_roundtrip():
    rng = np.random.RandomState(1)
    lay = sx.sample(50000, rng)
    S, Ho, C, Hf = lay[:, 0], lay[:, 1], lay[:, 2], lay[:, 3]
    U_C, U_F = sx.abduce(S, Ho, C, Hf)
    C2, Hf2 = sx.forward(S, Ho, U_C, U_F)
    assert np.array_equal(C2, C) and np.array_equal(Hf2, Hf)


def test_counterfactual_roundtrip_all_interventions():
    """do(x') then do(x) returns the original tuple, for every modeled variable and value."""
    rng = np.random.RandomState(2)
    lay = sx.sample(2000, rng)
    S, Ho, C, Hf, Hw, R = (lay[:, i] for i in range(6))
    for var in sx.MODELED:
        for v in range(sx.CARDINALITY[var]):
            a = sx.intervene(S, Ho, C, Hf, var, v)
            orig = {"shape": S, "object_hue": Ho, "scale": C, "floor_hue": Hf}[var]
            for v0 in np.unique(orig):
                m = orig == v0
                b = sx.intervene(a[0][m], a[1][m], a[2][m], a[3][m], var, v0)
                assert np.array_equal(b[0], S[m]) and np.array_equal(b[1], Ho[m])
                assert np.array_equal(b[2], C[m]) and np.array_equal(b[3], Hf[m])
            ci = sx.counterfactual_index(S, Ho, C, Hf, Hw, R, var, v)
            assert ci.min() >= 0 and ci.max() < sx.N_GRID


def test_intervention_preserves_exogenous_and_non_descendants():
    rng = np.random.RandomState(3)
    lay = sx.sample(5000, rng)
    S, Ho, C, Hf = (lay[:, i] for i in range(4))
    # do(floor_hue) has no descendants: nothing upstream may move
    S2, Ho2, C2, _ = sx.intervene(S, Ho, C, Hf, "floor_hue", 3)
    assert np.array_equal(S2, S) and np.array_equal(Ho2, Ho) and np.array_equal(C2, C)
    # do(scale) must not touch shape, and must move floor_hue by exactly the scale delta
    S3, Ho3, C3, Hf3 = sx.intervene(S, Ho, C, Hf, "scale", 5)
    assert np.array_equal(S3, S) and np.array_equal(Ho3, Ho) and (C3 == 5).all()
    assert np.array_equal(Hf3, (Hf + (5 - C)) % 10)


def test_scale_wrap_is_cyclic_not_ordered():
    """scale wraps mod 8, so shape 0 -> 3 shifts a scale-7 object to index 6, not off the end.
    This is why scale must be embedded, never encoded with a linear or Fourier metric."""
    S = np.zeros(1, dtype=np.int64)
    U_C = np.array([7])
    C, _ = sx.forward(S, np.zeros(1, dtype=np.int64), U_C, np.zeros(1, dtype=np.int64))
    assert C[0] == 7
    C2, _ = sx.forward(np.array([3]), np.zeros(1, dtype=np.int64), U_C,
                       np.zeros(1, dtype=np.int64))
    assert C2[0] == (7 + 7) % 8 == 6


def test_noise_pmfs():
    for K, p in ((8, sx.P8), (10, sx.P10)):
        assert len(p) == K and abs(p.sum() - 1.0) < 1e-12
        assert p[0] == 0.25 and p[1] == 0.125 and p[K - 1] == 0.125
        assert (p > 0).all(), "full support is what makes every counterfactual in-support"


# ------------------------------------------------------------------ frozen layout


@pytest.mark.skipif(not os.path.exists(LAYOUT_DIR), reason="frozen layout not present")
def test_layout_obeys_the_mechanisms():
    lay = np.load(f"{LAYOUT_DIR}/train_layout.npy", mmap_mode="r")[:500000].astype(np.int64)
    idx = np.load(f"{LAYOUT_DIR}/train_image_idx.npy", mmap_mode="r")[:500000].astype(np.int64)
    S, Ho, C, Hf, Hw, R = (lay[:, i] for i in range(6))
    U_C, U_F = sx.abduce(S, Ho, C, Hf)
    C2, Hf2 = sx.forward(S, Ho, U_C, U_F)
    assert np.array_equal(C2, C) and np.array_equal(Hf2, Hf)
    assert np.array_equal(sx.grid_index(Hf, Hw, Ho, C, S, R), idx), \
        "train_image_idx must equal the index of its own factor tuple"


@pytest.mark.skipif(not os.path.exists(LAYOUT_DIR), reason="frozen layout not present")
def test_layout_length_matches_configs():
    n = len(np.load(f"{LAYOUT_DIR}/train_image_idx.npy", mmap_mode="r"))
    assert n == N
    assert np.load(f"{LAYOUT_DIR}/heldout_cells.npy").shape == (48000,)


@pytest.mark.skipif(not os.path.exists(LAYOUT_DIR), reason="frozen layout not present")
def test_heldout_decode_matches_grid():
    """The datamodule builds held-out val attrs by decoding the index, with no labels lookup."""
    cells = np.load(f"{LAYOUT_DIR}/heldout_cells.npy")[:1000]
    Hf, Hw, Ho, C, S, R = sx.decode_index(cells)
    assert np.array_equal(sx.grid_index(Hf, Hw, Ho, C, S, R), cells)
    for name, col in zip(sx.LAYOUT_COLUMNS, (S, Ho, C, Hf, Hw, R)):
        assert col.min() >= 0 and col.max() < sx.CARDINALITY[name], name


# ------------------------------------------------------------------ model builds


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(ARMS))
def test_config_builds_model(name):
    from experiments.hdae.hdae.config_io import load_hdae_config
    c = load_hdae_config(os.path.join(CFG_DIR, name + ".yaml"),
                         require_data=os.path.exists(H5))
    e = c.hdae_conf.encoder
    assert list(e.hier_level_dims) == _raw(name)["encoder"]["hier_level_dims"]
    assert len(e.cond_specs) == 4
    assert all(s.kind == "categorical" for s in e.cond_specs)
    assert [s.num_classes for s in e.cond_specs] == [4, 10, 8, 10]
    assert all(s.lo is None and s.hi is None for s in e.cond_specs), \
        "an already-integer index must take the round() path, not the binning path"


@pytest.mark.slow
def test_fourier_freqs_is_inert_with_all_categorical_attrs():
    """Justifies setting fourier_freqs: 0 -- with no continuous attr it changes no parameter."""
    import tempfile
    from experiments.hdae.hdae.config_io import load_hdae_config
    from experiments.hdae.hdae.lit_module import HDAELitModule

    def build(freqs):
        raw = _raw("shapes3d_a_k1_d256")
        raw["encoder"]["fourier_freqs"] = freqs
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(raw, f)
            path = f.name
        c = load_hdae_config(path, require_data=False)
        os.unlink(path)
        m = HDAELitModule(c.train_conf).model
        return {k: tuple(v.shape) for k, v in m.state_dict().items()}

    a, b = build(0), build(16)
    assert a == b, "fourier_freqs changed the model despite every attribute being categorical"
