"""Tier 2 -- provenance, and the checkpoint load.

The key-match test is marked slow because it pulls ~20 GB from S3. It is the test that would have
caught a checkpoint being paired with the wrong config, which does not raise: PyTorch reports
missing and unexpected keys and carries on with a partly random model.
"""
import os
import subprocess
import pytest
from able import registry as R

MORPHO_IS_OUT_OF_SCOPE = "those models are being retrained"


def test_every_model_has_distinct_identity():
    assert len(R.MODELS) == 8
    assert len({m.s3_ckpt for m in R.MODELS.values()}) == 8, "two entries share a checkpoint"
    assert len({m.config for m in R.MODELS.values()}) == 8, "two entries share a config"


def test_morphomnist_is_absent():
    assert not [m for m in R.MODELS.values() if "morpho" in m.dataset], MORPHO_IS_OUT_OF_SCOPE


def test_both_celebahq_graphs_are_represented():
    graphs = {m.graph for m in R.by_dataset("celebahq")}
    assert graphs == {"A", "B"}
    assert len([m for m in R.by_dataset("celebahq") if m.graph == "A"]) == 3
    assert len([m for m in R.by_dataset("celebahq") if m.graph == "B"]) == 2
    assert len(R.by_dataset("c3di")) == 3


def test_graph_b_conditions_on_two_attributes():
    for m in R.by_dataset("celebahq"):
        assert len(m.attrs) == (2 if m.graph == "B" else 4)


def test_config_paths_exist():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for m in R.MODELS.values():
        assert os.path.exists(os.path.join(root, m.config)), f"{m.name}: {m.config} missing"


def test_checkpoint_uris_are_under_the_able_prefix():
    """The prefix was renamed from hdae-handoff/; a stale key here would 404 at load time."""
    for m in R.MODELS.values():
        assert m.s3_ckpt.startswith("able/"), f"{m.name}: {m.s3_ckpt} is not under able/"
        assert "hdae-handoff" not in m.s3_ckpt and "wip-from" not in m.s3_ckpt


def test_unknown_model_raises_with_a_useful_message():
    with pytest.raises(KeyError, match="known:"):
        R.get("nope")


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(R.MODELS))
def test_checkpoint_exists_in_s3(name):
    m = R.get(name)
    r = subprocess.run(["aws", "s3api", "head-object", "--bucket", R.BUCKET,
                        "--key", m.s3_ckpt, "--query", "ContentLength", "--output", "text"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"{name}: {m.uri} not found\n{r.stderr.strip()}"
    assert int(r.stdout.strip()) > 1_000_000_000, f"{name}: too small to be a checkpoint"


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(R.MODELS))
def test_checkpoint_keys_match_exactly(name, tmp_path):
    """Build from the registered config, then demand an exact state_dict key match."""
    import torch
    m = R.get(name)
    local = tmp_path / "ckpt.pt"
    r = subprocess.run(["aws", "s3", "cp", m.uri, str(local), "--only-show-errors"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"download failed: {r.stderr.strip()}"
    ck = torch.load(local, map_location="cpu")
    have = {k[len("model."):] for k in ck["state_dict"] if k.startswith("model.")}
    lit, _ = R.build(m)
    want = set(lit.model.state_dict())
    assert want == have, (
        f"{name}: {len(want - have)} missing, {len(have - want)} unexpected. "
        "A load would half-initialise the model without raising.")
    ema = {k: v for k, v in ck["state_dict"].items() if k.startswith("ema_model.")}
    bad = [k for k, v in ema.items()
           if torch.is_floating_point(v) and not torch.isfinite(v).all()]
    assert not bad, f"{name}: {len(bad)} non-finite EMA tensors, e.g. {bad[:3]}"
    local.unlink()
