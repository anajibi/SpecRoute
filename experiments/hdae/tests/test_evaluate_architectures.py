import pytest

np = pytest.importorskip("numpy")

from experiments.hdae.scripts.evaluate_architectures import (
    effective_num_levels,
    parse_named_paths,
    preservation_at_efficacy,
    pick_best_level_per_attribute,
    summarize_model_editing,
)


def test_effective_num_levels_identifies_collapse_and_spread():
    collapsed = effective_num_levels([1.0, 0.0, 0.0, 0.0, 0.0])
    spread = effective_num_levels([1.0, 1.0, 1.0, 1.0, 1.0])
    two = effective_num_levels([1.0, 1.0, 0.0, 0.0])
    assert collapsed["n_eff"] == pytest.approx(1.0)
    assert collapsed["n_eff_norm"] == pytest.approx(0.2)
    assert spread["n_eff"] == pytest.approx(5.0)
    assert spread["n_eff_norm"] == pytest.approx(1.0)
    assert two["n_eff"] == pytest.approx(2.0)


def test_preservation_at_efficacy_uses_first_threshold_hit():
    rows = [
        {"strength": 0.5, "target_intended_flip_rate": 0.4, "non_target_abs_delta_mean": 0.02},
        {"strength": 1.0, "target_intended_flip_rate": 0.8, "target_delta_abs_mean": 0.3,
         "non_target_abs_delta_mean": 0.05, "non_target_flip_fraction": 0.1, "non_target_severe_fraction": 0.0},
        {"strength": 2.0, "target_intended_flip_rate": 0.9, "non_target_abs_delta_mean": 0.12},
    ]
    out = preservation_at_efficacy(rows, threshold=0.8)
    assert out["reached_threshold"] is True
    assert out["selected_strength"] == pytest.approx(1.0)
    assert out["preservation_at_efficacy"] == pytest.approx(0.05)


def test_preservation_at_efficacy_reports_no_threshold():
    rows = [
        {"strength": 0.5, "target_intended_flip_rate": 0.2, "non_target_abs_delta_mean": 0.05},
        {"strength": 1.0, "target_intended_flip_rate": 0.6, "non_target_abs_delta_mean": 0.10},
    ]
    out = preservation_at_efficacy(rows, threshold=0.8)
    assert out["reached_threshold"] is False
    assert out["max_target_flip_rate"] == pytest.approx(0.6)
    assert out["best_available_non_target_abs_delta_mean"] == pytest.approx(0.05)


def test_best_level_selection_and_model_summary():
    rows = [
        {"model": "k5", "attribute": "Smiling", "direction": "positive", "level": 0,
         "resolution": 4, "dim": 192, "reached_threshold": True, "preservation_at_efficacy": 0.10},
        {"model": "k5", "attribute": "Smiling", "direction": "positive", "level": 1,
         "resolution": 8, "dim": 128, "reached_threshold": True, "preservation_at_efficacy": 0.04},
        {"model": "k5", "attribute": "Young", "direction": "negative", "level": 0,
         "resolution": 4, "dim": 192, "reached_threshold": False, "max_target_flip_rate": 0.3},
        {"model": "k5", "attribute": "Young", "direction": "negative", "level": 1,
         "resolution": 8, "dim": 128, "reached_threshold": False, "max_target_flip_rate": 0.7},
    ]
    best = pick_best_level_per_attribute(rows)
    by_key = {(r["attribute"], r["direction"]): r for r in best}
    assert by_key[("Smiling", "positive")]["best_level"] == 1
    assert by_key[("Young", "negative")]["best_level"] == 1
    summary = summarize_model_editing(best, efficacy_threshold=0.8)
    assert summary["coverage"] == pytest.approx(0.5)
    assert summary["mean_preservation_at_efficacy"] == pytest.approx(0.04)


def test_parse_named_paths_validates_entries(tmp_path):
    parsed = parse_named_paths([f"k3={tmp_path / 'config.yaml'}"])
    assert parsed["k3"] == tmp_path / "config.yaml"
    with pytest.raises(ValueError):
        parse_named_paths(["no_equals"])
