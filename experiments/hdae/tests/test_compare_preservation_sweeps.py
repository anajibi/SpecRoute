import csv

import pytest

np = pytest.importorskip("numpy")

from experiments.hdae.counterfactuals.compare_preservation_sweeps import (
    aggregate,
    filter_rows,
    parse_named_paths,
    read_sweep,
)


def _write_sweep(path):
    rows = [
        {"attribute": "Smiling", "level": 0, "level_dim": 256, "strength": 0.0, "direction": "positive",
         "non_target_abs_delta_mean": 0.0, "non_target_flip_fraction": 0.0, "target_intended_flip_rate": 0.0},
        {"attribute": "Smiling", "level": 0, "level_dim": 256, "strength": 1.0, "direction": "positive",
         "non_target_abs_delta_mean": 0.2, "non_target_flip_fraction": 0.1, "target_intended_flip_rate": 0.5},
        {"attribute": "Young", "level": 1, "level_dim": 128, "strength": 1.0, "direction": "negative",
         "non_target_abs_delta_mean": 0.4, "non_target_flip_fraction": 0.2, "target_intended_flip_rate": 0.25},
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def test_parse_named_paths_requires_name_equals_path(tmp_path):
    parsed = parse_named_paths([f"k3={tmp_path / 'sweep.csv'}"])
    assert parsed["k3"] == tmp_path / "sweep.csv"
    with pytest.raises(ValueError):
        parse_named_paths(["missing_equals"])


def test_read_filter_and_aggregate_sweep_rows(tmp_path):
    path = tmp_path / "preservation_sweep.csv"
    _write_sweep(path)
    rows = read_sweep("k3", path)
    assert rows[0]["config"] == "k3"
    assert rows[0]["resolution"] == 0
    subset = filter_rows(rows, attributes=["Smiling"], directions=["positive"], strengths=[1.0])
    assert len(subset) == 1
    grouped = aggregate(rows, ["config", "strength"], ["non_target_abs_delta_mean"])
    by_strength = {row["strength"]: row["non_target_abs_delta_mean"] for row in grouped}
    assert by_strength[0.0] == pytest.approx(0.0)
    assert by_strength[1.0] == pytest.approx(0.3)
