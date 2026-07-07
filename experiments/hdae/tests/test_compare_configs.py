import csv

import pytest

np = pytest.importorskip("numpy")

from experiments.hdae.scripts.compare_configs import markdown_table, probe_summary


def test_probe_summary_reports_resolution_counts():
    rows = [
        {"level": "0", "attribute_name": "Smiling", "test_balanced_accuracy": "0.70"},
        {"level": "1", "attribute_name": "Smiling", "test_balanced_accuracy": "0.90"},
        {"level": "0", "attribute_name": "Young", "test_balanced_accuracy": "0.80"},
        {"level": "1", "attribute_name": "Young", "test_balanced_accuracy": "0.60"},
    ]
    summary, by_level, best = probe_summary(rows, [4, 8])
    assert summary["probe_mean_test_balanced_accuracy"] == pytest.approx(0.75)
    assert summary["best_level_by_resolution"] == {8: 1, 4: 1}
    assert {row["resolution"] for row in by_level} == {4, 8}
    assert {row["attribute_name"] for row in best} == {"Smiling", "Young"}


def test_markdown_table_formats_float_cells():
    md = markdown_table([{"config": "flat", "mse_mean": 0.123456}], ["config", "mse_mean"])
    assert "flat" in md
    assert "0.1235" in md
