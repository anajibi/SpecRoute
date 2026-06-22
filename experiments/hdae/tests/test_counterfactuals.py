import csv
import numpy as np
import pytest

from experiments.hdae.counterfactuals.directions import choose_probe_row, summarize_attribute_changes


def test_choose_probe_row_best_or_explicit_level(tmp_path):
    path = tmp_path / "probe_metrics.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "attribute_name", "val_balanced_accuracy"])
        writer.writeheader()
        writer.writerow({"level": "0", "attribute_name": "Smiling", "val_balanced_accuracy": "0.60"})
        writer.writerow({"level": "2", "attribute_name": "Smiling", "val_balanced_accuracy": "0.85"})
        writer.writerow({"level": "1", "attribute_name": "Male", "val_balanced_accuracy": "0.90"})
    assert choose_probe_row(path, "Smiling", "best")["level"] == "2"
    assert choose_probe_row(path, "Smiling", 0)["level"] == "0"
    with pytest.raises(ValueError):
        choose_probe_row(path, "Young", "best")


def test_summarize_attribute_changes_reports_target_and_preservation():
    before = np.zeros((2, 4), dtype=float)
    after = np.array([[0.0, 0.7, 0.1, 0.0], [0.0, 0.5, -0.2, 0.3]])
    summary = summarize_attribute_changes(before, after, target_index=1, severe_threshold=0.25)
    assert summary["target_delta_mean"] == pytest.approx(0.6)
    assert summary["non_target_abs_delta_mean"] == pytest.approx((0 + .1 + 0 + 0 + .2 + .3) / 6)
    assert summary["non_target_severe_fraction"] == pytest.approx(1 / 6)


def test_torch_load_probe_checkpoint_requests_weights_only_false(monkeypatch, tmp_path):
    from experiments.hdae.counterfactuals import directions

    calls = {}

    def fake_load(path, **kwargs):
        calls.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(directions.inspect, "signature", lambda _fn: type("Sig", (), {"parameters": {"weights_only": object()}})())
    monkeypatch.setattr(directions, "torch_load_probe_checkpoint", directions.torch_load_probe_checkpoint)
    import types
    fake_torch = types.SimpleNamespace(load=fake_load)
    monkeypatch.setitem(__import__('sys').modules, 'torch', fake_torch)
    out = directions.torch_load_probe_checkpoint(tmp_path / "probe.pt")
    assert out == {"ok": True}
    assert calls["weights_only"] is False
