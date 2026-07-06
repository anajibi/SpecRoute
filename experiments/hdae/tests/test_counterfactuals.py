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
    before = np.array([[0.2, 0.1, 0.6, 0.1], [0.2, 0.2, 0.4, 0.8]])
    after = np.array([[0.2, 0.7, 0.4, 0.1], [0.2, 0.8, 0.7, 0.3]])
    summary = summarize_attribute_changes(before, after, target_index=1, severe_threshold=0.25)
    assert summary["target_delta_mean"] == pytest.approx(0.6)
    assert summary["target_flip_rate"] == pytest.approx(1.0)
    assert summary["non_target_abs_delta_mean"] == pytest.approx((0 + .2 + 0 + 0 + .3 + .5) / 6)
    assert summary["non_target_severe_fraction"] == pytest.approx(2 / 6)
    assert summary["non_target_flip_fraction"] == pytest.approx(3 / 6)
    assert summary["non_target_any_flip_rate"] == pytest.approx(1.0)


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


def test_transfer_for_scores_uses_reconstruction_baseline_and_masks_small_denominators():
    from experiments.hdae.counterfactuals.run_swap_eval import transfer_for_scores

    source = np.array([[0.2, 0.5], [0.4, 0.1]])
    donor = np.array([[0.6, 0.5], [0.8, 0.5]])
    recon = np.array([[0.3, 0.4], [0.5, 0.2]])
    swaps = np.array([
        [[0.5, 0.9], [0.7, 0.4]],
        [[0.4, 0.9], [0.6, 0.6]],
    ])
    matrix, valid_counts, ratios = transfer_for_scores(source, donor, recon, swaps, eps=1e-6)
    assert matrix[0, 0] == pytest.approx(0.5)
    assert matrix[0, 1] == pytest.approx(0.5)
    assert valid_counts[:, 0].tolist() == [2, 2]
    assert valid_counts[:, 1].tolist() == [1, 1]
    assert np.isnan(ratios[0, 0, 1])


def test_parse_strengths_injects_zero_and_preserves_values():
    from experiments.hdae.counterfactuals.run_preservation_sweep import parse_strengths

    assert parse_strengths("0.5,1,2") == [0.0, 0.5, 1.0, 2.0]
    assert parse_strengths("0,1") == [0.0, 1.0]


def test_intended_target_flip_rate_is_directional():
    from experiments.hdae.counterfactuals.run_preservation_sweep import intended_target_flip_rate

    before = np.array([[0.4, 0.2], [0.6, 0.2], [0.2, 0.2]])
    after = np.array([[0.7, 0.1], [0.8, 0.1], [0.3, 0.1]])
    assert intended_target_flip_rate(before, after, 0, "positive") == pytest.approx(1 / 3)
    assert intended_target_flip_rate(after, before, 0, "negative") == pytest.approx(1 / 3)
