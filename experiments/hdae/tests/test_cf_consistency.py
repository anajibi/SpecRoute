import csv
import json

import pytest

np = pytest.importorskip("numpy")

from experiments.hdae.build_cohorts import sample_attr_indices
from experiments.hdae.cf_aggregate import aggregate_rows
from experiments.hdae.run_cf_consistency import compute_consistency, parse_models, source_indices


def test_sample_attr_indices_balances_pos_and_neg():
    attrs = np.array([[1], [-1], [1], [-1], [1], [-1]])
    out = sample_attr_indices(attrs, ["Smiling"], "Smiling", 2, seed=0)
    assert len(out["pos_idx"]) == 2
    assert len(out["neg_idx"]) == 2
    assert all(attrs[i, 0] > 0 for i in out["pos_idx"])
    assert all(attrs[i, 0] <= 0 for i in out["neg_idx"])


def test_compute_consistency_restricts_to_recon_source_side():
    # target is column 0. Third image is dropped because recon0 already crossed positive.
    base = np.array([[0.2, 0.2, 0.8], [0.3, 0.7, 0.8], [0.6, 0.2, 0.2]])
    edit = np.array([[0.7, 0.8, 0.8], [0.4, 0.2, 0.1], [0.9, 0.8, 0.8]])
    out = compute_consistency(base, edit, target_idx=0, direction="positive")
    assert out["n_source"] == 2
    assert out["n_success"] == 1
    assert out["n_fail"] == 1
    assert out["counterfactual_consistency"] == pytest.approx(0.5)
    assert out["factual_flip_success"] == pytest.approx(0.5)
    assert out["factual_flip_fail"] == pytest.approx(1.0)


def test_source_indices_and_model_parser():
    cohorts = {"attributes": {"Male": {"pos_idx": [1, 2], "neg_idx": [3, 4]}}}
    assert source_indices(cohorts, "Male", "positive") == [3, 4]
    assert source_indices(cohorts, "Male", "negative") == [1, 2]
    parsed = parse_models(["k3=a.yaml,b.ckpt,c.csv,weights"])
    assert parsed["k3"]["ckpt"].name == "b.ckpt"
    with pytest.raises(ValueError):
        parse_models(["bad=a,b"])


def test_cf_aggregate_groups_duplicate_runs():
    rows = [
        {"model": "k3", "attribute": "Smiling", "latent_used": "0", "direction": "positive",
         "counterfactual_consistency": "0.5", "factual_flip_success": "0.1", "factual_flip_fail": "0.2",
         "n_source": "10", "n_success": "5", "n_fail": "5"},
        {"model": "k3", "attribute": "Smiling", "latent_used": "0", "direction": "positive",
         "counterfactual_consistency": "0.7", "factual_flip_success": "0.3", "factual_flip_fail": "0.4",
         "n_source": "10", "n_success": "7", "n_fail": "3"},
    ]
    out = aggregate_rows(rows)
    assert len(out) == 1
    assert out[0]["counterfactual_consistency"] == pytest.approx(0.6)
    assert out[0]["num_runs"] == 2
