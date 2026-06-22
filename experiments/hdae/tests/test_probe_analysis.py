import csv
import numpy as np

from experiments.hdae.latent_probing.analyze_probe_results import best_level_summary, metric_matrix


def rows():
    return [
        {"level": "0", "latent_key": "z_level_0", "attribute_name": "Smiling", "test_balanced_accuracy": "0.65"},
        {"level": "1", "latent_key": "z_level_1", "attribute_name": "Smiling", "test_balanced_accuracy": "0.80"},
        {"level": "0", "latent_key": "z_level_0", "attribute_name": "Male", "test_balanced_accuracy": "0.75"},
        {"level": "1", "latent_key": "z_level_1", "attribute_name": "Male", "test_balanced_accuracy": "0.70"},
    ]


def test_metric_matrix_and_best_level_summary():
    levels, attrs, matrix = metric_matrix(rows())
    assert levels == [0, 1]
    assert attrs == ["Male", "Smiling"]
    assert matrix.shape == (2, 2)
    assert np.isclose(matrix[1, 1], 0.80)
    summary = {r["attribute_name"]: r for r in best_level_summary(rows())}
    assert summary["Smiling"]["best_level"] == 1
    assert summary["Male"]["best_level"] == 0
