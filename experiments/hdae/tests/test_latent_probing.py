import numpy as np

from experiments.hdae.latent_probing.linear_probe import (
    binary_metrics,
    latent_keys,
    make_probe_jobs,
    split_indices,
)


def test_probe_job_count_for_three_levels_and_forty_attributes():
    names = [f"attr_{i}" for i in range(40)]
    jobs = make_probe_jobs(names, 3)
    assert len(jobs) == 120
    assert jobs[0].level == 0 and jobs[0].attribute_index == 0
    assert jobs[-1].level == 2 and jobs[-1].attribute_index == 39


def test_latent_keys_sort_numerically_and_metrics_are_balanced():
    arrays = {"z_level_10": np.zeros((2, 1)), "z_level_2": np.zeros((2, 1)), "attrs": np.zeros((2, 40))}
    assert latent_keys(arrays) == ["z_level_2", "z_level_10"]
    metrics = binary_metrics(np.array([-2.0, 1.0, -1.0, 3.0]), np.array([0, 1, 1, 1]))
    assert metrics["accuracy"] == 0.75
    assert metrics["positive_accuracy"] == 2 / 3
    assert metrics["negative_accuracy"] == 1.0


def test_split_indices_uses_partitions_or_fallback():
    train, val, test = split_indices(np.array([0, 1, 2, 0, 1, 2]))
    assert train.tolist() == [0, 3]
    assert val.tolist() == [1, 4]
    assert test.tolist() == [2, 5]
    train, val, test = split_indices(np.zeros(20, dtype=int))
    assert len(train) == 16 and len(val) == 2 and len(test) == 2
