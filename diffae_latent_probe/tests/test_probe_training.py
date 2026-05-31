from pathlib import Path

import numpy as np
import pandas as pd

from diffae_tools.probe_models import ProbeSuiteConfig, train_probe_suite


def test_probe_training_runs_on_synthetic_latents(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 60
    image_ids = [f"{i:06d}.png" for i in range(n)]
    semantic = rng.normal(size=(n, 8)).astype(np.float32)
    stochastic = rng.normal(size=(n, 3, 8, 8)).astype(np.float32)

    smiling = (semantic[:, 0] + 0.25 * rng.normal(size=n) > 0).astype(int)
    brightness = stochastic.mean(axis=(1, 2, 3)) * 3.0 + rng.normal(scale=0.1, size=n)

    labels = pd.DataFrame(
        {
            "image_id": image_ids,
            "smiling": smiling,
            "brightness": brightness,
        }
    )

    results = train_probe_suite(
        semantic_latents=semantic,
        stochastic_latents=stochastic,
        labels_frame=labels,
        image_id_column="image_id",
        output_dir=tmp_path,
        cfg=ProbeSuiteConfig(seed=0),
        label_columns=["smiling", "brightness"],
    )

    assert not results.empty
    assert (tmp_path / "probe_results.csv").exists()
    assert (tmp_path / "probe_coefficients").exists()
    assert (tmp_path / "predictions").exists()
    assert set(results["latent_source"]).issuperset({"semantic", "stochastic", "concat", "random_baseline"})

