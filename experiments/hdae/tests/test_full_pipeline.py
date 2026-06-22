from pathlib import Path
from experiments.hdae.scripts.run_full_pipeline import outputs_exist


def test_outputs_exist_requires_all_paths(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("done")
    assert not outputs_exist([a, b])
    b.write_text("done")
    assert outputs_exist([a, b])
