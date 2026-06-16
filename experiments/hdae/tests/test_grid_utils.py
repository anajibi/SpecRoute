import torch
from PIL import Image
from experiments.hdae.hdae.grid_utils import save_labeled_grid


def test_save_labeled_grid_adds_left_label_area(tmp_path):
    rows = [torch.zeros(2, 3, 4, 5), torch.ones(2, 3, 4, 5)]
    out = tmp_path / "grid.png"
    save_labeled_grid(rows, ["zero", "one"], out, label_width=20, pad=1)
    image = Image.open(out)
    assert image.size == (20 + 2 * 5 + 1, 2 * 4 + 1)
