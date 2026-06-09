import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest
import torch

from src.visualization import prepare_grid_columns


def test_prepare_grid_columns_interleaves_each_input_row():
    originals = torch.tensor([[[[1.0]]], [[[2.0]]]])
    reconstructions = torch.tensor([[[[10.0]]], [[[20.0]]]])
    grid = prepare_grid_columns([originals, reconstructions])
    assert grid.flatten().tolist() == [1.0, 10.0, 2.0, 20.0]


def test_prepare_grid_columns_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        prepare_grid_columns([torch.zeros(1, 3, 4, 4), torch.zeros(2, 3, 4, 4)])
