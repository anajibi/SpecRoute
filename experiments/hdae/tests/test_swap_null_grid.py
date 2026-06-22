import pytest

torch = pytest.importorskip("torch")
from experiments.hdae.latent_probing.swap_null_grid import (
    dynamic_null_rows,
    dynamic_swap_rows,
    swapped_zs,
)


def test_swapped_zs_replaces_only_requested_levels():
    src = [torch.full((1, 2), i, dtype=torch.float32) for i in range(3)]
    donor = [torch.full((1, 2), i + 10, dtype=torch.float32) for i in range(3)]
    out = swapped_zs(src, donor, [0, 2])
    assert torch.equal(out[0], donor[0])
    assert torch.equal(out[1], src[1])
    assert torch.equal(out[2], donor[2])


def test_swap_and_null_rows_follow_number_of_levels():
    swaps = dynamic_swap_rows(5)
    nulls = dynamic_null_rows(5)
    assert ("swap_Z5", [4]) in swaps
    assert ("swap_Z4_Z5", [3, 4]) in swaps
    assert ("swap_Z5_Z6", [4, 5]) not in swaps
    assert nulls == [(f"null_Z{i}", [i - 1]) for i in range(1, 6)]
