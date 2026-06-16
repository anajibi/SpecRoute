import torch
from experiments.hdae.latent_probing.swap_null_grid import swapped_zs


def test_swapped_zs_replaces_only_requested_levels():
    src = [torch.full((1, 2), i, dtype=torch.float32) for i in range(3)]
    donor = [torch.full((1, 2), i + 10, dtype=torch.float32) for i in range(3)]
    out = swapped_zs(src, donor, [0, 2])
    assert torch.equal(out[0], donor[0])
    assert torch.equal(out[1], src[1])
    assert torch.equal(out[2], donor[2])
