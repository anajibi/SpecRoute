from experiments.hdae.latent_probing.abduct_xt_z_grid import (
    complement_levels,
    forward_cumulative_rows,
    reverse_cumulative_rows,
)


def test_abduct_grid_level_schedules_for_five_levels():
    assert complement_levels(5, [0, 2, 4]) == [1, 3]
    assert forward_cumulative_rows(5)[0] == ("forward_Z0", [0])
    assert forward_cumulative_rows(5)[-1] == ("forward_Z0_Z1_Z2_Z3_Z4", [0, 1, 2, 3, 4])
    assert reverse_cumulative_rows(5)[0] == ("reverse_Z-1", [4])
    assert reverse_cumulative_rows(5)[-1] == ("reverse_Z-1_Z-2_Z-3_Z-4_Z-5", [0, 1, 2, 3, 4])
