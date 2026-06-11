import pytest, torch
from experiments.hdae.hdae.conditioning import ConcatProjectionMerger

def test_concat_shape_and_each_level_matters():
    merger=ConcatProjectionMerger([4,3],7); zs=[torch.randn(2,4),torch.randn(2,3)]; base=merger(zs)
    assert base.shape == (2,7)
    for i in range(2):
        changed=[z.clone() for z in zs]; changed[i].add_(1); assert not torch.equal(base,merger(changed))

def test_budget_validation():
    with pytest.raises(ValueError): ConcatProjectionMerger([2,3],6)
