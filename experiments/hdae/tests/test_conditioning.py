import pytest, torch
from experiments.hdae.hdae.conditioning import ConcatProjectionMerger

def test_concat_shape_and_each_level_matters():
    merger=ConcatProjectionMerger([4,3],7,latent_drop_prob=0.0); zs=[torch.randn(2,4),torch.randn(2,3)]; base=merger(zs)
    assert base.shape == (2,7)
    for i in range(2):
        changed=[z.clone() for z in zs]; changed[i].add_(1); assert not torch.equal(base,merger(changed))

def test_budget_validation():
    with pytest.raises(ValueError): ConcatProjectionMerger([2,3],6)


def test_learned_null_tokens_force_specific_levels_and_get_gradients():
    merger=ConcatProjectionMerger([4,3],7,latent_drop_prob=0.0)
    zs=[torch.randn(2,4,requires_grad=True),torch.randn(2,3,requires_grad=True)]
    cond, mask = merger(zs, null_levels=[1], return_mask=True)
    assert mask.tolist() == [[False, True], [False, True]]
    cond.sum().backward()
    assert merger.null_tokens[1].grad is not None
    assert merger.null_tokens[0].grad is None or torch.all(merger.null_tokens[0].grad == 0)


def test_training_dropout_uses_per_level_probability():
    merger=ConcatProjectionMerger([2,2],4,latent_drop_prob=1.0 - 1e-6)
    merger.train()
    zs=[torch.randn(8,2),torch.randn(8,2)]
    _, mask = merger(zs, return_mask=True)
    assert mask.shape == (8,2)
    assert mask.any()
