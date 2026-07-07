import pytest
import torch

from experiments.hdae.hdae.attr_conditioner import AttributeEmbedding, PerBlockStyle
from experiments.hdae.hdae.attr_utils import to_index_space


def test_to_index_space_detects_pm1_and_01():
    assert torch.equal(to_index_space(torch.tensor([[-1, 1]])), torch.tensor([[0, 1]]))
    assert torch.equal(to_index_space(torch.tensor([[0, 1]])), torch.tensor([[0, 1]]))
    with pytest.raises(ValueError):
        to_index_space(torch.tensor([[0.2, 1.0]]))


def test_attribute_embedding_and_per_block_style_shapes():
    emb = AttributeEmbedding(n_attributes=2, attr_embed_dim=4, attr_dropout_prob=0.0)
    y = torch.tensor([[0, 1], [2, 2]])
    attr = emb(y, apply_dropout=False)
    assert attr.shape == (2, 4)
    styles = PerBlockStyle([3, 5], [0, 1, 1], attr_embed_dim=4, embed_channels=8)(
        [torch.randn(2, 3), torch.randn(2, 5)], attr)
    assert [s.shape for s in styles] == [(2, 8), (2, 8), (2, 8)]
