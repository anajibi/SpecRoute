import pytest, torch
from model.unet import BeatGANsEncoderConfig
from experiments.hdae.hdae.hier_encoder import HierarchicalSemanticEncoder

def conf(): return BeatGANsEncoderConfig(32,3,8,16,16,1,(),channel_mult=(1,2,4),use_time_condition=False)
def test_outputs_gradients_and_flat():
    m=HierarchicalSemanticEncoder(conf(),[8,16],[5,3]); x=torch.randn(2,3,32,32,requires_grad=True);zs=m(x)
    assert [z.shape for z in zs]==[(2,5),(2,3)];sum(z.mean() for z in zs).backward();assert x.grad is not None
    assert any(p.grad is not None for p in m.heads.parameters()) and any(p.grad is not None for p in m.backbone.parameters())
    assert HierarchicalSemanticEncoder(conf(),[8],[7])(x.detach())[0].shape==(2,7)
def test_invalid_tap():
    with pytest.raises(ValueError): HierarchicalSemanticEncoder(conf(),[7],[4])
