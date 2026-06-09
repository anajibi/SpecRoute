import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import pytest,torch
from src.encoders import ChainEncoder

def test_requires_strict_decrease():
 with pytest.raises(AssertionError):ChainEncoder(8,[4,4])
def test_shapes_and_gradients():
 m=ChainEncoder(8,[6,4,2],hidden_dim=12,num_layers_per_encoder=2);out=m(torch.randn(3,8));assert [x.shape for x in out]==[(3,6),(3,4),(3,2)];sum(x.sum() for x in out).backward();assert all(p.grad is not None for p in m.parameters())
def test_later_levels_only_depend_on_previous_latents():
 m=ChainEncoder(8,[6,4,2],hidden_dim=12,num_layers_per_encoder=2,noise_std=0).eval();z1=torch.randn(2,6);z2=m.encoders[1](z1);z3=m.encoders[2](torch.cat([z1,z2],-1));assert torch.equal(z2,m.encoders[1](z1));assert torch.equal(z3,m.encoders[2](torch.cat([z1,z2],-1)))
