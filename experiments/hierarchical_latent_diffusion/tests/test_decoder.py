import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import pytest,torch
from src.decoder import HierarchicalDecoder
@pytest.mark.parametrize('arch',['mlp','tokens'])
def test_shape_and_all_levels_used(arch):
 torch.manual_seed(1);m=HierarchicalDecoder([8,4,2],(2,4,4),arch,hidden_dim=16,num_transformer_layers=1,num_heads=4).eval();zs=[torch.randn(2,d) for d in [8,4,2]];base=m(zs);assert base.shape==(2,2,4,4)
 for i in range(3):
  changed=list(zs);changed[i]=torch.zeros_like(changed[i]);assert not torch.allclose(base,m(changed))
