import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import torch
from src.priors import LevelPrior,HierarchicalPriorStack

def test_prior_loss_sampling_and_inversion():
 p=LevelPrior(4,3,hidden_dim=16,num_layers=2,num_timesteps=20);z=torch.randn(5,4);c=torch.randn(5,3);assert torch.isfinite(p(z,c));assert p.sample(5,c,'cpu',5).shape==(5,4);assert p.invert(z,c,5).shape==(5,4)
def test_stack():
 p=HierarchicalPriorStack([6,4,2],{'hidden_dim':16,'num_layers':1,'num_timesteps':10});zs=[torch.randn(3,d) for d in [6,4,2]];loss,levels=p.compute_loss(zs);assert torch.isfinite(loss) and len(levels)==3;assert [x.shape for x in p.sample_full(3,'cpu',3)]==[(3,6),(3,4),(3,2)]
