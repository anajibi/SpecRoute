import torch
from dino_vae_hierarchical_diffusion.src.evidence import EvidencePyramid
from dino_vae_hierarchical_diffusion.src.encoders import HierarchicalEncoderK5
def test_k5_shapes():
 b=1;e=EvidencePyramid()(torch.randn(b,4,32,32),torch.randn(b,768),torch.randn(b,768,16,16));o=HierarchicalEncoderK5()(e)
 for n,s in {'Z5':(b,512),'Z4':(b,256),'Z3':(b,128,4,4),'Z2':(b,64,8,8),'Z1':(b,64,16,16)}.items():assert o[n].shape==s
