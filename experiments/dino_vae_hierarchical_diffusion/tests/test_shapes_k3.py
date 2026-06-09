import torch
from dino_vae_hierarchical_diffusion.src.evidence import EvidencePyramid
from dino_vae_hierarchical_diffusion.src.encoders import HierarchicalEncoderK3
from dino_vae_hierarchical_diffusion.src.decoders import DeterministicLatentDecoder,LatentDecoderDiffusion32x32
def test_k3_shapes():
 b=2;e=EvidencePyramid()(torch.randn(b,4,32,32),torch.randn(b,768),torch.randn(b,768,16,16));o=HierarchicalEncoderK3()(e);assert o['Z3'].shape==(b,512);assert o['Z2'].shape==(b,128,8,8);assert o['Z1'].shape==(b,64,16,16);levels=(o['Z3'],o['Z2'],o['Z1']);assert DeterministicLatentDecoder()(*levels).shape==(b,4,32,32);assert LatentDecoderDiffusion32x32()(torch.randn(b,4,32,32),torch.ones(b,dtype=torch.long),*levels).shape==(b,4,32,32)
