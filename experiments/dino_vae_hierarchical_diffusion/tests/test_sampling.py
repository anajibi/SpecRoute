import torch
from dino_vae_hierarchical_diffusion.src.priors import VectorDiffusionPrior,SpatialDiffusionPrior
from dino_vae_hierarchical_diffusion.src.decoders import LatentDecoderDiffusion32x32
def test_k3_sampling():
 b=1;z3=VectorDiffusionPrior(512).sample(batch_size=b,steps=2);z2=SpatialDiffusionPrior(128,8,0,512).sample(None,z3,steps=2);z1=SpatialDiffusionPrior(64,16,128,512).sample(z2,z3,steps=2);z0=LatentDecoderDiffusion32x32().sample(z3,z2,z1,steps=2);assert z3.shape==(b,512);assert z2.shape==(b,128,8,8);assert z1.shape==(b,64,16,16);assert z0.shape==(b,4,32,32)
