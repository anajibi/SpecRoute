import torch
from test_backbones import V,D
from dino_vae_hierarchical_diffusion.src import FrozenSDVAE,FrozenDINOv2,EvidencePyramid,HierarchicalEncoderK3,DeterministicLatentDecoder
def test_gradient_flow():
 vae=FrozenSDVAE(model=V());dino=FrozenDINOv2(model=D());ev=EvidencePyramid();en=HierarchicalEncoderK3();dec=DeterministicLatentDecoder();x=torch.randn(1,3,256,256);z=vae.encode(x);c,m=dino(x);o=en(ev(z,c,m));dec(o['Z3'],o['Z2'],o['Z1']).mean().backward();assert all(p.grad is None for p in vae.parameters());assert all(p.grad is None for p in dino.parameters());assert any(p.grad is not None for p in ev.parameters());assert any(p.grad is not None for p in en.parameters());assert any(p.grad is not None for p in dec.parameters())
