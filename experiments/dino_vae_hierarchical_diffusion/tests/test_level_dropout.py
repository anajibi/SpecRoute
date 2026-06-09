import torch
from dino_vae_hierarchical_diffusion.src.decoders import apply_level_dropout,DeterministicLatentDecoder
def test_all_dropout_modes():
 z={'Z3':torch.randn(2,512),'Z2':torch.randn(2,128,8,8),'Z1':torch.randn(2,64,16,16)}
 for mode in ['all','drop_fine','drop_two','random','coarse_only']:
  q=apply_level_dropout(z,mode=mode);assert all(q[n].shape==z[n].shape for n in z);assert DeterministicLatentDecoder()(q['Z3'],q['Z2'],q['Z1']).shape==(2,4,32,32)
