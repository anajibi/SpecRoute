import torch
from model.unet_autoenc import BeatGANsAutoencConfig
from experiments.hdae.hdae.hier_autoenc import HierarchicalAutoencModel
from experiments.hdae.hdae.hier_config import HDAEConfig,EncoderHierarchyConfig,ConditioningConfig

def test_full_shapes():
    c=BeatGANsAutoencConfig(image_size=32,in_channels=3,model_channels=8,out_channels=3,num_res_blocks=1,
      attention_resolutions=(),channel_mult=(1,2),embed_channels=32,resnet_two_cond=True,enc_out_channels=16,
      enc_num_res_block=1,enc_channel_mult=(1,2,4),enc_attn_resolutions=(),enc_pool='adaptivenonzero')
    h=HDAEConfig(EncoderHierarchyConfig(tap_resolutions=[8,16],level_dims=[10,6]),ConditioningConfig(style_ch=16))
    m=HierarchicalAutoencModel(c,h);x=torch.randn(2,3,32,32);e=m.encode(x);out=m(x,torch.zeros(2,dtype=torch.long),x_start=x)
    assert e['cond'].shape==(2,16) and len(e['zs'])==2 and out.pred.shape==x.shape
