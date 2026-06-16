"""Hierarchical semantic encoder integrated with the unchanged DiffAE decoder."""
from diffae_upstream.model.unet import BeatGANsEncoderConfig
from diffae_upstream.model.unet_autoenc import BeatGANsAutoencModel
from .conditioning import build_merger
from .hier_encoder import HierarchicalSemanticEncoder


class HierarchicalAutoencModel(BeatGANsAutoencModel):
    def __init__(self, conf, hdae_conf):
        super().__init__(conf)
        enc_conf = BeatGANsEncoderConfig(
            image_size=conf.image_size, in_channels=conf.in_channels,
            model_channels=conf.model_channels, out_hid_channels=conf.enc_out_channels,
            out_channels=conf.enc_out_channels, num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=conf.enc_attn_resolutions or conf.attention_resolutions,
            dropout=conf.dropout, channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False, conv_resample=conf.conv_resample, dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads, num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order, pool=conf.enc_pool)
        e = hdae_conf.encoder
        self.encoder = HierarchicalSemanticEncoder(enc_conf, e.tap_resolutions, e.level_dims,
                                                   e.pool, e.proj)
        c = hdae_conf.conditioning
        self.merge = build_merger(c.strategy, e.level_dims, c.style_ch, c.latent_drop_prob)
        self.hdae_conf = hdae_conf
        self.last_zs = None

    def encode(self, x, null_levels=None):
        """Encode an image and optionally force selected latent levels to learned null tokens."""
        zs = self.encoder(x)
        self.last_zs = zs
        cond, null_mask = self.merge(zs, null_levels=null_levels, return_mask=True)
        self.last_null_mask = null_mask
        return {"cond": cond, "zs": zs, "null_mask": null_mask}

    def encode_with_nulls(self, x, null_levels):
        """Convenience helper for test-time latent-level ablations."""
        return self.encode(x, null_levels=null_levels)
