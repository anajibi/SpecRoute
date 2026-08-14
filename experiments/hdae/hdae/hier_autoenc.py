"""Conditional hierarchical DiffAE with per-decoder-block latent injection."""
from diffae_upstream.model.nn import timestep_embedding
from diffae_upstream.model.unet import BeatGANsEncoderConfig
from diffae_upstream.model.unet_autoenc import AutoencReturn, BeatGANsAutoencModel

from .attr_conditioner import (AttributeEmbedding, ConcatAttributeEmbedding, MixedAttributeEmbedding,
                               PerBlockStyle, PerBlockStyleFiLM)
from .hier_encoder import HierarchicalSemanticEncoder

# Both accept a null_mask kwarg (mask-based CFG/attr dropout); only the legacy binary
# AttributeEmbedding doesn't.
_NULL_MASK_AWARE = (MixedAttributeEmbedding, ConcatAttributeEmbedding)


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
        fusion = hdae_conf.conditioning.attr_fusion
        self.hdae_conf = hdae_conf
        self.encoder = HierarchicalSemanticEncoder(enc_conf, e.hier_tap_block_ids, e.hier_level_dims)
        if e.cond_specs and fusion == "concat_film":
            self.attr_embedding = ConcatAttributeEmbedding(e.cond_specs, e.attr_embed_dim, e.attr_dropout_prob,
                                                            hdae_conf.conditioning.cfg_drop_prob)
            self.per_block_style = PerBlockStyleFiLM(e.hier_level_dims, e.hier_block_to_level,
                                                      e.attr_embed_dim, conf.embed_channels)
        elif e.cond_specs:
            self.attr_embedding = MixedAttributeEmbedding(e.cond_specs, e.attr_embed_dim, e.attr_dropout_prob,
                                                           hdae_conf.conditioning.cfg_drop_prob)
            self.per_block_style = PerBlockStyle(e.hier_level_dims, e.hier_block_to_level,
                                                 e.attr_embed_dim, conf.embed_channels)
        else:
            self.attr_embedding = AttributeEmbedding(e.n_attributes, e.attr_embed_dim, e.attr_dropout_prob,
                                                     hdae_conf.conditioning.cfg_drop_prob)
            self.per_block_style = PerBlockStyle(e.hier_level_dims, e.hier_block_to_level,
                                                 e.attr_embed_dim, conf.embed_channels)
        self.last_zs = None

        assert conf.resnet_two_cond, "per-block style needs resnet_two_cond=True or cond= is ignored"

    def encode(self, x):
        self.last_zs = self.encoder(x)
        return self.last_zs

    def make_cond(self, zs, y_idx, null_mask=None):
        return {"zs": zs, "y_idx": y_idx, "null_mask": null_mask}

    def _styles(self, zs, y_idx, null_mask=None):
        if isinstance(self.attr_embedding, _NULL_MASK_AWARE):
            attr_emb = self.attr_embedding(y_idx, null_mask=null_mask)
        else:
            attr_emb = self.attr_embedding(y_idx)
        return self.per_block_style(zs, attr_emb)

    def forward(self, x, t, cond, t_cond=None, **_):
        zs, y_idx, null_mask = cond["zs"], cond["y_idx"], cond.get("null_mask")
        t_cond = t if t_cond is None else t_cond
        t_emb = timestep_embedding(t, self.conf.model_channels)
        t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        emb = self.time_embed.forward(time_emb=t_emb, cond=None, time_cond_emb=t_cond_emb).time_emb
        styles = self._styles(zs, y_idx, null_mask)

        hs = [[] for _ in range(len(self.conf.channel_mult))]
        h = x
        k = 0
        for i, n_blocks in enumerate(self.input_num_blocks):
            for _ in range(n_blocks):
                h = self.input_blocks[k](h, emb=emb, cond=None)
                hs[i].append(h)
                k += 1
        h = self.middle_block(h, emb=emb, cond=None)

        k = 0
        for i, n_blocks in enumerate(self.output_num_blocks):
            for _ in range(n_blocks):
                lateral = hs[-i - 1].pop()
                h = self.output_blocks[k](h, emb=emb, cond=styles[k], lateral=lateral)
                k += 1
        return AutoencReturn(pred=self.out(h), cond=cond)
