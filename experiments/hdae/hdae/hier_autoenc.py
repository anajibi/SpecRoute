"""Conditional hierarchical DiffAE with per-decoder-block latent injection."""
from diffae_upstream.model.nn import timestep_embedding
from diffae_upstream.model.unet import BeatGANsEncoderConfig
from diffae_upstream.model.unet_autoenc import AutoencReturn, BeatGANsAutoencModel

from .attr_conditioner import AttributeEmbedding, PerBlockStyle
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
        self.hdae_conf = hdae_conf
        self.encoder = HierarchicalSemanticEncoder(enc_conf, e.hier_tap_block_ids, e.hier_level_dims)
        self.attr_embedding = AttributeEmbedding(e.n_attributes, e.attr_embed_dim, e.attr_dropout_prob)
        self.per_block_style = PerBlockStyle(e.hier_level_dims, e.hier_block_to_level,
                                             e.attr_embed_dim, conf.embed_channels)
        self.last_zs = None

        assert conf.resnet_two_cond, "per-block style needs resnet_two_cond=True or cond= is ignored"

    def encode(self, x):
        self.last_zs = self.encoder(x)
        return self.last_zs

    def make_cond(self, zs, y_idx):
        return {"zs": zs, "y_idx": y_idx}

    def _styles(self, zs, y_idx):
        return self.per_block_style(zs, self.attr_embedding(y_idx))

    def forward(self, x, t, cond, t_cond=None, **_):
        zs, y_idx = cond["zs"], cond["y_idx"]
        t_cond = t if t_cond is None else t_cond
        t_emb = timestep_embedding(t, self.conf.model_channels)
        t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        emb = self.time_embed.forward(time_emb=t_emb, cond=None, time_cond_emb=t_cond_emb).time_emb
        styles = self._styles(zs, y_idx)

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
