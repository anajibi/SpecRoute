"""Conditional hierarchical semantic encoder with per-decoder-block injection."""
import torch

from diffae_upstream.model.nn import timestep_embedding
from diffae_upstream.model.unet import BeatGANsEncoderConfig
from diffae_upstream.model.unet_autoenc import AutoencReturn, BeatGANsAutoencModel

from .attr_conditioner import AttributeEmbedding, PerBlockStyle
from .hier_encoder import HierarchicalSemanticEncoder


class HierarchicalAutoencModel(BeatGANsAutoencModel):
    """DiffAE autoencoder whose decoder style is selected per output block.

    The upstream input/middle/output blocks are reused unchanged; only the decode
    loop is re-run so each decoder block receives its assigned style vector from
    ``z_level ⊕ attr_embedding``.
    """

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
        if len(e.hier_block_to_level) != len(self.output_blocks):
            raise ValueError(f"hier_block_to_level has {len(e.hier_block_to_level)} entries but decoder has {len(self.output_blocks)} output blocks")
        self.encoder = HierarchicalSemanticEncoder(enc_conf, e.hier_tap_block_ids, e.hier_level_dims,
                                                   pool="mean_std", proj=e.hier_proj)
        self.attr_embedding = AttributeEmbedding(e.n_attributes, e.attr_embed_dim, e.attr_dropout_prob)
        self.per_block_style = PerBlockStyle(e.hier_level_dims, e.hier_block_to_level,
                                             e.attr_embed_dim, conf.embed_channels, e.hier_proj)
        # Compatibility: render/encode_stochastic callers still pass a single
        # cond tensor.  We project concatenated latents to the old style width;
        # conditional training/eval paths should prefer ``cond_from_zs`` with y.
        self.legacy_merge = torch.nn.Linear(sum(e.hier_level_dims), conf.embed_channels)
        self.hdae_conf = hdae_conf
        self.last_zs = None
        self.last_null_mask = None

    def _default_y_idx(self, batch_size, device):
        # Null all attributes when no labels are supplied, matching CFG state=2.
        return torch.full((batch_size, self.hdae_conf.encoder.n_attributes), 2,
                          dtype=torch.long, device=device)

    def encode(self, x, null_levels=None):
        """Encode image into a list of K latents, coarse-to-fine.

        ``null_levels`` is accepted for legacy callers but no learned null-token
        machinery remains in the encoder; use per-level ablations in the decoder
        path for influence metrics.
        """
        zs = self.encoder(x)
        self.last_zs = zs
        cond = self.merge(zs, null_levels=null_levels)
        return {"cond": cond, "zs": zs, "null_mask": self.last_null_mask}


    def encode_with_nulls(self, x, null_levels):
        """Compatibility helper for test-time latent-level ablations."""
        return self.encode(x, null_levels=null_levels)

    def merge(self, zs, null_levels=None):
        forced = set(int(i) for i in (null_levels or []))
        merged = []
        masks = []
        for i, z in enumerate(zs):
            if i in forced:
                masks.append(torch.ones(z.shape[0], 1, dtype=torch.bool, device=z.device))
                merged.append(torch.zeros_like(z))
            else:
                masks.append(torch.zeros(z.shape[0], 1, dtype=torch.bool, device=z.device))
                merged.append(z)
        self.last_null_mask = torch.cat(masks, dim=1).detach() if masks else None
        return self.legacy_merge(torch.cat(merged, dim=1))

    def cond_from_zs(self, zs, y_idx=None):
        if y_idx is None:
            y_idx = self._default_y_idx(zs[0].shape[0], zs[0].device)
            apply_dropout = False
        else:
            apply_dropout = True
        attr = self.attr_embedding(y_idx, apply_dropout=apply_dropout)
        if attr.dtype != zs[0].dtype:
            attr = attr.to(dtype=zs[0].dtype)
        return self.per_block_style(zs, attr)

    def abduct(self, x):
        encoded = self.encode(x)
        return encoded["zs"], None

    def decode(self, x, t, zs, y_idx=None, t_cond=None):
        return self.forward(x=x, t=t, x_start=None, zs=zs, y_idx=y_idx, t_cond=t_cond).pred

    def forward(self, x, t, y=None, x_start=None, cond=None, style=None, noise=None,
                t_cond=None, y_idx=None, zs=None, **kwargs):
        if t_cond is None:
            t_cond = t
        if noise is not None:
            cond = self.noise_to_cond(noise)
        if zs is None:
            if cond is None:
                if x is not None:
                    assert len(x) == len(x_start), f'{len(x)} != {len(x_start)}'
                tmp = self.encode(x_start)
                zs = tmp["zs"]
                cond = tmp["cond"]
            else:
                zs = None
        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        else:
            _t_emb = None
            _t_cond_emb = None
        res = self.time_embed.forward(time_emb=_t_emb, cond=cond, time_cond_emb=_t_cond_emb)
        emb = res.time_emb if self.conf.resnet_two_cond else res.emb
        if zs is not None:
            block_styles = self.cond_from_zs(zs, y_idx=y_idx)
        else:
            # Legacy cond path: send the same projected cond to each block.
            block_styles = [res.emb for _ in self.output_blocks]
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        enc_cond_emb = None
        mid_cond_emb = None
        hs = [[] for _ in range(len(self.conf.channel_mult))]
        if x is not None:
            h = x.type(self.dtype)
            k = 0
            for i in range(len(self.input_num_blocks)):
                for _ in range(self.input_num_blocks[i]):
                    h = self.input_blocks[k](h, emb=enc_time_emb, cond=enc_cond_emb)
                    hs[i].append(h)
                    k += 1
            assert k == len(self.input_blocks)
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        else:
            h = None
        k = 0
        for i in range(len(self.output_num_blocks)):
            for _ in range(self.output_num_blocks[i]):
                try:
                    lateral = hs[-i - 1].pop()
                except IndexError:
                    lateral = None
                h = self.output_blocks[k](h, emb=dec_time_emb, cond=block_styles[k], lateral=lateral)
                k += 1
        pred = self.out(h)
        return AutoencReturn(pred=pred, cond=self.merge(zs) if zs is not None else cond)
