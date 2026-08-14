"""CFModelAdapter for the conditioned per_block_attr HDAE (experiments/hdae/hdae/)."""
import torch

from experiments.hdae.hdae.attr_conditioner import ConcatAttributeEmbedding, MixedAttributeEmbedding

_NULL_MASK_AWARE = (MixedAttributeEmbedding, ConcatAttributeEmbedding)
from experiments.hdae.hdae.attr_utils import to_cond_values, to_index_space
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule

from .cf_contract import CFModelAdapter, CFState, register_adapter


class AttributeCFGWrapper(torch.nn.Module):
    """Dual forward pass for attribute classifier-free guidance at sample time."""

    def __init__(self, base_model, guidance_scale: float):
        super().__init__()
        self.base_model = base_model
        self.guidance_scale = float(guidance_scale)

    def forward(self, x, t, cond, **kwargs):
        cond_out = self.base_model.forward(x=x, t=t, cond=cond, **kwargs)
        if self.guidance_scale == 1.0:
            return cond_out
        if isinstance(self.base_model.attr_embedding, _NULL_MASK_AWARE):
            # Continuous attrs have no reserved "null" value (unlike binary's spare index 2) --
            # the unconditional pass is signalled via an explicit mask, not a magic input value.
            null_mask = torch.ones_like(cond["y_idx"], dtype=torch.bool)
            null_cond = {"zs": cond["zs"], "y_idx": cond["y_idx"], "null_mask": null_mask}
        else:
            y_null = torch.full_like(cond["y_idx"], 2)
            null_cond = {"zs": cond["zs"], "y_idx": y_null}
        uncond_out = self.base_model.forward(x=x, t=t, cond=null_cond, **kwargs)
        guided = uncond_out.pred + self.guidance_scale * (cond_out.pred - uncond_out.pred)
        return cond_out.__class__(pred=guided, cond=cond)


@register_adapter("hdae")
class HDAEAdapter(CFModelAdapter):
    """``edit_strength`` is the attribute-CFG guidance scale (>= 1.0)."""

    def __init__(self, module, modeled_attrs, attr_input_range, guidance_scale, T, device, cond_specs=None):
        self.module = module
        self.model = module.ema_model
        self.modeled_attrs = list(modeled_attrs)
        self._attr_input_range = attr_input_range
        self._cond_specs = cond_specs
        self.guidance_scale = float(guidance_scale)
        self.edit_strength = self.guidance_scale
        self.T = T
        self.device = device

    @classmethod
    def load(cls, config_path, ckpt_path, device, edit_strength=None, T=None, compile_model=True, **_):
        cfg = load_hdae_config(config_path)
        guidance_scale = float(edit_strength if edit_strength is not None else cfg.hdae_conf.conditioning.cfg_guidance_scale)
        if guidance_scale < 1.0:
            raise ValueError("HDAE edit_strength (attribute-CFG guidance_scale) must be >= 1.0")
        module = HDAELitModule.load_from_checkpoint(ckpt_path, conf=cfg.train_conf, map_location="cpu").to(device).eval()
        if compile_model:
            module.ema_model = torch.compile(module.ema_model)
        modeled_attrs = list(module.ema_model.hdae_conf.encoder.conditioning_attrs)
        attr_input_range = module.ema_model.hdae_conf.encoder.attr_input_range
        cond_specs = module.ema_model.hdae_conf.encoder.cond_specs or None
        eval_T = T if T is not None else cfg.raw["train"]["T_eval"]
        return cls(module, modeled_attrs, attr_input_range, guidance_scale, eval_T, device, cond_specs)

    def _sampler(self):
        if self.T is None:
            return self.module.eval_sampler
        return self.module.conf._make_diffusion_conf(self.T).make_sampler()

    def _encode_stochastic(self, x, cond):
        out = self._sampler().ddim_reverse_sample_loop(self.model, x, model_kwargs={"cond": cond})
        return out["sample"]

    def _render(self, noise, cond):
        render_model = self.model if self.guidance_scale == 1.0 else \
            AttributeCFGWrapper(self.model, self.guidance_scale).to(noise.device).eval()
        with torch.inference_mode():
            pred_img = self._sampler().sample(model=render_model, noise=noise, model_kwargs={"cond": cond})
        return (pred_img + 1) / 2

    def encode(self, images, attrs_raw, attr_names) -> CFState:
        cond_indices = [attr_names.index(a) for a in self.modeled_attrs]
        y_raw = attrs_raw[:, cond_indices].to(self.device)
        if self._cond_specs:
            y_idx = to_cond_values(y_raw, self._cond_specs).to(self.device)
        else:
            y_idx = to_index_space(y_raw, self._attr_input_range).to(self.device)
        with torch.inference_mode():
            zs = [z.clone() for z in self.model.encode(images)]
            cond = self.model.make_cond(zs, y_idx)
            x_t = self._encode_stochastic(images, cond)
        return {"zs": zs, "y_idx": y_idx, "x_t": x_t, "cond": cond}

    def intervene(self, state, attr, direction, cf_attrs) -> CFState:
        y_cf = state["y_idx"].clone()
        for i, a in enumerate(self.modeled_attrs):
            y_cf[:, i] = cf_attrs[a].to(device=y_cf.device, dtype=y_cf.dtype).view(-1)
        cf_cond = self.model.make_cond(state["zs"], y_cf)
        return {**state, "y_idx": y_cf, "cond": cf_cond}

    def render(self, state) -> torch.Tensor:
        return self._render(state["x_t"], state["cond"])
