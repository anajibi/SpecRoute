"""CFModelAdapter for a frozen, pretrained DiffAE edited via linear-probe directions.

No attribute-conditioned training, no CFG: an intervention is
``z_sem + alpha * w`` along a logistic-regression direction fit per attribute
(see ``train_diffae_directions.py``). This is the second concrete adapter for
the CFModelAdapter contract (TODO item 1's acceptance test) -- a genuinely
different model (single z_sem, no hierarchy, no attribute conditioning)
behind the same encode/intervene/render interface as HDAEAdapter.

Resolution note: the shared CelebA-HQ packed pipeline everything else in
experiments/hdae runs on is 64x64 (see AGENDA.md Sec.5); the only real frozen
DiffAE checkpoint available locally is ffhq256_autoenc (256x256, FFHQ, not
CelebA). This adapter resizes 64->native_image_size on encode and back to
64 on render, so CC/FC scoring stays on identical images across adapters.
That trades reconstruction/edit quality (upsampled input, domain mismatch)
for genericity -- it validates the contract, not this checkpoint's edit
quality.
"""
import sys

import torch
import torch.nn.functional as F
import yaml

sys.path.append("/home/anajibi/HDM/diffae_upstream")
import templates  # noqa: E402
from experiment import LitModel  # noqa: E402

from .cf_contract import CFModelAdapter, CFState, register_adapter

CLASSIFIER_RES = 64  # resolution the rest of the CF1 pipeline (dataset, attr classifier) operates at


@register_adapter("diffae_probe")
class DiffAEProbeAdapter(CFModelAdapter):
    """``edit_strength`` is alpha, the probe-direction offset magnitude."""

    def __init__(self, module, directions, modeled_attrs, native_image_size, edit_alpha, reverse_steps,
                 render_steps, device):
        self.module = module
        self.directions = directions
        self.modeled_attrs = list(modeled_attrs)
        self.native_image_size = int(native_image_size)
        self.edit_strength = float(edit_alpha)
        self.reverse_steps = int(reverse_steps)
        self.render_steps = int(render_steps)
        self.device = device

    @classmethod
    def load(cls, config_path, ckpt_path, device, edit_strength=None, T=None, **_):
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        conf = getattr(templates, raw["template"])()
        conf.pretrain = None
        conf.latent_infer_path = None
        conf.eval_programs = tuple()
        module = LitModel(conf)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        module.load_state_dict(state["state_dict"], strict=False)
        module = module.to(device).eval()
        module.ema_model.eval()

        modeled_attrs = list(raw["modeled_attrs"])
        directions_raw = torch.load(raw["directions_path"], map_location=device)
        missing = [a for a in modeled_attrs if a not in directions_raw]
        if missing:
            raise ValueError(f"directions_path {raw['directions_path']} missing directions for {missing}; "
                             f"run train_diffae_directions.py first")
        directions = {a: {"w": directions_raw[a]["w"].to(device).float()} for a in modeled_attrs}

        edit_alpha = edit_strength if edit_strength is not None else raw.get("edit_alpha", 1.0)
        reverse_steps = T if T is not None else raw.get("reverse_steps", 250)
        render_steps = T if T is not None else raw.get("render_steps", 20)
        return cls(module, directions, modeled_attrs, raw.get("native_image_size", 256), edit_alpha,
                   reverse_steps, render_steps, device)

    def _resize(self, x, size):
        if x.shape[-1] == size:
            return x
        return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)

    def encode(self, images, attrs_raw, attr_names) -> CFState:
        x = self._resize(images, self.native_image_size)
        with torch.inference_mode():
            z_sem = self.module.encode(x)
            x_t = self.module.encode_stochastic(x, z_sem, T=self.reverse_steps)
        return {"z_sem": z_sem, "x_t": x_t}

    def intervene(self, state, attr, direction, cf_attrs) -> CFState:
        sign = 1.0 if direction == "positive" else -1.0
        w = self.directions[attr]["w"].unsqueeze(0)
        z_cf = state["z_sem"] + sign * self.edit_strength * w
        return {**state, "z_sem": z_cf}

    def render(self, state) -> torch.Tensor:
        with torch.inference_mode():
            img = self.module.render(state["x_t"], cond=state["z_sem"], T=self.render_steps)
        return self._resize(img, CLASSIFIER_RES)
