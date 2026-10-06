"""YAML-to-upstream TrainConfig bridge."""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(os.path.join(_REPO_ROOT, "diffae_upstream"))

import yaml
import templates
from choices import ModelName
from .attr_conditioner import load_cond_specs
from .hier_config import HDAEConfig, EncoderHierarchyConfig, ConditioningConfig
from .hier_encoder import stage_channels
from model.unet import BeatGANsEncoderConfig


@dataclass
class LoadedConfig:
    train_conf: object
    hdae_conf: HDAEConfig
    raw: dict
    path: str

    def lightning_kwargs(self):
        l, t = self.raw["lightning"], self.raw["train"]
        devices = int(l["devices"])
        return dict(gpus=devices if l["accelerator"] == "gpu" else 0,
                    accelerator="ddp" if l["strategy"] == "ddp" else None,
                    precision=16 if str(t["precision"]).startswith("16") else 32,
                    max_steps=t["max_steps"], gradient_clip_val=t["grad_clip"],
                    accumulate_grad_batches=int(t.get("accum_batches", 1)),
                    log_every_n_steps=l["log_every_n_steps"],
                    val_check_interval=l["val_check_interval"])


def load_hdae_config(path, require_data=True):
    with open(path) as f:
        raw = yaml.safe_load(f)
    conf = getattr(templates, raw["base_template"])()
    conf.model_name = ModelName.hier_autoenc
    t, l = raw["train"], raw["lightning"]
    conf.batch_size = t["batch_size_per_gpu"]
    # accum_batches must reach BOTH Lightning and conf, and for different reasons.
    #
    # Lightning uses it to decide when to call optimizer.step(). conf.accum_batches is what
    # LitModel.is_last_accum() reads, and HDAELitModule.on_train_batch_end gates the EMA update
    # on that. Setting only the Lightning side would leave conf.accum_batches at its default 1,
    # so EMA would advance once per MICRO-batch instead of once per optimiser step -- 8x too
    # fast at accum 8, silently, with nothing in the logs to show it.
    #
    # This was never exercised before: every earlier run used accum 1, and `total_batch_size`
    # in those configs was dead config that nothing read.
    conf.accum_batches = int(t.get("accum_batches", 1))
    eff = conf.batch_size * conf.accum_batches
    if "total_batch_size" in t and int(t["total_batch_size"]) != eff:
        raise ValueError(
            f"total_batch_size={t['total_batch_size']} contradicts "
            f"batch_size_per_gpu={conf.batch_size} x accum_batches={conf.accum_batches} = {eff}")
    conf.lr = t["lr"]
    conf.ema_decay = t["ema_decay"]
    conf.T = t["T"]
    conf.T_eval = t["T_eval"]
    conf.grad_clip = t["grad_clip"]
    conf.img_size = raw["data"]["image_size"]
    conf.style_ch = raw["conditioning"]["style_ch"]
    conf.make_model_conf()
    hdae = HDAEConfig(EncoderHierarchyConfig(**raw["encoder"]), ConditioningConfig(**raw["conditioning"]))
    if hdae.encoder.causal_graph_path:
        hdae.encoder.cond_specs = load_cond_specs(hdae.encoder.causal_graph_path, hdae.encoder.conditioning_attrs)
    conf.hdae_conf = hdae
    conf.make_model_conf()
    data_key = "h5_path" if raw["data"].get("type") in ("morphomnist", "causal3dident") else "lmdb_path"
    if require_data and not Path(raw["data"][data_key]).exists():
        raise FileNotFoundError(
            f"Packed data missing. Run: python experiments/hdae/scripts/preprocess_data.py --config {path}")
    return LoadedConfig(conf, hdae, raw, str(path))
