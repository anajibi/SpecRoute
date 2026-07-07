"""YAML-to-upstream TrainConfig bridge."""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append("/home/anajibi/HDM/diffae_upstream")

import yaml
import templates
from choices import ModelName
from .hier_config import HDAEConfig, EncoderHierarchyConfig, ConditioningConfig


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
                    log_every_n_steps=l["log_every_n_steps"],
                    val_check_interval=l["val_check_interval"])


def load_hdae_config(path, require_data=True):
    with open(path) as f:
        raw = yaml.safe_load(f)
    conf = getattr(templates, raw["base_template"])()
    conf.model_name = ModelName.hier_autoenc
    t, l = raw["train"], raw["lightning"]
    conf.batch_size = t["batch_size_per_gpu"]
    conf.lr = t["lr"]
    conf.ema_decay = t["ema_decay"]
    conf.T = t["T"]
    conf.T_eval = t["T_eval"]
    conf.grad_clip = t["grad_clip"]
    conf.img_size = raw["data"]["image_size"]
    conf.style_ch = raw["conditioning"]["style_ch"]
    conf.make_model_conf()
    hdae = HDAEConfig(EncoderHierarchyConfig(**raw["encoder"]), ConditioningConfig(**raw["conditioning"]))
    conf.hdae_conf = hdae
    conf.make_model_conf()
    if require_data and not Path(raw["data"]["lmdb_path"]).exists():
        raise FileNotFoundError(f"Packed data missing. Run: python experiments/hdae/scripts/preprocess_data.py --config {path}")
    return LoadedConfig(conf, hdae, raw, str(path))
