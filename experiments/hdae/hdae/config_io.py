"""YAML-to-upstream TrainConfig bridge."""
from dataclasses import dataclass
from pathlib import Path
import yaml
import templates
from choices import ModelName
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
        """Trainer args compatible with the upstream pinned Lightning 1.4 API."""
        l, t = self.raw["lightning"], self.raw["train"]
        devices = int(l["devices"])
        return dict(gpus=devices if l["accelerator"] == "gpu" else 0,
                    accelerator="ddp" if l["strategy"] == "ddp" else None,
                    precision=16 if str(t["precision"]).startswith("16") else 32,
                    max_steps=t["max_steps"], gradient_clip_val=t["grad_clip"],
                    log_every_n_steps=l["log_every_n_steps"],
                    val_check_interval=l["val_check_interval"])


def _encoder_conf(conf):
    m=conf.model_conf
    return BeatGANsEncoderConfig(image_size=m.image_size,in_channels=m.in_channels,model_channels=m.model_channels,
        out_hid_channels=m.enc_out_channels,out_channels=m.enc_out_channels,num_res_blocks=m.enc_num_res_block,
        attention_resolutions=m.enc_attn_resolutions or m.attention_resolutions,dropout=m.dropout,
        channel_mult=m.enc_channel_mult or m.channel_mult,use_time_condition=False,conv_resample=m.conv_resample,
        dims=m.dims,use_checkpoint=m.use_checkpoint or m.enc_grad_checkpoint,num_heads=m.num_heads,
        num_head_channels=m.num_head_channels,resblock_updown=m.resblock_updown,
        use_new_attention_order=m.use_new_attention_order,pool=m.enc_pool)


def load_hdae_config(path, require_data=True):
    with open(path) as f: raw=yaml.safe_load(f)
    fn=getattr(templates, raw["base_template"], None)
    if fn is None: raise ValueError(f"unknown base_template: {raw['base_template']}")
    conf=fn(); conf.model_name=ModelName.hier_autoenc
    t,l=raw["train"],raw["lightning"]
    if t["total_batch_size"] != t["batch_size_per_gpu"] * int(l["devices"]):
        raise ValueError("total_batch_size must equal batch_size_per_gpu * devices")
    conf.batch_size=t["batch_size_per_gpu"]; conf.lr=t["lr"]; conf.ema_decay=t["ema_decay"]
    conf.T=t["T"]; conf.T_eval=t["T_eval"]; conf.grad_clip=t["grad_clip"]; conf.img_size=raw["data"]["image_size"]
    e=EncoderHierarchyConfig(**raw["encoder"]); c=ConditioningConfig(**raw["conditioning"])
    if e.type not in {"flat","hierarchical"}: raise ValueError("encoder.type must be flat or hierarchical")
    if c.strategy == "concat_proj" and sum(e.level_dims) != c.style_ch: raise ValueError("sum(level_dims) must equal style_ch")
    if not 0 <= c.latent_drop_prob < 1: raise ValueError("conditioning.latent_drop_prob must be in [0, 1)")
    conf.style_ch=c.style_ch; hdae=HDAEConfig(e,c); conf.hdae_conf=hdae; conf.make_model_conf()
    valid=stage_channels(_encoder_conf(conf))
    bad=set(e.tap_resolutions)-set(valid)
    if bad: raise ValueError(f"invalid taps {sorted(bad)}; valid: {sorted(valid)}")
    if require_data and not Path(raw["data"]["lmdb_path"]).exists():
        raise FileNotFoundError(f"Packed data missing. Run: python experiments/hdae/scripts/preprocess_data.py --config {path}")
    return LoadedConfig(conf,hdae,raw,str(path))
