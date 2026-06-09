import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT.parent))

import torch

from dino_vae_hierarchical_diffusion.src.backbones import FrozenDINOv2, FrozenSDVAE
from dino_vae_hierarchical_diffusion.src.datasets import image_loader
from dino_vae_hierarchical_diffusion.src.utils import build_trainable, levels, load_config


def setup(args, trainable=True):
    config = load_config(args.config)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    loader = image_loader(
        config,
        split=getattr(args, "split", None),
        max_images=args.max_images,
        shuffle=getattr(args, "shuffle", None),
    )
    vae = FrozenSDVAE(config["backbones"]["vae_model_id"]).to(device)
    dino = FrozenDINOv2(config["backbones"]["dino_variant"]).to(device)
    modules = build_trainable(config["hierarchy"]["K"]) if trainable else ()
    return config, device, loader, vae, dino, tuple(module.to(device) for module in modules)


def encode(x, vae, dino, evidence, encoder):
    z0 = vae.encode(x)
    dino_cls, dino_map = dino(x)
    return z0, encoder(evidence(z0, dino_cls, dino_map))


def checkpoint_path(output_dir, name):
    return output_dir / "checkpoints" / name


def load_checkpoint(path, modules, device):
    state = torch.load(path, map_location=device, weights_only=False)
    for name, module in modules.items():
        if name in state:
            module.load_state_dict(state[name])
