import argparse

import torch

from _common import checkpoint_path, encode, load_checkpoint, setup
from dino_vae_hierarchical_diffusion.src.utils import ensure_output

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--max_images", type=int)
parser.add_argument("--split")
args = parser.parse_args()
config, device, loader, vae, dino, (evidence, encoder, _, _) = setup(args)
output_dir = ensure_output(config)
load_checkpoint(checkpoint_path(output_dir, "stage1.pt"), {"evidence": evidence, "encoder": encoder}, device)
encoder.eval()

level_names = [f"Z{i}" for i in range(config["hierarchy"]["K"], 0, -1)]
posterior_names = [name for i in range(config["hierarchy"]["K"], 0, -1) for name in (f"mu{i}", f"logvar{i}")]
cache = {"image_ids": [], "z0": [], **{name: [] for name in level_names + posterior_names}}
with torch.no_grad():
    for batch in loader:
        z0, posterior = encode(batch["image"].to(device), vae, dino, evidence, encoder)
        cache["image_ids"].extend(batch["image_id"])
        cache["z0"].append(z0.cpu())
        for name in level_names + posterior_names:
            cache[name].append(posterior[name].cpu())

payload = {
    "image_ids": cache["image_ids"],
    "z0": torch.cat(cache["z0"]),
    "level_names": level_names,
    "latents": tuple(torch.cat(cache[name]) for name in level_names),
    "posterior_means": tuple(torch.cat(cache[f"mu{i}"]) for i in range(config["hierarchy"]["K"], 0, -1)),
    "posterior_logvars": tuple(torch.cat(cache[f"logvar{i}"]) for i in range(config["hierarchy"]["K"], 0, -1)),
}
torch.save(payload, output_dir / "latents.pt")
print(f"saved {len(payload['image_ids'])} latents")
