import argparse

import torch
from torch.utils.data import DataLoader

from dino_vae_hierarchical_diffusion.src.datasets import LatentDataset
from dino_vae_hierarchical_diffusion.src.priors import SpatialDiffusionPrior, VectorDiffusionPrior
from dino_vae_hierarchical_diffusion.src.utils import ensure_output, load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--epochs", type=int)
parser.add_argument("--max_images", type=int)
args = parser.parse_args()
config = load_config(args.config)
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
output_dir = ensure_output(config)
dataset = LatentDataset(output_dir / "latents.pt")
if args.max_images is not None:
    dataset = torch.utils.data.Subset(dataset, range(min(args.max_images, len(dataset))))
loader = DataLoader(dataset, batch_size=config.get("stage2", {}).get("batch_size", 32), shuffle=True)
k = config["hierarchy"]["K"]

if k == 3:
    priors = {
        "s3": VectorDiffusionPrior(512).to(device),
        "s2": SpatialDiffusionPrior(128, 8, 0, 512).to(device),
        "s1": SpatialDiffusionPrior(64, 16, 128, 512).to(device),
    }
else:
    priors = {
        "s5": VectorDiffusionPrior(512).to(device),
        "s4": VectorDiffusionPrior(256, 512).to(device),
        "s3": SpatialDiffusionPrior(128, 4, 0, 768).to(device),
        "s2": SpatialDiffusionPrior(64, 8, 128, 768).to(device),
        "s1": SpatialDiffusionPrior(64, 16, 64, 768).to(device),
    }
optimizer = torch.optim.AdamW(
    [parameter for prior in priors.values() for parameter in prior.parameters()], lr=config["optim"]["lr"]
)
for epoch in range(args.epochs or config["train"]["stage2_epochs"]):
    for levels in loader:
        levels = tuple(level.to(device) for level in levels)
        zs = {f"Z{i}": level for i, level in zip(range(k, 0, -1), levels)}
        loss = priors[f"s{k}"].loss(zs[f"Z{k}"])
        if k == 3:
            loss = loss + priors["s2"].loss(zs["Z2"], None, zs["Z3"])
            loss = loss + priors["s1"].loss(zs["Z1"], zs["Z2"], zs["Z3"])
        else:
            global_condition = torch.cat([zs["Z5"], zs["Z4"]], dim=1)
            loss = loss + priors["s4"].loss(zs["Z4"], zs["Z5"])
            loss = loss + priors["s3"].loss(zs["Z3"], None, global_condition)
            loss = loss + priors["s2"].loss(zs["Z2"], zs["Z3"], global_condition)
            loss = loss + priors["s1"].loss(zs["Z1"], zs["Z2"], global_condition)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch={epoch + 1} loss={loss.item():.4f}")
(output_dir / "checkpoints").mkdir(exist_ok=True)
torch.save({name: prior.state_dict() for name, prior in priors.items()}, output_dir / "checkpoints" / "priors.pt")
