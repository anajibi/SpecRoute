"""Train the VAE-only hierarchy and deterministic latent decoder."""
import argparse
import time

import torch
from torch.nn import functional as F

from _common import checkpoint_path, encode, setup
from dino_vae_hierarchical_diffusion.src.losses import kl_loss
from dino_vae_hierarchical_diffusion.src.utils import atomic_torch_save, cpu_state_dict, ensure_output, levels, seed_all

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--max_images", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--split")
args = parser.parse_args()

config, device, loader, vae, dino, (evidence, encoder, deterministic, _) = setup(args, use_dino=False)
seed_all(config["seed"])
k = config["hierarchy"]["K"]
output_dir = ensure_output(config)
modules = {"evidence": evidence, "encoder": encoder, "deterministic": deterministic}
parameters = [parameter for module in modules.values() for parameter in module.parameters()]
optimizer = torch.optim.AdamW(parameters, lr=config["optim"]["lr"])
epochs = args.epochs or config["train"]["stage1_epochs"]
global_step = 0
if len(loader) == 0:
    raise RuntimeError("Stage 1 received an empty dataset; check dataset paths, split, and --max_images.")

print(
    f"Stage 1 starting: epochs={epochs}, images={len(loader.dataset)}, batches/epoch={len(loader)}, "
    f"device={device}, output={output_dir}",
    flush=True,
)
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch in loader:
        image = batch["image"].to(device)
        z0, posterior = encode(image, vae, dino, evidence, encoder)
        hierarchy = levels(posterior, k)
        z0_deterministic = deterministic(*hierarchy)
        loss = (
            F.l1_loss(z0_deterministic, z0)
            + config["loss"]["beta_kl"]
            * sum(kl_loss(posterior[f"mu{i}"], posterior[f"logvar{i}"]) for i in range(1, k + 1))
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config["optim"]["grad_clip"])
        optimizer.step()
        epoch_loss += loss.item()
        global_step += 1
    print(f"epoch={epoch + 1}/{epochs} mean_loss={epoch_loss / len(loader):.4f}", flush=True)

checkpoint = checkpoint_path(output_dir, "stage1.pt")
print(f"Training complete. Preparing CPU checkpoint: {checkpoint}", flush=True)
prepare_started = time.monotonic()
payload = {
    **{name: cpu_state_dict(module) for name, module in modules.items()},
    "config": config,
    "epoch": epochs,
    "global_step": global_step,
}
print(f"CPU checkpoint prepared in {time.monotonic() - prepare_started:.1f}s. Writing to disk...", flush=True)
save_started = time.monotonic()
atomic_torch_save(payload, checkpoint)
size_mib = checkpoint.stat().st_size / (1024 * 1024)
print(
    f"Stage 1 complete. Saved checkpoint to {checkpoint} ({size_mib:.1f} MiB) "
    f"in {time.monotonic() - save_started:.1f}s.",
    flush=True,
)
print(
    "Next: python experiments/dino_vae_hierarchical_diffusion/scripts/extract_latents.py "
    f"--config {args.config}" + (f" --max_images {args.max_images}" if args.max_images else ""),
    flush=True,
)
