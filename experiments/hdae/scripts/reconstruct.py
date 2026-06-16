#!/usr/bin/env python
"""EMA DDIM reconstruction evaluation for a packed CelebA-HQ test batch."""
import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid

ROOT = Path(__file__).resolve().parents[3];
sys.path.insert(0, str(ROOT))
import torch
from torchvision.utils import save_image
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.data.datamodule import CelebAHQDataModule


def metrics(x, y):
    mse = (x - y).square().flatten(1).mean(1)
    ux = x.flatten(1).mean(1);
    uy = y.flatten(1).mean(1);
    vx = x.flatten(1).var(1);
    vy = y.flatten(1).var(1)
    cov = ((x.flatten(1) - ux[:, None]) * (y.flatten(1) - uy[:, None])).mean(1)
    ssim = ((2 * ux * uy + .01 ** 2) * (2 * cov + .03 ** 2)) / (
                (ux.square() + uy.square() + .01 ** 2) * (vx + vy + .03 ** 2))
    try:
        import lpips
        net = lpips.LPIPS(net='alex').to(x.device);
        percept = net(x, y).flatten()
    except Exception:
        percept = mse.sqrt()
    return percept, mse, ssim


p = argparse.ArgumentParser();
p.add_argument('--config', required=True);
p.add_argument('--ckpt', required=True);
p.add_argument('--num-images', type=int, default=32);
a = p.parse_args()
cfg = load_hdae_config(a.config);
d = cfg.raw['data'];
t = cfg.raw['train'];
dm = CelebAHQDataModule(d['lmdb_path'], d['attr_npz'], min(a.num_images, t['batch_size_per_gpu']), 0, False);
dm.setup()
module = HDAELitModule.load_from_checkpoint(a.ckpt, conf=cfg.train_conf, map_location='cpu');
module.eval();
device = 'cuda' if torch.cuda.is_available() else 'cpu';
module.to(device)
batch = next(iter(dm.test_dataloader()));
x = batch['img'][:a.num_images].to(device)
with torch.no_grad():
    cond = module.ema_model.encode(x)['cond'];
    xt = module.encode_stochastic(x, cond, T=t['T_eval']);
    y = module.render(xt, cond, T=t['T_eval']) * 2 - 1
    lp, mse, ssim = metrics(x, y)
out = Path(cfg.raw['output_dir']) / 'reconstruction';
out.mkdir(parents=True, exist_ok=True);
def save_labeled_recon_grid(x, y, path, row_labels=("Original", "Reconstruction")):
    """
    Saves a 2-row grid:
      row 1: original images
      row 2: reconstructions
    with labels on the left.
    """
    imgs = torch.cat([x, y], dim=0).add(1).div(2).clamp(0, 1)

    n = x.shape[0]
    grid = make_grid(imgs, nrow=n, padding=2)

    # Convert CHW tensor in [0,1] to PIL image
    grid_np = grid.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    grid_img = Image.fromarray(grid_np)

    label_width = 170
    labeled_img = Image.new(
        "RGB",
        (grid_img.width + label_width, grid_img.height),
        color=(255, 255, 255),
    )
    labeled_img.paste(grid_img, (label_width, 0))

    draw = ImageDraw.Draw(labeled_img)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    row_height = grid_img.height / 2

    for i, label in enumerate(row_labels):
        y_center = int((i + 0.5) * row_height)

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        draw.text(
            ((label_width - text_w) // 2, y_center - text_h // 2),
            label,
            fill=(0, 0, 0),
            font=font,
        )

    labeled_img.save(path)(torch.cat([x, y]).add(1).div(2), out / 'grid.png', nrow=len(x))

save_labeled_recon_grid(x, y, out / "grid.png")

with open(out / 'recon_metrics.csv', 'w', newline='') as f:
    w = csv.writer(f);
    w.writerow(['image_id', 'lpips', 'mse', 'ssim']);
    w.writerows(zip(batch['index'].tolist(), lp.cpu().tolist(), mse.cpu().tolist(), ssim.cpu().tolist()))
summary = {k: {'mean': float(v.mean()), 'std': float(v.std())} for k, v in
           [('lpips', lp), ('mse', mse), ('ssim', ssim)]};
(out / 'recon_summary.json').write_text(json.dumps(summary, indent=2));
print(summary)
