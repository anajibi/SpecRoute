#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse, torch
from src.datasets import image_loader
from src.batches import batch_image_ids, batch_images
from src.metrics import cosine_similarity, mse_distance
from src.pipeline import load_models


def main():
    p = argparse.ArgumentParser();
    p.add_argument('--config', required=True);
    p.add_argument('--max-images', type=int, default=10);
    p.add_argument('--steps', type=int, default=50);
    a = p.parse_args();
    cfg = load_config(a.config);
    dev = get_device(cfg['device']);
    out = output_dir(cfg);
    dino, vae, enc, dec, pri = load_models(cfg, dev)
    with torch.no_grad():
        for batch in image_loader(cfg, split='test', shuffle=False, max_images=a.max_images):
            x = batch_images(batch, dev);
            ids = batch_image_ids(batch);
            zs = enc(dino(x));
            base = vae.decode(dec(zs));
            base_f = dino(base)
            for level in range(cfg['K']):
                mod = list(zs);
                cond = None if level == 0 else torch.cat(mod[:level], -1);
                mod[level] = pri.levels[level].sample(x.shape[0], cond, dev, a.steps);
                img = vae.decode(dec(mod));
                sim = cosine_similarity(base_f, dino(img));
                mse = mse_distance(base, img)
                for i, name in enumerate(ids): append_csv(out / 'counterfactual.csv',
                                                          {'image_id': name, 'K': cfg['K'], 'level': level + 1,
                                                           'dinov2_sim': sim[i].item(), 'image_mse': mse[i].item()})


if __name__ == '__main__': main()
