"""VRAM / throughput profile for a Causal3DIdent HDAE arm, before committing GPU time.

Runs real training steps (encode -> diffusion training_losses -> backward -> step) at a
range of batch sizes and reports peak VRAM and it/s for each, plus the projected wall-clock
for a 50k-step run -- both alone and with N arms sharing one GPU.

    python experiments/hdae/scripts/bench_causal3dident.py --config <cfg> --batches 16 32 64
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from experiments.hdae.data.causal3dident import Causal3DIdentPacked  # noqa: E402
from experiments.hdae.hdae.attr_utils import to_cond_values  # noqa: E402
from experiments.hdae.hdae.config_io import load_hdae_config  # noqa: E402
from experiments.hdae.hdae.lit_module import HDAELitModule  # noqa: E402


def cond_columns(attribute_names, specs, conditioning_attrs):
    dims = {s.name: int(getattr(s, "dim", 1)) for s in specs}
    idx = []
    for name in conditioning_attrs:
        d = dims.get(name, 1)
        idx.extend([attribute_names.index(name)] if d == 1
                   else [attribute_names.index(f"{name}_{j}") for j in range(d)])
    return idx


def bench(cfg_path, batch_sizes, steps, warmup, h5_override, device):
    cfg = load_hdae_config(cfg_path, require_data=False)
    e = cfg.hdae_conf.encoder
    h5 = h5_override or cfg.raw["data"]["h5_path"]
    ds = Causal3DIdentPacked(h5, preload_images=False)
    cols = cond_columns(ds.attribute_names, e.cond_specs, e.conditioning_attrs)
    print(f"config     : {os.path.basename(cfg_path)}")
    print(f"fourier    : {e.fourier_freqs}   attr_norm: {e.attr_norm}   fusion: "
          f"{cfg.hdae_conf.conditioning.attr_fusion}")
    print(f"data       : {h5}  ({len(ds)} images @ {ds.image_size}px)")
    print(f"cond cols  : {cols}  ({len(cols)} raw columns for {e.conditioning_attrs})")

    results = []
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            module = HDAELitModule(cfg.train_conf).to(device)
            module.train()
            nparam = sum(p.numel() for p in module.model.parameters())
            opt = torch.optim.AdamW(module.model.parameters(), lr=1e-4)
            scaler = torch.cuda.amp.GradScaler()

            idx = torch.randint(0, len(ds), (bs,))
            batch = [ds[int(i)] for i in idx]
            img = torch.stack([b["img"] for b in batch]).to(device)
            attr = torch.stack([b["attr"] for b in batch])[:, cols]
            y = to_cond_values(attr, e.cond_specs).to(device)

            def one_step():
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    zs = module.model.encode(img)
                    t, _ = module.T_sampler.sample(img.shape[0], device)
                    losses = module.sampler.training_losses(
                        model=module.model, x_start=img, t=t,
                        model_kwargs={"cond": module.model.make_cond(zs, y)})
                    loss = losses["loss"].mean()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                return float(loss.detach())

            for _ in range(warmup):
                one_step()
            torch.cuda.synchronize()
            t0 = time.time()
            last = None
            for _ in range(steps):
                last = one_step()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / steps
            peak = torch.cuda.max_memory_allocated() / 2**30
            resv = torch.cuda.max_memory_reserved() / 2**30
            results.append((bs, peak, resv, dt, last))
            print(f"  batch {bs:4d}: peak {peak:6.2f} GiB (reserved {resv:6.2f})  "
                  f"{dt*1000:7.1f} ms/step  {1/dt:6.2f} it/s  loss {last:.4f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {bs:4d}: OOM", flush=True)
        finally:
            del module, opt
            torch.cuda.empty_cache()

    if results:
        print(f"\nmodel parameters: {nparam:,}")
        print(f"\n{'batch':>6} {'peak GiB':>9} {'it/s':>7} {'50k alone':>12} {'50k, 3 arms sharing':>21}")
        for bs, peak, resv, dt, _ in results:
            solo = 50000 * dt / 3600
            print(f"{bs:>6} {peak:>9.2f} {1/dt:>7.2f} {solo:>10.1f} h {solo*3:>18.1f} h")
        print("\n'3 arms sharing' assumes the GPU serialises them (compute-bound), which is the "
              "pessimistic bound; real concurrency recovers some of it.")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/hdae/configs/c3di_hier_k1_both.yaml")
    ap.add_argument("--batches", type=int, nargs="+", default=[16, 32, 64, 96])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--h5", default=None, help="override data.h5_path (e.g. the testset)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    bench(args.config, args.batches, args.steps, args.warmup, args.h5, torch.device(args.device))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
