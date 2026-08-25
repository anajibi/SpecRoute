"""Is the GPU actually the bottleneck? Phase-level profile of one training step.

Answers it three ways rather than one, because a single number can't distinguish
"GPU saturated" from "GPU waiting":

1. LOADER-ONLY   iterate the DataLoader with no GPU work at all -> images/s the input
                 pipeline can supply. If this is below the training rate, data is the wall.
2. CACHED-BATCH  train on one batch held in VRAM, no loader, no H2D -> the pure-GPU
                 ceiling for this model and batch size.
3. REAL          the actual training loop. REAL / CACHED is the fraction of the ceiling
                 being reached; the shortfall is stall (loading, transfer, sync).

Then a CUDA-event breakdown of encode / forward+loss / backward / optimizer inside a step,
plus sampled GPU utilization, so a slow phase can be attributed.

    python experiments/hdae/scripts/profile_training.py --config <cfg> --batch 32
"""
import argparse
import os
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

from experiments.hdae.data.causal3dident import Causal3DIdentPacked  # noqa: E402
from experiments.hdae.hdae.attr_utils import to_cond_values  # noqa: E402
from experiments.hdae.hdae.config_io import load_hdae_config  # noqa: E402
from experiments.hdae.hdae.lit_module import HDAELitModule  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


class GpuSampler(threading.Thread):
    """Sample SM utilization in the background so we can tell a busy GPU from an idle one."""

    def __init__(self, period=0.25):
        super().__init__(daemon=True)
        self.period, self.samples, self.stop_flag = period, [], False

    def run(self):
        while not self.stop_flag:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                      "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip()
                self.samples.append(int(out.splitlines()[0]))
            except Exception:
                pass
            time.sleep(self.period)

    def stop(self):
        self.stop_flag = True
        self.join(timeout=3)
        return self.samples


def make_step(module, cols_specs, dev):
    model = module.model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    ev = {k: (torch.cuda.Event(True), torch.cuda.Event(True))
          for k in ("encode", "fwd", "bwd", "opt")}

    def step(img, y, timed=False):
        if timed: ev["encode"][0].record()
        with torch.autocast("cuda", dtype=torch.float16):
            zs = model.encode(img)
        if timed: ev["encode"][1].record(); ev["fwd"][0].record()
        with torch.autocast("cuda", dtype=torch.float16):
            t, _ = module.T_sampler.sample(img.shape[0], dev)
            losses = module.sampler.training_losses(model=model, x_start=img, t=t,
                                                    model_kwargs={"cond": model.make_cond(zs, y)})
            loss = losses["loss"].mean()
        if timed: ev["fwd"][1].record(); ev["bwd"][0].record()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if timed: ev["bwd"][1].record(); ev["opt"][0].record()
        scaler.step(opt); scaler.update()
        if timed: ev["opt"][1].record()
        return float(loss.detach())

    return step, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/hdae/configs/c3di_hier_k1_final.yaml")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--h5", default="experiments/hdae/data/causal3dident/causal3dident_trainset_128.h5")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device)
    cfg = load_hdae_config(os.path.join(REPO, args.config), require_data=False)
    specs = cfg.hdae_conf.encoder.cond_specs
    ds = Causal3DIdentPacked(os.path.join(REPO, args.h5))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    module = HDAELitModule(cfg.train_conf).to(dev); module.train()
    step, ev = make_step(module, specs, dev)
    print(f"config={os.path.basename(args.config)}  batch={args.batch}  workers={args.workers}  "
          f"params={sum(p.numel() for p in module.model.parameters())/1e6:.1f}M", flush=True)

    # ---- 1. loader-only ceiling
    it = iter(loader)
    for _ in range(4): next(it)
    t0 = time.time(); nb = 0
    for _ in range(args.steps):
        next(it); nb += 1
    loader_ips = nb * args.batch / (time.time() - t0)
    print(f"\n1. LOADER-ONLY   {loader_ips:8.1f} img/s   "
          f"({nb} batches, {args.workers} workers, no GPU work)", flush=True)

    # ---- 2. cached-batch ceiling (pure GPU)
    b = next(it)
    img_c = b["img"].to(dev, non_blocking=True)
    y_c = to_cond_values(b["attr"][:, :8], specs).to(dev)
    for _ in range(args.warmup): step(img_c, y_c)
    torch.cuda.synchronize()
    smp = GpuSampler(); smp.start()
    t0 = time.time()
    for _ in range(args.steps): step(img_c, y_c)
    torch.cuda.synchronize()
    cached_ips = args.steps * args.batch / (time.time() - t0)
    util_cached = smp.stop()
    print(f"2. CACHED-BATCH  {cached_ips:8.1f} img/s   pure-GPU ceiling "
          f"(GPU util {statistics.mean(util_cached):.0f}%)", flush=True)

    # ---- 3. real loop
    it = iter(loader)
    for _ in range(args.warmup):
        b = next(it)
        step(b["img"].to(dev, non_blocking=True), to_cond_values(b["attr"][:, :8], specs).to(dev))
    torch.cuda.synchronize()
    smp = GpuSampler(); smp.start()
    wait = 0.0; t0 = time.time()
    for _ in range(args.steps):
        tw = time.time(); b = next(it); wait += time.time() - tw
        step(b["img"].to(dev, non_blocking=True), to_cond_values(b["attr"][:, :8], specs).to(dev))
    torch.cuda.synchronize()
    el = time.time() - t0
    real_ips = args.steps * args.batch / el
    util_real = smp.stop()
    print(f"3. REAL LOOP     {real_ips:8.1f} img/s   "
          f"(GPU util {statistics.mean(util_real):.0f}%)", flush=True)

    frac = real_ips / cached_ips
    print(f"\n   real / ceiling      = {frac*100:5.1f}%")
    print(f"   time waiting on data= {wait/el*100:5.1f}%  ({wait:.2f}s of {el:.2f}s)")
    verdict = ("GPU-BOUND -- the input pipeline is keeping up" if frac > 0.92 else
               "DATA-BOUND -- the loader is the wall" if loader_ips < cached_ips else
               "PARTIALLY STALLED -- loader can keep up in isolation but not overlapped")
    print(f"   VERDICT: {verdict}")

    # ---- 4. phase breakdown
    torch.cuda.synchronize()
    acc = {k: [] for k in ev}
    for _ in range(args.steps):
        step(img_c, y_c, timed=True)
        torch.cuda.synchronize()
        for k, (a, bb) in ev.items():
            acc[k].append(a.elapsed_time(bb))
    tot = sum(statistics.mean(v) for v in acc.values())
    print(f"\n4. PHASE BREAKDOWN (cached batch, {args.steps} steps)")
    for k in ("encode", "fwd", "bwd", "opt"):
        m = statistics.mean(acc[k])
        print(f"   {k:8s} {m:7.2f} ms  {m/tot*100:5.1f}%")
    print(f"   {'TOTAL':8s} {tot:7.2f} ms  -> {1000/tot:.2f} it/s, {args.batch*1000/tot:.0f} img/s")
    print(f"\n   peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
          f"(reserved {torch.cuda.max_memory_reserved()/2**30:.2f})")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
