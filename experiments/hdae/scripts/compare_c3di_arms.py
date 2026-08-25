"""Pick the best conditioning variant: reconstruction quality vs conditioning strength.

Two axes, both measured on identical held-out images with identical noise, so the arms are
compared pairwise rather than against separately-sampled baselines.

RECONSTRUCTION -- DDIM-encode each image to x_T, decode without guidance, compare to source.
    mse / psnr        pixel fidelity
    lpips             perceptual distance (AlexNet backbone)

CONDITIONING STRENGTH -- how much the attribute signal actually moves the denoiser. This is
the same idea as diagnostics/t9_conditioning_ablation.py: if conditioning carries real
information, giving the model true attributes must lower the diffusion loss relative to
giving it the learned null. Predictor-free, so it works before the attribute predictors
finish training.
    dL_all            L(all attrs nulled) - L(true attrs)   -- overall conditioning value
    dL_<attr>         L(only that attr nulled) - L(true)    -- per-attribute value
    cfg_delta         ||eps(cond) - eps(null)|| / ||eps(null)||, the size of the guidance
                      direction CFG actually amplifies at sample time

All timesteps and noise draws are seeded identically per model, so differences are the model.

    python experiments/hdae/scripts/compare_c3di_arms.py --n-recon 48 --n-loss 256
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

from experiments.hdae.data.causal3dident import Causal3DIdentPacked  # noqa: E402
from experiments.hdae.hdae.attr_utils import to_cond_values  # noqa: E402
from experiments.hdae.hdae.config_io import load_hdae_config  # noqa: E402
from experiments.hdae.hdae.lit_module import HDAELitModule  # noqa: E402

ARMS = [
    ("A fourier", "experiments/hdae/configs/c3di_hier_k1_fourier_b16.yaml",
     "experiments/hdae/outputs/c3di_k1_fourier_b16/checkpoints/epoch=3-step=50000.ckpt"),
    ("B rmsnorm", "experiments/hdae/configs/c3di_hier_k1_rmsnorm_b16.yaml",
     "experiments/hdae/outputs/c3di_k1_rmsnorm_b16/checkpoints/last.ckpt"),
    ("C both", "experiments/hdae/configs/c3di_hier_k1_both.yaml",
     "experiments/hdae/outputs/c3di_k1_both/checkpoints/epoch=6-step=50000.ckpt"),
]
ATTR_ORDER = ["class", "pos_spl", "pos_obj", "rot_obj"]


def load(cfg_path, ckpt, device):
    cfg = load_hdae_config(cfg_path, require_data=False)
    m = HDAELitModule.load_from_checkpoint(ckpt, conf=cfg.train_conf, map_location="cpu").to(device).eval()
    return cfg, m


@torch.no_grad()
def reconstruction(module, sampler, x, y, lpips_fn, device):
    zs = [z.clone() for z in module.ema_model.encode(x)]
    cond = module.ema_model.make_cond(zs, y)
    x_T = sampler.ddim_reverse_sample_loop(module.ema_model, x, model_kwargs={"cond": cond})["sample"]
    with torch.inference_mode():
        rec = sampler.sample(model=module.ema_model, noise=x_T, model_kwargs={"cond": cond})
    mse = F.mse_loss(rec, x).item()                      # both in [-1, 1]
    psnr = 10 * np.log10(4.0 / max(mse, 1e-12))          # peak-to-peak of [-1,1] is 2 -> 2^2
    lp = lpips_fn(rec.clamp(-1, 1), x).mean().item()
    return {"mse": round(mse, 6), "psnr": round(float(psnr), 3), "lpips": round(lp, 5)}, zs


@torch.no_grad()
def conditioning_strength(module, sampler, x, y, zs, device, seed, n_t=16):
    """Diffusion loss with true attrs vs with attrs nulled. Same t and same noise for every
    variant and every model."""
    model = module.ema_model
    B = x.shape[0]
    g = torch.Generator(device=device); g.manual_seed(seed)
    T = sampler.num_timesteps
    ts = torch.linspace(0, T - 1, n_t).long().to(device)
    noise = torch.randn(x.shape, device=device, generator=g)
    n_attr = len(module.model.hdae_conf.encoder.cond_specs)

    def loss_for(mask):
        tot = 0.0
        for t_val in ts:
            t = torch.full((B,), int(t_val), device=device, dtype=torch.long)
            x_t = sampler.q_sample(x, t, noise=noise)
            cond = model.make_cond(zs, y, null_mask=mask)
            eps = model.forward(x=x_t, t=t, cond=cond).pred
            tot += F.mse_loss(eps, noise).item()
        return tot / len(ts)

    l_true = loss_for(None)
    all_mask = torch.ones(B, n_attr, dtype=torch.bool, device=device)
    out = {"loss_true": round(l_true, 6),
           "dL_all": round(loss_for(all_mask) - l_true, 6)}
    for i, name in enumerate(ATTR_ORDER):
        m = torch.zeros(B, n_attr, dtype=torch.bool, device=device); m[:, i] = True
        out[f"dL_{name}"] = round(loss_for(m) - l_true, 6)

    # size of the CFG direction, averaged over the same timesteps
    rels = []
    for t_val in ts:
        t = torch.full((B,), int(t_val), device=device, dtype=torch.long)
        x_t = sampler.q_sample(x, t, noise=noise)
        e_c = model.forward(x=x_t, t=t, cond=model.make_cond(zs, y)).pred
        e_n = model.forward(x=x_t, t=t, cond=model.make_cond(zs, y, null_mask=all_mask)).pred
        rels.append(((e_c - e_n).flatten(1).norm(dim=1) / e_n.flatten(1).norm(dim=1)).mean().item())
    out["cfg_delta_rel"] = round(float(np.mean(rels)), 5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-recon", type=int, default=48)
    ap.add_argument("--n-loss", type=int, default=256)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(REPO, "experiments/hdae/outputs/c3di_arm_comparison.json"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(dev).eval()

    ds = Causal3DIdentPacked(os.path.join(REPO, "experiments/hdae/data/causal3dident/causal3dident_testset_128.h5"))
    rng = np.random.RandomState(args.seed)
    pick = sorted(rng.choice(len(ds), max(args.n_recon, args.n_loss), replace=False).tolist())
    batch = [ds[i] for i in pick]
    X = torch.stack([b["img"] for b in batch])
    Yraw = torch.stack([b["attr"] for b in batch])[:, :8]

    results = {}
    for label, cfgp, ckpt in ARMS:
        if not os.path.exists(os.path.join(REPO, ckpt)):
            print(f"{label}: checkpoint missing, skipping"); continue
        cfg, module = load(cfgp, os.path.join(REPO, ckpt), dev)
        sampler = module.conf._make_diffusion_conf(args.T).make_sampler()
        specs = cfg.hdae_conf.encoder.cond_specs
        step = torch.load(os.path.join(REPO, ckpt), map_location="cpu").get("global_step")

        rec_acc, cond_acc = [], []
        for s in range(0, args.n_recon, args.batch):
            x = X[s:s + args.batch].to(dev)
            y = to_cond_values(Yraw[s:s + args.batch], specs).to(dev)
            r, _ = reconstruction(module, sampler, x, y, lpips_fn, dev)
            rec_acc.append(r)
        for s in range(0, args.n_loss, args.batch):
            x = X[s:s + args.batch].to(dev)
            y = to_cond_values(Yraw[s:s + args.batch], specs).to(dev)
            with torch.no_grad():
                zs = [z.clone() for z in module.ema_model.encode(x)]
            cond_acc.append(conditioning_strength(module, sampler, x, y, zs, dev, args.seed + s))

        agg = {k: round(float(np.mean([d[k] for d in rec_acc])), 6) for k in rec_acc[0]}
        agg.update({k: round(float(np.mean([d[k] for d in cond_acc])), 6) for k in cond_acc[0]})
        agg["global_step"] = step
        agg["fourier_freqs"] = cfg.hdae_conf.encoder.fourier_freqs
        agg["attr_norm"] = cfg.hdae_conf.encoder.attr_norm
        agg["batch_size"] = cfg.raw["train"]["batch_size_per_gpu"]
        results[label] = agg
        print(f"{label}: {json.dumps(agg)}", flush=True)
        del module; torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", args.out)

    if results:
        print(f"\n{'arm':11s} {'step':>7s} {'mse':>9s} {'psnr':>7s} {'lpips':>8s} "
              f"{'dL_all':>9s} {'cfg_delta':>10s}")
        for k, v in results.items():
            print(f"{k:11s} {v['global_step']:>7} {v['mse']:>9.5f} {v['psnr']:>7.2f} "
                  f"{v['lpips']:>8.4f} {v['dL_all']:>9.5f} {v['cfg_delta_rel']:>10.4f}")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
