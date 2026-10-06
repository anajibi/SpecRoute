"""Guidance sweep for CelebA-HQ: CC and FC per attribute per g, for one depth.

All four attributes are binary, so the metrics are simpler than the MorphoMNIST versions:
  CC_a = P(predictor reads the REQUESTED value of a, after do(a))      -- did the edit land
  FC_a = P(predictor reads the UNCHANGED value of b != a, after do(a)) -- did the rest hold
Both are already in [0,1] with no baseline normalisation needed, unlike the continuous case.

Interventions respect the attestation rule: do(Beard) and do(Bald) are only requested of male
subjects, since Bald&Female is 1 image in 30,000 and Beard&Female is 10.
"""
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn as nn
from torchvision.models import convnext_tiny
sys.path.insert(0, "/home/exouser/SpecRoute/diffae_upstream")
sys.path.insert(0, "/home/exouser/SpecRoute")
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.attr_utils import to_cond_values
from experiments.hdae.counterfactuals.hdae_adapter import AttributeCFGWrapper as CFG

ATTRS = ["Male", "Young", "Beard", "Bald"]
MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)


class Reader:
    def __init__(self, attr, dev):
        b = torch.load(f"experiments/hdae/outputs/attr_predictors_celebahq/{attr}.pt", map_location="cpu")
        m = convnext_tiny(); m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
        m.load_state_dict(b["state_dict"]); self.net = m.to(dev).eval()
        self.size = b["img_size"]; self.dev = dev

    @torch.no_grad()
    def __call__(self, img, bs=32):
        out = []
        for i in range(0, len(img), bs):
            x = (img[i:i+bs].to(self.dev) + 1) / 2
            if x.shape[-1] != self.size:
                x = torch.nn.functional.interpolate(x, size=(self.size,)*2, mode="bilinear", align_corners=False)
            x = (x - MEAN.to(self.dev)) / STD.to(self.dev)
            with torch.autocast("cuda", dtype=torch.float16):
                out.append((self.net(x).squeeze(1) > 0).float().cpu())
        return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)              # k11 | k1
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--strengths", type=float, nargs="+", default=[1, 1.25, 1.5, 2, 2.5, 3])
    a = ap.parse_args()
    dev = "cuda"
    ck = sorted(glob.glob(f"experiments/hdae/outputs/celebahq256_{a.tag}/checkpoints/*.ckpt"))
    ck = [c for c in ck if "last" not in c] or ck
    cfg = load_hdae_config(f"experiments/hdae/configs/celebahq256_{a.tag}.yaml")
    net = HDAELitModule.load_from_checkpoint(ck[-1], conf=cfg.train_conf, map_location="cpu").ema_model.cuda().eval()
    samp = cfg.train_conf._make_diffusion_conf(a.T).make_sampler()
    specs = cfg.hdae_conf.encoder.cond_specs
    rd = {x: Reader(x, dev) for x in ATTRS}
    print(f"[{a.tag}] {ck[-1]}", flush=True)

    D = "experiments/hdae/data/packed"
    ds = CelebAHQPacked(f"{D}/celebahq_256.lmdb", f"{D}/celebahq_256_attrs.npz")
    names = ds.attribute_names; cols = [names.index(x) for x in ATTRS]
    test = np.where(ds.partitions == 2)[0]
    A = ds.attrs[test][:, cols]
    rng = np.random.default_rng(0)
    pools = {x: (test[A[:, 0] == 1] if x in ("Beard", "Bald") else test) for x in ATTRS}
    res = {}
    for ai, attr in enumerate(ATTRS):
        pool = pools[attr]
        idx = np.sort(rng.choice(pool, min(a.n, len(pool)), replace=False))
        x = torch.stack([ds[int(i)]["img"] for i in idx])
        y_raw = torch.from_numpy(ds.attrs[idx][:, cols].astype("float32"))
        y = to_cond_values(y_raw, specs)
        tgt = (-(y_raw[:, ai]) > 0).float()                     # requested value after the flip
        keep = {b: (y_raw[:, bi] > 0).float() for bi, b in enumerate(ATTRS) if b != attr}
        for g in a.strengths:
            P = {b: [] for b in ATTRS}
            for lo in range(0, len(idx), a.bs):
                xb = x[lo:lo+a.bs].cuda(); yb = y[lo:lo+a.bs].cuda()
                with torch.no_grad():
                    zs = [z.clone() for z in net.encode(xb)]
                    c = net.make_cond(zs, yb)
                    xT = samp.ddim_reverse_sample_loop(net, xb, model_kwargs={"cond": c})["sample"]
                    y2 = yb.clone(); y2[:, ai] = -y2[:, ai]
                    m = net if g == 1 else CFG(net, float(g)).cuda().eval()
                    ed = samp.sample(model=m, noise=xT, model_kwargs={"cond": net.make_cond(zs, y2)})
                for b in ATTRS: P[b].append(rd[b](ed.cpu()))
            P = {b: torch.cat(v) for b, v in P.items()}
            cc = float((P[attr] == tgt).float().mean())
            fc = float(np.mean([float((P[b] == keep[b]).float().mean()) for b in keep]))
            res[f"{attr}|g{g:g}"] = dict(CC=cc, FC=fc, CF1=0.0 if cc+fc == 0 else 2*cc*fc/(cc+fc))
            print(f"  do({attr:6s}) g={g:<5g} CC {cc:.4f}  FC {fc:.4f}  CF1 {res[f'{attr}|g{g:g}']['CF1']:.4f}", flush=True)
    os.makedirs("experiments/hdae/outputs/celebahq_sweep", exist_ok=True)
    json.dump(dict(tag=a.tag, n=a.n, T=a.T, strengths=a.strengths, results=res),
              open(f"experiments/hdae/outputs/celebahq_sweep/sweep_{a.tag}.json", "w"), indent=2)
    print(f"wrote celebahq_sweep/sweep_{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
