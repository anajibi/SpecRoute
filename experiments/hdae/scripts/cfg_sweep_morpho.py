"""Guidance sweep and counterfactual evaluation for MorphoMNIST -- the port of cfg_sweep_c3di.py.

Same protocol as Causal3DIdent: DDIM-invert each cohort image, ask the SCM what the world looks
like under do(attr = target), render with attribute-CFG at each guidance strength, then read all
twelve attributes back off the generated image. Per-sample errors are saved so any metric can be
recomputed later without re-rendering.

THREE THINGS DIFFER FROM THE C3DI VERSION, all forced by the dataset:

  1. The SCM is a different class. experiments.hdae.causal.scm.SCM takes a raw attribute TENSOR
     plus a column index map, and its propagate() returns Z-SPACE for continuous nodes -- _to_raw
     is the inverse. The C3DI SCM took a dict and returned raw units directly.
  2. The predictors are MIXED-ARCHITECTURE. ConvNeXt-Tiny (upsampled to 128, ImageNet-normalised)
     wins on eleven attributes, but its 4x4 stride-4 patchify stem destroys per-pixel noise, so
     texture_amplitude uses the SmallCNN at native 64 instead -- 1.8% of spread against ConvNeXt's
     88.5%. Each predictor declares its own preprocessing here.
  3. The graph has one edge, thickness -> intensity, so only do(thickness) has a descendant.

CAVEAT ON THAT EDGE, measured during SCM validation: the fitted coupling is ~10% too strong
(Pearson r +0.797 from the SCM against +0.723 in the data). do(thickness) therefore asks intensity
to move further than the renderer actually moves it, which biases descendant-CC downward for that
one intervention.
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

from PIL import Image, ImageDraw  # noqa: E402
from experiments.hdae.causal.graph import CausalGraph  # noqa: E402
from experiments.hdae.causal.scm import SCM, NodeSpec  # noqa: E402
from experiments.hdae.counterfactuals.hdae_adapter import AttributeCFGWrapper  # noqa: E402
from experiments.hdae.hdae.attr_utils import to_cond_values  # noqa: E402
from experiments.hdae.hdae.config_io import load_hdae_config  # noqa: E402
from experiments.hdae.hdae.lit_module import HDAELitModule  # noqa: E402
from torchvision.models import convnext_tiny  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
MODELLED = ["digit", "thickness", "intensity", "hue"]
UNOBS = ["rotation", "scale", "translate_x", "translate_y",
         "bg_freq", "bg_phase", "bg_amplitude", "texture_amplitude"]
ALL12 = MODELLED + UNOBS
EDGES = [("thickness", "intensity")]
CATEGORICAL = {"digit", "hue"}
# best predictor per attribute, chosen on held-out test error (see the retraining run)
PRED_DIR = {a: "attr_predictors_morpho_v2" for a in ALL12}
PRED_DIR["bg_amplitude"] = "attr_predictors_morpho_v3"
PRED_DIR["texture_amplitude"] = "SMALLCNN"



def load_hdae_module(cls, ckpt_path, conf):
    """Load a checkpoint that may carry adversary heads the eval-time module never builds.

    `_adv_loss` constructs its per-tap probe heads LAZILY, on the first training step, so a
    freshly-constructed module has none and Lightning's strict load rejects the checkpoint on
    the extra keys. Loading non-strictly is safe only in that direction, so this asserts the
    dangerous direction separately: every parameter the model itself expects must be present
    in the checkpoint. Extra keys are tolerated, missing ones still raise.
    """
    import torch as _torch
    sd = _torch.load(ckpt_path, map_location="cpu")["state_dict"]
    m = cls.load_from_checkpoint(ckpt_path, conf=conf, map_location="cpu", strict=False)
    missing = [k for k in m.state_dict() if k not in sd]
    if missing:
        raise RuntimeError(f"{len(missing)} model params absent from {ckpt_path}: {missing[:6]}")
    extra = [k for k in sd if k.startswith(("_adv_heads.", "adv_heads."))]
    if extra:
        print(f"ignored {len(extra)} adversary-head keys (training-only)", flush=True)
    return m

def descendants(a):
    out, fr = set(), [a]
    while fr:
        n = fr.pop()
        for p, c in EDGES:
            if p == n and c not in out:
                out.add(c); fr.append(c)
    return out


class Reader:
    """One attribute predictor plus the preprocessing it was trained with."""

    def __init__(self, attr, dev):
        self.attr, self.dev = attr, dev
        d = PRED_DIR[attr]
        if d == "SMALLCNN":
            from experiments.hdae.data.attr_predictor import load_attr_predictor
            spec = json.load(open(f"{REPO}/experiments/hdae/outputs/attr_predictors_morpho_smallcnn/"
                                  f"{attr}/spec.json"))
            m = load_attr_predictor(f"{REPO}/experiments/hdae/outputs/attr_predictors_morpho_smallcnn/"
                                    f"{attr}/best.ckpt", 0)
            self.net = (m.net if hasattr(m, "net") else m.model).to(dev).eval()
            self.kind, self.lo, self.hi, self.size, self.imagenet = "scalar", spec["lo"], spec["hi"], 64, False
        else:
            b = torch.load(f"{REPO}/experiments/hdae/outputs/{d}/{attr}.pt", map_location="cpu")
            sd = {k[4:] if k.startswith("net.") else k: v for k, v in b["state_dict"].items()}
            m = convnext_tiny()
            f = m.classifier[2].in_features
            m.classifier[2] = torch.nn.Sequential(torch.nn.Dropout(0.0),
                                                  torch.nn.Linear(f, b["out_dim"]))
            m.load_state_dict(sd)
            self.net = m.to(dev).eval()
            self.kind, self.lo, self.hi = b["kind"], b["lo"], b["hi"]
            self.size, self.imagenet = b.get("img_size", 128), True

    @torch.no_grad()
    def __call__(self, img, bs=128):
        out = []
        for i in range(0, len(img), bs):
            x = img[i:i + bs].to(self.dev)
            if self.imagenet:
                x = (x + 1) / 2
                if self.size != x.shape[-1]:
                    x = F.interpolate(x, size=(self.size,) * 2, mode="bilinear", align_corners=False)
                x = (x - IMAGENET_MEAN.to(self.dev)) / IMAGENET_STD.to(self.dev)
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(self.net(x).float().cpu())
        p = torch.cat(out)
        if self.kind == "categorical":
            return p.argmax(1).float()                       # class index
        v = p.reshape(-1).clamp(-1, 1)
        return (v + 1) / 2 * (self.hi - self.lo) + self.lo    # back to raw units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--h5", default=os.path.join(REPO, "experiments/hdae/data/morphomnist/morphomnist_70k_v2.h5"))
    ap.add_argument("--scm", default=os.path.join(REPO, "experiments/hdae/outputs/scm/morpho_scm.pt"))
    ap.add_argument("--strengths", type=float, nargs="+", default=[1, 1.5, 2, 2.5, 3, 5, 8])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--sample-bs", type=int, default=128)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid-rows", type=int, default=0)
    ap.add_argument("--attr-g", nargs="+", default=None)
    ap.add_argument("--outdir", default=os.path.join(REPO, "experiments/hdae/outputs/morpho_sweep"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    per_attr_g = None
    if args.attr_g:
        per_attr_g = {k: [float(v)] for k, v in (x.split("=") for x in args.attr_g)}
    dev = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed); rng = np.random.RandomState(args.seed)

    with h5py.File(args.h5, "r") as h:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in h["attribute_names"][:]]
        A_all = h["attrs"][:].astype(np.float64)
        part = h["partitions"][:]
        test = np.where(part == 1)[0]
        idx = np.sort(rng.choice(test, args.n, replace=False))
        X = torch.from_numpy(h["images"][idx].astype(np.float32) / 255.).permute(0, 3, 1, 2) * 2 - 1
    COL = {a: names.index(a) for a in ALL12}
    A = A_all[idx]

    # ---- intervention targets: a real move, not a nudge ----
    p5 = {a: np.percentile(A_all[:, COL[a]], 5) for a in ("thickness", "intensity")}
    p95 = {a: np.percentile(A_all[:, COL[a]], 95) for a in ("thickness", "intensity")}
    mind = {"thickness": 0.6, "intensity": 30.0}
    tgt = {}
    for a in MODELLED:
        src = A[:, COL[a]]
        if a in CATEGORICAL:
            n_cls = 10
            if a == "digit":
                tgt[a] = ((src.astype(int) + 5) % n_cls).astype(np.float64)
            else:  # hue stored as bin centre
                ci = np.round(src * n_cls - 0.5).astype(int)
                tgt[a] = ((ci + 5) % n_cls + 0.5) / n_cls
        else:
            v = rng.uniform(p5[a], p95[a], len(idx))
            for _ in range(500):
                bad = np.abs(v - src) < mind[a]
                if not bad.any():
                    break
                v[bad] = rng.uniform(p5[a], p95[a], int(bad.sum()))
            tgt[a] = v
    print("intervention targets: " + "  ".join(
        f"{a} mean|move|={np.abs(tgt[a]-A[:,COL[a]]).mean():.3f}" for a in MODELLED), flush=True)

    cfg = load_hdae_config(os.path.join(REPO, args.config), require_data=False)
    module = load_hdae_module(HDAELitModule, os.path.join(REPO, args.ckpt), cfg.train_conf).to(dev).eval()
    sampler = module.conf._make_diffusion_conf(args.T).make_sampler()
    specs_cond = cfg.hdae_conf.encoder.cond_specs
    b = torch.load(args.scm, map_location="cpu")
    nspecs = {k: (v if isinstance(v, NodeSpec) else NodeSpec(**v)) for k, v in b["node_specs"].items()}
    scm = SCM(CausalGraph(b["attributes"], b["edges"]), nspecs)
    scm.load_state_dict(b["state_dict"]); scm.eval()
    scm_index = {a: i for i, a in enumerate(b["attributes"])}
    readers = {a: Reader(a, dev) for a in ALL12}
    print(f"loaded {len(readers)} predictors", flush=True)

    x = X.to(dev)
    y = to_cond_values(torch.from_numpy(A[:, [COL[a] for a in MODELLED]]), specs_cond).to(dev)
    net = module.ema_model.eval()
    chunks = [(i, min(i + args.sample_bs, len(idx))) for i in range(0, len(idx), args.sample_bs)]

    def render(model, y_vec, x_T, zs):
        out = []
        for lo, hi in chunks:
            with torch.no_grad():
                c = net.make_cond([z[lo:hi] for z in zs], y_vec[lo:hi])
            with torch.no_grad(), torch.inference_mode():
                out.append(sampler.sample(model=model, noise=x_T[lo:hi], model_kwargs={"cond": c}).cpu())
        return torch.cat(out)

    with torch.no_grad():
        zs = [z.clone() for z in net.encode(x)]
        inv = []
        for lo, hi in chunks:
            c = net.make_cond([z[lo:hi] for z in zs], y[lo:hi])
            inv.append(sampler.ddim_reverse_sample_loop(net, x[lo:hi], model_kwargs={"cond": c})["sample"])
        x_T = torch.cat(inv)
    recon = render(net, y, x_T, zs)
    per_sample, results, panels = {}, {}, {}

    def to_class_index(k, raw):
        """Categorical comparison happens in CLASS INDEX space, not raw storage.

        `digit` stores the index itself (0..9). `hue` stores its BIN CENTRE (0.05 .. 0.95).
        The predictors return an index for both. Comparing hue's index against its bin centre
        marks every image wrong -- that bug showed up as a reconstruction error rate of exactly
        1.0000 before this conversion existed.
        """
        v = np.asarray(raw, dtype=np.float64)
        return np.round(v) if k == "digit" else np.round(v * 10.0 - 0.5)

    def score(key, k, read, ref):
        if k in CATEGORICAL:
            e = (read != torch.from_numpy(to_class_index(k, ref)).float()).float()
        else:
            e = (read - torch.from_numpy(np.asarray(ref, dtype=np.float64)).float()).abs()
        per_sample[f"{key}|{k}"] = e.numpy()
        return {"metric": "error_rate" if k in CATEGORICAL else "mae",
                "value": round(float(e.mean()), 5)}

    read_r = {a: readers[a](recon) for a in ALL12}
    results["recon"] = {a: dict(score("recon", a, read_r[a], A[:, COL[a]]), role="unchanged") for a in ALL12}
    print("[recon] " + "  ".join(f"{a}={results['recon'][a]['value']:.4f}" for a in MODELLED), flush=True)
    if args.grid_rows:
        panels[("recon", None)] = recon[:args.grid_rows]

    attrs_t = torch.from_numpy(A[:, [COL[a] for a in b["attributes"]]]).float()
    for a in MODELLED:
        desc = descendants(a)
        with torch.no_grad():
            eps = scm.abduct(attrs_t, scm_index)
            zc = scm.propagate(eps, {a: torch.from_numpy(tgt[a]).float().view(-1, 1)})
        cf = {}
        for k in b["attributes"]:
            v = zc[k].view(-1)
            cf[k] = (v if nspecs[k].kind == "categorical" else scm._to_raw(k, v.view(-1, 1)).view(-1)).numpy()
        y_cf = y.clone()
        for j, k in enumerate(MODELLED):
            y_cf[:, j] = torch.from_numpy(
                to_cond_values(torch.from_numpy(np.stack([cf[m] for m in MODELLED], 1)),
                               specs_cond).numpy()[:, j]).to(y.dtype).to(dev)
        for g in (per_attr_g[a] if per_attr_g else args.strengths):
            m = net if g == 1.0 else AttributeCFGWrapper(net, g).to(dev).eval()
            t0 = time.time()
            img = render(m, y_cf, x_T, zs)
            if args.grid_rows:
                panels[(a, g)] = img[:args.grid_rows]
            key = f"do({a})|g{g:g}"
            row = {}
            for k in ALL12:
                if k == a:
                    ref, role = tgt[a], "target"
                elif k in desc:
                    ref, role = cf[k], "descendant"
                else:
                    ref, role = A[:, COL[k]], "unchanged"
                row[k] = dict(score(key, k, readers[k](img), ref), role=role)
            results[key] = row
            print(f"  do({a:10s}) g={g:<4g} target {row[a]['value']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    if args.grid_rows:
        cols = [("source", X[:args.grid_rows]), ("recon", panels[("recon", None)])]
        for a in MODELLED:
            for g in (per_attr_g[a] if per_attr_g else args.strengths):
                cols.append((f"do({a})\ng={g:g}", panels[(a, g)]))
        cell, pad, hdr = 64, 3, 26
        sheet = Image.new("RGB", (len(cols) * (cell + pad) + pad,
                                  hdr + args.grid_rows * (cell + pad) + pad), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for ci, (nm, imgs) in enumerate(cols):
            for li, line in enumerate(nm.split("\n")):
                d.text((pad + ci * (cell + pad) + 1, 2 + li * 11), line, fill=(15, 15, 25))
            arr = ((imgs.clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(0, 2, 3, 1).numpy()
            for ri in range(args.grid_rows):
                sheet.paste(Image.fromarray(arr[ri]), (pad + ci * (cell + pad), hdr + ri * (cell + pad)))
        p = os.path.join(args.outdir, f"grid_{args.label}.png")
        sheet.save(p); print(f"wrote {p} {sheet.size}", flush=True)

    np.savez_compressed(os.path.join(args.outdir, f"persample_{args.label}.npz"), **per_sample)
    json.dump({"label": args.label, "ckpt": args.ckpt, "n": len(idx), "T": args.T,
               "strengths": args.strengths, "results": results,
               "cohort_idx": [int(i) for i in idx],
               "targets": {a: [float(v) for v in tgt[a]] for a in MODELLED}},
              open(os.path.join(args.outdir, f"sweep_{args.label}.json"), "w"), indent=2)
    print(f"\nwrote sweep_{args.label}.json")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
