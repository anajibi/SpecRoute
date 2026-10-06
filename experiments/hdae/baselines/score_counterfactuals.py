"""Score counterfactual images with the shared readers. Model-agnostic by design.

A baseline does not need to share our architecture, our sampler, or even be a diffusion model.
It writes PNGs to disk in the layout below; this script reads them and produces CC / FC_obs /
FC_unobs / CF1. That keeps the evaluator identical across papers, which is the only way the
numbers end up in one table honestly.

LAYOUT
  <root>/recon/<index>.png          the model's reconstruction of the cohort image, NO edit
  <root>/<attr>/<strength>/<index>.png   the counterfactual with <attr> flipped
  <index> is the CelebA-HQ index from cohort_2048.json, zero-padded to 5 digits.

WHY A RECONSTRUCTION SET IS REQUIRED
  FC asks whether an attribute the edit should not touch stayed put. Measured against the source
  photograph it would also charge the model for its own encode/decode error, which has nothing to
  do with the intervention. Measured against the model's own reconstruction, that error cancels.
  Every model therefore supplies its reconstruction of each cohort image.

  A model with no reconstruction path (e.g. one that only samples) can pass --recon-is-source,
  which scores FC against the photograph instead. That is a DIFFERENT and stricter measurement;
  it must not be put in the same column as reconstruction-based FC.

STRENGTH
  <strength> is a free-form directory name for whatever knob the model exposes:
    diffusion + classifier-free guidance -> the guidance scale, e.g. "g3.0"
    GAN latent editing                   -> the edit magnitude, e.g. "alpha2.5"
    a model with a single deterministic counterfactual -> "fixed"
  The scorer treats it as an opaque label and reports one row per (attr, strength).

THRESHOLDS
  Readers are applied UNTUNED (logit > 0) by default. Tuning thresholds on real photographs was
  measured to HURT accuracy on generated images -- it helped only 15 of 37 attributes and lowered
  the mean from 0.8356 to 0.8317 -- so the untuned operating point is the honest default.
  --target-f1-threshold switches the attribute being intervened on to its F1-optimal threshold,
  which matters because CC counts positive readings and a permissive reader inflates it.
"""
import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn as nn
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from torchvision import models

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
OBS = ["Male", "Young", "Beard", "Bald"]


class Reader:
    def __init__(self, path, thr, dev):
        b = torch.load(path, map_location="cpu")
        m = getattr(models, b.get("arch", "convnext_small"))()
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
        m.load_state_dict({k: v.float() for k, v in b["state_dict"].items()})
        self.net = m.to(dev).eval(); self.size = b["img_size"]; self.thr = thr; self.dev = dev
        self.mean, self.std = MEAN.to(dev), STD.to(dev)

    @torch.no_grad()
    def __call__(self, batch):           # batch: float tensor in [-1,1], N3HW
        x = (batch.to(self.dev) + 1) / 2
        if x.shape[-1] != self.size:
            x = torch.nn.functional.interpolate(x, size=(self.size,) * 2, mode="bilinear",
                                                align_corners=False)
        with torch.autocast(self.dev, dtype=torch.float16, enabled=(self.dev == "cuda")):
            return (self.net((x - self.mean) / self.std).squeeze(1).float() > self.thr).float().cpu()


def load_dir(d, indices, size=256):
    """Read <d>/<index>.png for every cohort index. Missing files are a hard error: a silently
    short cohort would change the denominator and make the numbers incomparable."""
    out, missing = [], []
    for i in indices:
        p = os.path.join(d, f"{i:05d}.png")
        if not os.path.exists(p):
            missing.append(i); continue
        im = Image.open(p).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        out.append(torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1)
    if missing:
        raise SystemExit(f"{len(missing)} images missing from {d} (first: {missing[:5]}). "
                         "Every cohort member must be present.")
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory holding recon/ and <attr>/<strength>/")
    ap.add_argument("--readers", required=True, help="dir of per-attribute .pt readers")
    ap.add_argument("--cohort", default="experiments/hdae/baselines/cohort_2048.json")
    ap.add_argument("--keep", default="experiments/hdae/baselines/reader_keep.json",
                    help="which readers are trustworthy enough to score FC_unobs")
    ap.add_argument("--out", default="baseline_results.json")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target-f1-threshold", action="store_true")
    ap.add_argument("--recon-is-source", action="store_true")
    a = ap.parse_args()

    co = json.load(open(a.cohort)); idx = co["indices"]
    keep = json.load(open(a.keep))
    fc_unobs_attrs = keep["fc_unobs"]
    thr_f1 = keep.get("target_f1_thresholds", {})

    print(f"cohort {len(idx)} subjects | FC_unobs over {len(fc_unobs_attrs)} readers", flush=True)
    recon = load_dir(os.path.join(a.root, "recon"), idx)
    print(f"loaded {len(recon)} reconstructions", flush=True)

    readers = {}
    for n in OBS + fc_unobs_attrs:
        p = os.path.join(a.readers, f"{n}.pt")
        if os.path.exists(p):
            readers[n] = Reader(p, 0.0, a.device)
    print(f"loaded {len(readers)} readers", flush=True)

    def read_all(imgs):
        return {n: torch.cat([r(imgs[i:i + a.bs]) for i in range(0, len(imgs), a.bs)])
                for n, r in readers.items()}

    base = read_all(recon)
    hm = lambda x, z: 0.0 if x + z == 0 else 2 * x * z / (x + z)
    res = {}
    for attr in OBS:
        for sd in sorted(glob.glob(os.path.join(a.root, attr, "*"))):
            if not os.path.isdir(sd): continue
            strength = os.path.basename(sd)
            imgs = load_dir(sd, idx)
            if a.target_f1_threshold and attr in thr_f1:
                readers[attr].thr = thr_f1[attr]
            rd = read_all(imgs)
            readers[attr].thr = 0.0
            want = torch.tensor([1.0 if v < 0 else 0.0 for v in co["factual"][attr]])  # flipped
            CC = float((rd[attr] == want).float().mean())
            fo = {n: float((rd[n] == base[n]).float().mean()) for n in OBS if n != attr and n in rd}
            fu = {n: float((rd[n] == base[n]).float().mean()) for n in fc_unobs_attrs if n in rd}
            FCo, FCu = float(np.mean(list(fo.values()))), float(np.mean(list(fu.values())))
            res[f"{attr}|{strength}"] = dict(
                attr=attr, strength=strength, CC=CC, FC_obs=FCo, FC_unobs=FCu,
                CF1_obs=hm(CC, FCo), CF1_unobs=hm(CC, FCu),
                fc_obs_per=fo, fc_unobs_per=fu, n=len(idx),
                fc_reference="source photo" if a.recon_is_source else "model reconstruction")
            json.dump(res, open(a.out, "w"), indent=2)
            print(f"{attr:7} {strength:10} CC {CC:.4f}  FC_obs {FCo:.4f}  FC_unobs {FCu:.4f}  "
                  f"CF1 {hm(CC, FCu):.4f}", flush=True)
    print(f"\nwrote {a.out} ({len(res)} rows)")


if __name__ == "__main__":
    main()
