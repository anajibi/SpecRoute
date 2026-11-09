"""Score a multi-head CelebA reader on the TEST partition and apply the admission gate.

The trainer reports mean balanced accuracy on its own selection slice, which is a slice of
partition 0 -- the training partition. That is not a held-out number. This scores partition 2,
which no reader has seen, and reports the full set the protocol's gate needs: balanced
accuracy, F1, precision, recall, base rate and a bootstrap CI.

Accuracy is deliberately not the headline. CelebA's rare attributes make it meaningless: a
reader can call almost everything positive, score .879 balanced, and have F1 .033.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from experiments.hdae.data.celeba_hq import CelebAHQPacked           # noqa: E402

GATE = {"bal": 0.80, "f1": 0.55, "ci_lo": 0.75}


def boot_ci(y, p, n=1000, seed=7):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        yi, pi = y[i], p[i]
        if yi.sum() == 0 or yi.sum() == len(yi):
            continue
        tp = ((pi == 1) & (yi == 1)).sum(); fp = ((pi == 1) & (yi == 0)).sum()
        fn = ((pi == 0) & (yi == 1)).sum(); tn = ((pi == 0) & (yi == 0)).sum()
        vals.append(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (None, None)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lmdb", default="experiments/hdae/data/packed/celebahq_256.lmdb")
    ap.add_argument("--npz", default="experiments/hdae/data/packed/celebahq_256_attrs.npz")
    ap.add_argument("--partition", type=int, default=2)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(os.path.join(REPO, a.ckpt), map_location="cpu", weights_only=False)
    attrs, size, arch = ck["attrs"], ck["img_size"], ck["arch"]
    m = getattr(models, arch)(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(attrs))
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in ck["state_dict"].items()}
    m.load_state_dict(sd); m = m.to(dev).eval()

    ds = CelebAHQPacked(os.path.join(REPO, a.lmdb), os.path.join(REPO, a.npz))
    names = list(ds.attribute_names)
    cols = [names.index(x) for x in attrs]
    rows = np.where(ds.partitions == a.partition)[0]
    print(f"{arch} @ {size}px   {len(attrs)} heads   partition {a.partition}: {len(rows)} images")

    P, Y = [], []
    for s in range(0, len(rows), a.bs):
        idx = rows[s:s + a.bs]
        x = torch.stack([(ds[int(j)]["img"] + 1) / 2 for j in idx])
        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
        P.append((m(x.to(dev)) > 0).float().cpu().numpy())
        Y.append((ds.attrs[idx][:, cols] == 1).astype(np.float32))
    P, Y = np.concatenate(P), np.concatenate(Y)

    res, passed = {}, 0
    for j, n in enumerate(attrs):
        y, p = Y[:, j].astype(int), P[:, j].astype(int)
        tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum()
        fn = ((p == 0) & (y == 1)).sum(); tn = ((p == 0) & (y == 0)).sum()
        rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        bal = 0.5 * (rec + spec)
        lo, hi = boot_ci(y, p)
        ok = bool(bal >= GATE["bal"] and f1 >= GATE["f1"] and (lo or 0) >= GATE["ci_lo"])
        passed += ok
        res[n] = {"bal": round(float(bal), 4), "f1": round(float(f1), 4),
                  "prec": round(float(prec), 4), "rec": round(float(rec), 4),
                  "acc": round(float((tp + tn) / len(y)), 4),
                  "base": round(float(y.mean()), 4), "n_pos": int(y.sum()),
                  "ci95_bal": [None if lo is None else round(lo, 4),
                               None if hi is None else round(hi, 4)],
                  "admitted": ok}
    order = sorted(res, key=lambda k: -res[k]["f1"])
    print(f"\n{'attribute':22s} {'bal':>7s} {'F1':>7s} {'prec':>6s} {'rec':>6s} {'base':>6s} {'CI lo':>7s}  gate")
    for n in order:
        r = res[n]
        print(f"{n:22s} {r['bal']:7.4f} {r['f1']:7.4f} {r['prec']:6.3f} {r['rec']:6.3f} "
              f"{r['base']:6.3f} {str(r['ci95_bal'][0]):>7s}  {'PASS' if r['admitted'] else 'fail'}")
    mb = float(np.mean([res[n]['bal'] for n in res])); mf = float(np.mean([res[n]['f1'] for n in res]))
    print(f"\nmean balanced {mb:.4f}   mean F1 {mf:.4f}   admitted {passed}/{len(attrs)}")
    rec_out = {"ckpt": a.ckpt, "arch": arch, "img_size": size, "partition": a.partition,
               "n": len(rows), "gate": GATE, "mean_bal": round(mb, 4), "mean_f1": round(mf, 4),
               "n_admitted": passed, "attributes": res}
    if a.out:
        json.dump(rec_out, open(a.out, "w"), indent=1); print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
