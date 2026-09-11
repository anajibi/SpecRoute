"""How much does each semantic tap know about each attribute?

The depth result on MorphoMNIST inverts Causal3DIdent: k=1 beats k=11, and the whole advantage
sits in the two GLOBAL photometric attributes (intensity -0.0399, hue -0.1201 for k=11) while
the two spatial ones are a wash (+0.0005). The proposed mechanism: a tap ladder encodes the
SOURCE image at every level and the intervention never touches those taps, so each one keeps
asserting the original attribute. A global property like colour is present at every location and
every scale, so k=11 carries eleven redundant copies of it fighting the edit -- whereas a
spatial property concentrates in the taps whose scale matches it.

That predicts a specific, falsifiable pattern:
  global attrs (hue, intensity)  -> decodable from EVERY tap, roughly uniformly
  spatial attrs (digit, thickness) -> concentrated in some taps, weak in others

This fits a linear probe per (tap, attribute) on frozen encoder outputs. Reported as R^2 for
continuous attributes and accuracy for the two categorical ones, on a held-out split. Ridge
closed-form rather than SGD: the probe must measure what is linearly present, and an
under-converged optimiser would understate it.
"""
import argparse, json, os, sys
import h5py
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from experiments.hdae.hdae.config_io import load_hdae_config            # noqa: E402
from experiments.hdae.hdae.lit_module import HDAELitModule              # noqa: E402

MOD = ["digit", "thickness", "intensity", "hue"]
UNOB = ["rotation", "scale", "translate_x", "translate_y",
        "bg_freq", "bg_phase", "bg_amplitude", "texture_amplitude"]
CAT = {"digit", "hue"}
NATURE = {"digit": "spatial", "thickness": "spatial", "intensity": "global", "hue": "global"}



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

def ridge_fit(Z, y, lam=1.0):
    """Closed-form ridge with an intercept; Z is (n, d)."""
    Zc = np.hstack([Z, np.ones((len(Z), 1))])
    d = Zc.shape[1]
    R = lam * np.eye(d); R[-1, -1] = 0.0
    return np.linalg.solve(Zc.T @ Zc + R, Zc.T @ y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--h5", default=os.path.join(REPO, "experiments/hdae/data/morphomnist/morphomnist_70k_v2.h5"))
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(REPO, "experiments/hdae/outputs/tap_probe.json"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)

    with h5py.File(a.h5, "r") as h:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in h["attribute_names"][:]]
        A_all = h["attrs"][:].astype(np.float64)
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(A_all), size=min(a.n, len(A_all)), replace=False))
        X = torch.from_numpy(h["images"][idx].astype(np.float32) / 255.).permute(0, 3, 1, 2) * 2 - 1
    COL = {k: names.index(k) for k in names}
    A = A_all[idx]
    ntr = int(0.8 * len(idx))

    def target(k):
        v = A[:, COL[k]]
        if k in CAT:
            return (np.round(v) if k == "digit" else np.round(v * 10 - 0.5)).astype(int), True
        return v, False

    out = {}
    for cfg_p, ck, lab in zip(a.configs, a.ckpts, a.labels):
        cfg = load_hdae_config(os.path.join(REPO, cfg_p), require_data=False)
        mod = load_hdae_module(HDAELitModule, os.path.join(REPO, ck), cfg.train_conf).to(dev).eval()
        net = mod.ema_model.eval()
        taps = None
        with torch.no_grad():
            for i in range(0, len(X), a.bs):
                zs = net.encode(X[i:i + a.bs].to(dev))
                zs = [z.float().cpu().numpy() for z in zs]
                if taps is None:
                    taps = [[] for _ in zs]
                for j, z in enumerate(zs):
                    taps[j].append(z)
        taps = [np.concatenate(t) for t in taps]
        del mod, net
        torch.cuda.empty_cache()

        res = {}
        for ti, Z in enumerate(taps):
            mu, sd = Z[:ntr].mean(0), Z[:ntr].std(0) + 1e-6
            Zs = (Z - mu) / sd
            row = {}
            for k in MOD + UNOB:
                y, is_cat = target(k)
                if is_cat:
                    Y = np.eye(10)[y]                      # one-vs-rest ridge, argmax read-out
                    W = ridge_fit(Zs[:ntr], Y[:ntr], a.lam)
                    pred = (np.hstack([Zs[ntr:], np.ones((len(Zs) - ntr, 1))]) @ W).argmax(1)
                    row[k] = float((pred == y[ntr:]).mean())
                else:
                    ym, ys = y[:ntr].mean(), y[:ntr].std() + 1e-12
                    w = ridge_fit(Zs[:ntr], ((y - ym) / ys)[:ntr, None], a.lam)
                    p = (np.hstack([Zs[ntr:], np.ones((len(Zs) - ntr, 1))]) @ w).ravel()
                    t = (y[ntr:] - ym) / ys
                    row[k] = float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())
            res[f"tap{ti}"] = row
        out[lab] = dict(n_taps=len(taps), dims=[int(z.shape[1]) for z in taps], per_tap=res)
        print(f"\n=== {lab}: {len(taps)} taps, {out[lab]['dims'][0]} dims each "
              f"({len(idx)} images, {len(idx)-ntr} held out) ===")
        print(f"{'tap':6s} " + "".join(f"{k[:9]:>10s}" for k in MOD)
              + f"{'| unobs mean':>14s}")
        for ti in range(len(taps)):
            r = res[f"tap{ti}"]
            print(f"{'tap'+str(ti):6s} " + "".join(f"{r[k]:10.4f}" for k in MOD)
                  + f"{np.mean([r[k] for k in UNOB]):14.4f}")
        print(f"{'SPREAD':6s} " + "".join(
            f"{max(res[f'tap{t}'][k] for t in range(len(taps))) - min(res[f'tap{t}'][k] for t in range(len(taps))):10.4f}"
            for k in MOD))
        print("       " + "".join(f"{NATURE[k]:>10s}" for k in MOD))
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
