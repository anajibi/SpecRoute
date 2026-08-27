"""Final convergence report: stitched loss curves, power-law fits, floors, and the table.

Each extension run logs into its OWN output directory, so the full history of a model is two
TensorBoard directories that have to be stitched on the shared global-step axis. The fit is then
done on the whole stitched curve, not on either half, which is the only way the projection and
the outcome are comparable quantities.

    python experiments/hdae/scripts/convergence_report.py
"""
import glob
import json
import os

import numpy as np
from scipy.optimize import curve_fit
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRATCH = "/tmp/claude-1001/-home-exouser/ee943fc6-c5ef-4405-b557-1b557434dfe9/scratchpad"

RUNS = {
    "k=1":  dict(dirs=[f"{REPO}/experiments/hdae/outputs/c3di_k1_final/logs",
                       f"{REPO}/experiments/hdae/outputs/c3di_k1_ext75/logs"],
                 base_step=217936, base_epoch=50, lr="1e-4 (optimiser state; see config header)"),
    "k=11": dict(dirs=[f"{SCRATCH}/tb_k11",
                       f"{REPO}/experiments/hdae/outputs/c3di_k11_ext75/logs"],
                 base_step=192936, base_epoch=50, lr="2e-4"),
}
MODEL = lambda t, a, b, c: a * np.power(t, -b) + c


def load(dirs):
    s, v = [], []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "events.out.tfevents.*"), recursive=True)):
            ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
            tags = [t for t in ea.Tags()["scalars"] if "loss" in t.lower()]
            if not tags:
                continue
            for e in ea.Scalars(tags[0]):
                s.append(e.step); v.append(e.value)
    if not s:
        return np.array([]), np.array([])
    s = np.asarray(s, float); v = np.asarray(v, float)
    o = np.argsort(s); s, v = s[o], v[o]
    keep = np.concatenate([[True], np.diff(s) > 0])      # de-duplicate overlapping resume points
    return s[keep], v[keep]


def binned(s, v, nbins=160):
    edges = np.linspace(s.min(), s.max(), nbins + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, nbins - 1)
    xs, ys = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() > 20:
            xs.append(s[m].mean()); ys.append(np.median(v[m]))
    return np.asarray(xs), np.asarray(ys)


def analyse(name, cfg):
    s, v = load(cfg["dirs"])
    if not len(s):
        return None
    x, y = binned(s, v)
    e = x / cfg["base_step"] * cfg["base_epoch"]          # steps -> epochs on the run's own scale
    popt, _ = curve_fit(MODEL, e, y, p0=[y[0] * 10, 0.5, y.min() * 0.9], maxfev=400000,
                        bounds=([0, .01, 0], [np.inf, 5, y.min()]))
    a, b, c = popt
    resid = y - MODEL(e, *popt)
    rel = lambda t: (a * b * np.power(t, -b - 1)) / MODEL(t, *popt) * 100
    last = float(e.max())
    return dict(name=name, a=a, b=b, c=c, epochs=last, final_loss=float(y[-1]),
                rms=float(resid.std()), rms_pct=float(resid.std() / y.mean() * 100),
                reducible=float((y[-1] - c) / y[-1] * 100), rate_now=float(rel(last)),
                lr=cfg["lr"], e=e, y=y, fit=MODEL(e, *popt),
                rate_at_50=float(rel(50)) if last >= 50 else None)


def main():
    res = {k: analyse(k, c) for k, c in RUNS.items()}
    res = {k: v for k, v in res.items() if v}
    print(f"{'model':6s} {'epochs':>7s} {'final loss':>11s} {'floor c':>10s} {'reducible':>10s} "
          f"{'%/epoch now':>12s} {'rms resid':>10s}  lr")
    for k, r in res.items():
        print(f"{k:6s} {r['epochs']:7.1f} {r['final_loss']:11.6f} {r['c']:10.6f} "
              f"{r['reducible']:9.1f}% {r['rate_now']:11.3f}% {r['rms_pct']:9.2f}%  {r['lr']}")
    for k, r in res.items():
        print(f"\n{k}:  L(e) = {r['a']:.5g} * e^-{r['b']:.4f} + {r['c']:.6g}")
        if r["rate_at_50"]:
            print(f"      improvement per epoch: {r['rate_at_50']:.3f}% at epoch 50 -> "
                  f"{r['rate_now']:.3f}% at epoch {r['epochs']:.0f}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        C = {"k=1": "#C0553B", "k=11": "#2F6FB0"}
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        for k, r in res.items():
            ax[0].plot(r["e"], r["y"], lw=1.1, color=C[k], alpha=.55, label=f"{k} observed")
            ax[0].plot(r["e"], r["fit"], lw=2, color=C[k], ls="--", label=f"{k} fit")
            ax[0].axhline(r["c"], color=C[k], lw=.9, ls=":")
            rate = (r["a"] * r["b"] * np.power(r["e"], -r["b"] - 1)) / MODEL(r["e"], r["a"], r["b"], r["c"]) * 100
            ax[1].plot(r["e"], rate, lw=2, color=C[k], label=k)
        ax[0].axvline(50, color="#888", lw=.9, ls="-.")
        ax[0].text(50.6, ax[0].get_ylim()[1] * .96, "extension begins", fontsize=8, color="#666")
        ax[0].set_xscale("log"); ax[0].set_yscale("log")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss")
        ax[0].set_title("loss curve and power-law fit (dotted = fitted floor)", fontsize=11, fontweight="bold")
        ax[0].legend(frameon=False, fontsize=8); ax[0].grid(alpha=.25, lw=.6)
        for th in (0.5, 0.25):
            ax[1].axhline(th, color="#888", lw=.8, ls=":")
            ax[1].text(ax[1].get_xlim()[1], th, f" {th}%/epoch", fontsize=8, color="#666", va="bottom", ha="right")
        ax[1].set_xscale("log"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("% loss improvement per epoch")
        ax[1].set_title("convergence rate", fontsize=11, fontweight="bold")
        ax[1].legend(frameon=False, fontsize=9); ax[1].grid(alpha=.25, lw=.6)
        for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        p = f"{REPO}/experiments/hdae/outputs/convergence_report.png"
        plt.savefig(p, dpi=150, facecolor="white"); print(f"\nwrote {p}")
    except Exception as ex:
        print(f"\nplot skipped: {ex}")

    out = {k: {kk: vv for kk, vv in r.items() if kk not in ("e", "y", "fit")} for k, r in res.items()}
    p = f"{REPO}/experiments/hdae/outputs/convergence_report.json"
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
