"""Fit the training-loss curve and project when it plateaus.

The stopping criterion for the depth runs is a TRAINING LOSS plateau, so the useful thing to
do before spending a GPU-day is to ask what the curve already on disk implies. Diffusion
training loss follows a power law with an irreducible floor,

    L(s) = a * s^(-b) + c

where c is the noise floor the model can never beat. Fitting that on the observed range and
solving for the epoch at which the per-epoch relative improvement drops below a threshold
gives a projection -- not a promise, because the fit is an extrapolation, but a far better
starting point than doubling the epoch count and hoping.

Reported for several thresholds so the sensitivity is visible rather than hidden in one number.

    python experiments/hdae/scripts/loss_convergence.py
"""
import argparse
import glob
import os

import numpy as np
from scipy.optimize import curve_fit
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = {
    "k=1":  ("experiments/hdae/outputs/c3di_k1_final/logs/version_0", 217936, 50),
    "k=11": ("/tmp/claude-1001/-home-exouser/ee943fc6-c5ef-4405-b557-1b557434dfe9/scratchpad/tb_k11", 192936, 50),
}


def load(d):
    steps, vals = [], []
    for f in sorted(glob.glob(os.path.join(d, "events.out.tfevents.*"))):
        ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
        tags = [t for t in ea.Tags()["scalars"] if "loss" in t.lower()]
        if not tags:
            continue
        for e in ea.Scalars(tags[0]):
            steps.append(e.step); vals.append(e.value)
    s = np.asarray(steps, dtype=np.float64); v = np.asarray(vals, dtype=np.float64)
    o = np.argsort(s)
    return s[o], v[o]


def binned(s, v, nbins=120):
    """Median within equal-width step bins -- the raw curve is far too noisy to fit directly."""
    edges = np.linspace(s.min(), s.max(), nbins + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, nbins - 1)
    xs, ys = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() > 20:
            xs.append(s[m].mean()); ys.append(np.median(v[m]))
    return np.asarray(xs), np.asarray(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=float, nargs="+", default=[2.0, 1.0, 0.5, 0.25],
                    help="percent relative loss improvement per epoch that counts as 'still moving'")
    args = ap.parse_args()

    for name, (d, final_step, final_epoch) in RUNS.items():
        s, v = load(d)
        if not len(s):
            print(f"{name}: no scalars found in {d}"); continue
        x, y = binned(s, v)
        # x is the optimiser's sample counter; convert to epochs on the run's own scale
        e = x / x.max() * final_epoch
        f = lambda t, a, b, c: a * np.power(t, -b) + c
        p0 = [y[0] * 10, 0.5, y.min() * 0.9]
        try:
            popt, _ = curve_fit(f, e, y, p0=p0, maxfev=200000,
                                bounds=([0, 0.01, 0], [np.inf, 5, y.min()]))
        except Exception as ex:
            print(f"{name}: fit failed ({ex})"); continue
        a, b, c = popt
        resid = y - f(e, *popt)
        print(f"\n=== {name}   {len(s):,} points, {final_epoch} epochs, final loss {y[-1]:.6f}")
        print(f"  fit  L(e) = {a:.5g} * e^-{b:.4f} + {c:.6g}     rms residual {resid.std():.2e} "
              f"({resid.std()/y.mean()*100:.2f}% of mean)")
        print(f"  irreducible floor c = {c:.6g}   ->  {(y[-1]-c)/y[-1]*100:.1f}% of the current loss "
              f"is still reducible")
        # per-epoch relative improvement at epoch e:  -L'(e)/L(e)
        rel = lambda t: (a * b * np.power(t, -b - 1)) / f(t, *popt) * 100
        print(f"  per-epoch improvement now (epoch {final_epoch}): {rel(final_epoch):.3f}%")
        for th in args.thresholds:
            lo, hi = final_epoch, 100000.0
            if rel(final_epoch) < th:
                print(f"    already below {th}% / epoch")
                continue
            for _ in range(200):
                mid = (lo + hi) / 2
                if rel(mid) > th: lo = mid
                else: hi = mid
            hrs = (hi - final_epoch) * (28.0 / 50)
            print(f"    drops below {th:4.2f}% / epoch at epoch {hi:8.0f}"
                  + (f"   (+{hi-final_epoch:.0f} epochs, ~{hrs:.0f} GPU-h)" if hi < 5000 else "   (not reachable in practice)"))


if __name__ == "__main__":
    main()
