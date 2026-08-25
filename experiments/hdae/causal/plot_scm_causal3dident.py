"""Distribution / sampling diagnostics for the Causal3DIdent attribute SCM.

Ancestral-samples the fitted SCM and compares against the real attribute data:
  fig 1  per-attribute marginals (real vs sampled)
  fig 2  conditional structure: E[.|class], and pos_obj_x vs pos_spl per class
  fig 3  counterfactual response curves (do(pos_spl) sweep, do(class))

Companion to train_scm_causal3dident.py; imports the SCM from it by path, so it
still needs nothing from experiments.hdae.

    python experiments/hdae/causal/plot_scm_causal3dident.py
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_scm_causal3dident import SCM, CausalGraph, load_attributes, REPO_ROOT, DEFAULT_CONFIG

DIMS = [("pos_obj", 0, "pos_obj x"), ("pos_obj", 1, "pos_obj y"), ("pos_obj", 2, "pos_obj z"),
        ("rot_obj", 0, "rot_obj alpha"), ("rot_obj", 1, "rot_obj beta"), ("rot_obj", 2, "rot_obj gamma"),
        ("pos_spl", 0, "pos_spl")]
REAL_C, MODEL_C = "#4C72B0", "#DD8452"


def fig_marginals(real, samp, path):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    bins = np.linspace(-1.25, 1.25, 70)
    for ax, (node, d, label) in zip(axes.flat, DIMS):
        r = real[node][:, d].numpy()
        s = samp[node][:, d].cpu().numpy()
        ax.hist(r, bins=bins, density=True, color=REAL_C, alpha=.55, label="real")
        ax.hist(s, bins=bins, density=True, histtype="step", lw=2.0, color=MODEL_C, label="SCM samples")
        oob = float(np.mean((s < -1) | (s > 1)) * 100)
        ax.axvline(-1, color="k", ls=":", lw=.8); ax.axvline(1, color="k", ls=":", lw=.8)
        ax.set_title(f"{label}\nreal sd={r.std():.3f}  model sd={s.std():.3f}  OOB={oob:.1f}%", fontsize=9)
        ax.tick_params(labelsize=8)
    ax = axes.flat[7]
    rc = real["class"].squeeze(-1).numpy(); sc = samp["class"].squeeze(-1).cpu().numpy()
    w = 0.38; xs = np.arange(7)
    ax.bar(xs - w/2, [np.mean(rc == c) for c in xs], w, color=REAL_C, alpha=.75, label="real")
    ax.bar(xs + w/2, [np.mean(sc == c) for c in xs], w, color=MODEL_C, alpha=.9, label="SCM samples")
    ax.axhline(1/7, color="k", ls=":", lw=.8)
    ax.set_title("class (categorical)", fontsize=9); ax.set_xticks(xs); ax.tick_params(labelsize=8)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Causal3DIdent SCM — per-attribute marginals: real vs ancestral samples "
                 "(dotted = data bounds ±1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(path, dpi=110); plt.close(fig)
    print("wrote", path)


def fig_conditional(real, samp, names, path):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    rc = real["class"].squeeze(-1).numpy(); sc = samp["class"].squeeze(-1).cpu().numpy()
    # row 0: E[.|class] real vs model, for each of the 6 object dims + pos_spl
    for ax, (node, d, label) in zip(axes.flat[:4], DIMS[:4]):
        rm = [real[node][:, d].numpy()[rc == c].mean() for c in range(7)]
        sm = [samp[node][:, d].cpu().numpy()[sc == c].mean() for c in range(7)]
        ax.plot(range(7), rm, "o-", color=REAL_C, label="real")
        ax.plot(range(7), sm, "s--", color=MODEL_C, label="SCM samples")
        ax.set_title(f"E[{label} | class]", fontsize=9)
        ax.set_xticks(range(7)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.tick_params(labelsize=8); ax.grid(alpha=.3)
    axes.flat[0].legend(fontsize=8)
    # row 1: pos_obj x vs pos_spl, per class — the sign-flipping edge
    for k, c in enumerate([0, 1, 3, 6]):
        ax = axes.flat[4 + k]
        m_r, m_s = rc == c, sc == c
        ax.scatter(real["pos_spl"][:, 0].numpy()[m_r][:1500], real["pos_obj"][:, 0].numpy()[m_r][:1500],
                   s=3, alpha=.35, color=REAL_C, label="real")
        ax.scatter(samp["pos_spl"][:, 0].cpu().numpy()[m_s][:1500], samp["pos_obj"][:, 0].cpu().numpy()[m_s][:1500],
                   s=3, alpha=.35, color=MODEL_C, label="SCM")
        rr = np.corrcoef(real["pos_spl"][:, 0].numpy()[m_r], real["pos_obj"][:, 0].numpy()[m_r])[0, 1]
        ss = np.corrcoef(samp["pos_spl"][:, 0].cpu().numpy()[m_s], samp["pos_obj"][:, 0].cpu().numpy()[m_s])[0, 1]
        ax.set_title(f"class {c} ({names[c]})\npos_spl -> pos_obj x   real r={rr:+.2f}  model r={ss:+.2f}",
                     fontsize=9)
        ax.set_xlabel("pos_spl", fontsize=8); ax.set_ylabel("pos_obj x", fontsize=8)
        ax.tick_params(labelsize=8)
    axes.flat[4].legend(fontsize=8, markerscale=3)
    fig.suptitle("Causal3DIdent SCM — conditional structure recovered from samples", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(path, dpi=110); plt.close(fig)
    print("wrote", path)


@torch.no_grad()
def fig_counterfactual(scm, real, names, device, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    rc = real["class"].squeeze(-1).numpy()
    modeled = scm.graph.attributes
    grid = np.linspace(-1, 1, 21)

    ax = axes[0]
    for c in range(7):
        s = {k: real[k][rc == c][:1024].to(device) for k in modeled}
        ys = []
        for v in grid:
            cf = scm.counterfactual(s, {"pos_spl": torch.full((1, 1), float(v), device=device)})
            ys.append(cf["pos_obj"][:, 0].mean().item())
        ax.plot(grid, ys, marker=".", label=f"{c} {names[c]}")
    ax.set_title("do(pos_spl = v)  ->  E[pos_obj x]\n(descendant: responds, sign is class-specific)", fontsize=10)
    ax.set_xlabel("intervened pos_spl"); ax.grid(alpha=.3); ax.legend(fontsize=7)

    ax = axes[1]
    for c in range(7):
        s = {k: real[k][rc == c][:1024].to(device) for k in modeled}
        ys = []
        for v in grid:
            cf = scm.counterfactual(s, {"pos_spl": torch.full((1, 1), float(v), device=device)})
            ys.append(cf["rot_obj"][:, 0].mean().item())
        ax.plot(grid, ys, marker=".", label=f"{c} {names[c]}")
    ax.set_title("do(pos_spl = v)  ->  E[rot_obj alpha]\n(NON-descendant: must be flat)", fontsize=10)
    ax.set_xlabel("intervened pos_spl"); ax.grid(alpha=.3)

    ax = axes[2]
    s = {k: real[k][:4096].to(device) for k in modeled}
    obs = [real["rot_obj"][:, 0].numpy()[rc == c].mean() for c in range(7)]
    cfm = []
    for c in range(7):
        cf = scm.counterfactual(s, {"class": torch.full((1, 1), float(c), device=device)})
        cfm.append(cf["rot_obj"][:, 0].mean().item())
    ax.plot(range(7), obs, "o-", color=REAL_C, label="real E[rot_obj a | class]")
    ax.plot(range(7), cfm, "s--", color=MODEL_C, label="do(class=c) -> E[rot_obj a]")
    ax.set_title("do(class = c)  ->  E[rot_obj alpha]", fontsize=10)
    ax.set_xticks(range(7)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    fig.suptitle("Causal3DIdent SCM — interventional response curves", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(path, dpi=110); plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--n", type=int, default=25200)
    ap.add_argument("--outdir", default=os.path.join(REPO_ROOT, "experiments/hdae/outputs/scm"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    graph = CausalGraph(cfg["attributes"], cfg["edges"])
    device = torch.device(args.device)
    ckpt = args.ckpt or os.path.join(REPO_ROOT, cfg["scm_checkpoint"])
    blob = torch.load(ckpt, map_location=device)
    mech = blob.get("mechanism", "gaussian")
    scm = SCM(graph, cfg["nodes"], mechanism=mech, bins=blob.get("bins", 16)).to(device)
    scm.load_state_dict(blob["state_dict"])
    scm.eval()
    print(f"loaded {ckpt}  (mechanism={mech})")

    real = load_attributes(cfg, "testset")
    samp = scm.sample(args.n, device)
    names = cfg["nodes"]["class"]["class_names"]
    os.makedirs(args.outdir, exist_ok=True)

    # numeric summary alongside the plots
    print(f"\n{'attribute':14s} {'real mean':>10s} {'model mean':>11s} {'real sd':>9s} {'model sd':>9s} {'OOB%':>7s}")
    for node, d, label in DIMS:
        r = real[node][:, d].numpy(); s = samp[node][:, d].cpu().numpy()
        oob = float(np.mean((s < -1) | (s > 1)) * 100)
        print(f"{label:14s} {r.mean():10.4f} {s.mean():11.4f} {r.std():9.4f} {s.std():9.4f} {oob:7.2f}")

    fig_marginals(real, samp, os.path.join(args.outdir, f"scm_marginals{args.tag}.png"))
    fig_conditional(real, samp, names, os.path.join(args.outdir, f"scm_conditional{args.tag}.png"))
    fig_counterfactual(scm, real, names, device, os.path.join(args.outdir, f"scm_counterfactual{args.tag}.png"))


if __name__ == "__main__":
    main()
