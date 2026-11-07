"""CC / FC_obs / FC_unobs / CF1 for a trained 3DShapes Graph A model.

The metric definitions follow the Causal3DIdent ones exactly (see k11_epoch_curves.py), so the
numbers are comparable across datasets. For a factor k with error `err` (misclassification rate
against that factor's reference value):

    CC_k  = 1 - max(0, err - floor) / (move - floor)      intervened factor and its DESCENDANTS
    FC_k  = 1 - max(0, err - floor) / (base - floor)      everything else

  floor  reconstruction error for k, measured on the same cohort. Credits the model only for
         what conditioning does, not for what autoencoding already costs.
  move   fraction of sources whose reference value for k actually differs from the factual one.
         Normalising by it stops a factor that barely had to move from looking easy.
  base   majority-class error over the cohort's factual values -- the error a constant
         predictor would make, i.e. how much natural variation there is to preserve.

Both are clipped to [0,1]. CC averages over {intervened} u descendants, FC_obs over the
conditioned non-descendants, FC_unobs over wall_hue and orientation. CF1 is the harmonic mean
of CC with each FC, so an arm cannot win by being effective and destructive, or by being
faithful and inert.

EVERYTHING HERE IS CATEGORICAL, so `err` is exact-match misclassification. No MAE: scale wraps
mod 8 and the hues mod 10, and an L1 score would read a legitimate wrap as the worst possible
miss.

TWO PROPAGATION PATHS, REPORTED SEPARATELY. Every counterfactual's attributes are computed
twice, so a failure can be attributed instead of guessed at:

  ref_modeled   the attributes the MODEL WAS CONDITIONED ON -- whatever SCM the pipeline uses to
                abduct, intervene and propagate. This is the model's instruction.
  ref_gt        the attributes of the TRUE counterfactual cell, read back from the grid index
                `counterfactual_index` names. This is the render shown as ground truth.

    scm_disagree   fraction where ref_modeled != ref_gt      -> error on the SCM side
    err_vs_modeled did the model render what it was told?    -> error on the MODEL side
    err_vs_gt      did it produce the true counterfactual?   -> end to end

FOR THIS DATASET THE TWO PATHS COINCIDE BY CONSTRUCTION, so scm_disagree must be exactly 0 and
err_vs_gt must equal err_vs_modeled. Graph A's mechanisms are modular arithmetic with exact
abduction -- there are no learned SCM parameters to be wrong. The columns are computed anyway
rather than assumed, for two reasons: it is a live integrity check on the whole pipeline (a
factor-order mismatch between LAYOUT_COLUMNS and the conditioning order would show up here as a
nonzero scm_disagree, and nowhere else), and the moment a LEARNED SCM is substituted the
attribution becomes real with no change to this script. A nonzero scm_disagree here means a bug,
not a modelling limitation.

PROTOCOL (fixed by the user, 2026-11-06):
  - one random non-factual target per source per factor, so `move` for the intervened factor
    is 1.0 by construction
  - 256 source cells drawn from the 48,000 HELD-OUT cells
  - the full CC-vs-FC picture at three guidance values; no single g is selected
  - the final checkpoint only

The measurement instruments are the six predictors, each at 1.000000 on all 48,000 held-out
cells, so instrument error is not a meaningful term in any of these numbers.

    python able/datasets/shapes3d/eval_cf.py --ckpt <path> --n 256 --guidance 3,6,10
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from able.datasets.shapes3d.cf_images import (load_model, load_predictors,  # noqa: E402
                                              render, score, y_of)
from able.datasets.shapes3d.scm_exact import (CARDINALITY, LAYOUT_COLUMNS,  # noqa: E402
                                              MODELED, counterfactual_index, decode_index,
                                              intervene)

UNOBS = ("wall_hue", "orientation")
DESCENDANTS = {"shape": ("scale", "floor_hue"), "object_hue": ("floor_hue",),
               "scale": ("floor_hue",), "floor_hue": ()}


def clip01(x):
    return float(min(1.0, max(0.0, x)))


def norm(err, floor, denom):
    """The shared normalisation. Returns None where the denominator leaves nothing to measure."""
    if denom - floor <= 1e-9:
        return None
    return clip01(1.0 - max(0.0, err - floor) / (denom - floor))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="able/datasets/shapes3d/configs/shapes3d_a_k11_d256.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h5", default="/home/exouser/able_data/src_3dshapes/3dshapes.h5")
    ap.add_argument("--layout-dir", default="/home/exouser/able_data/shapes3d/graph_a/v1")
    ap.add_argument("--predictors", default="experiments/hdae/outputs/attr_predictors_shapes3d")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--guidance", default="3,6,10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="experiments/hdae/outputs/shapes3d_cf_figs/cf_metrics.json")
    ap.add_argument("--mem-frac", type=float, default=0.0,
                    help="hard cap on this process's GPU share; protects a concurrent training run")
    args = ap.parse_args()

    gs = [float(g) for g in args.guidance.split(",")]
    dev = torch.device("cuda")
    # A cap means an OOM lands on THIS process instead of starving a training run that may have
    # many hours invested in it -- the cheap failure instead of the expensive one.
    if args.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)
    lit, cfg, step = load_model(args.config, args.ckpt, dev)
    specs = lit.model.hdae_conf.encoder.cond_specs
    sampler = lit.conf._make_diffusion_conf(args.T).make_sampler()
    preds = load_predictors(args.predictors, dev)

    heldout = np.load(f"{args.layout_dir}/heldout_cells.npy").astype(np.int64)
    rng = np.random.RandomState(args.seed)
    cells = np.sort(rng.choice(heldout, args.n, replace=False))
    Hf, Hw, Ho, C, S, R = decode_index(cells)
    fac = {"shape": S, "object_hue": Ho, "scale": C, "floor_hue": Hf}
    factual = np.stack([S, Ho, C, Hf, Hw, R], axis=1)

    print(f"cohort {args.n} held-out cells | guidance {gs} | T={args.T} | step {step}", flush=True)
    with h5py.File(args.h5, "r") as f:
        images = f["images"][:]

    def batches(n):
        for i in range(0, n, args.batch):
            yield i, min(i + args.batch, n)

    def grab(idx):
        a = np.stack([images[int(g)] for g in np.atleast_1d(idx)])
        return torch.from_numpy(a).permute(0, 3, 1, 2).float().div_(127.5).sub_(1).to(dev)

    # --- invert every source once; x_T and the semantic latents are shared by every render.
    zs_all, xT_all, recon_pred = [], [], []
    for a, b in batches(args.n):
        x = grab(cells[a:b])
        y = y_of({k: v[a:b] for k, v in fac.items()}, specs, dev)
        with torch.no_grad():
            zs = [z.clone() for z in lit.ema_model.encode(x)]
            cond = lit.ema_model.make_cond(zs, y)
            x_T = sampler.ddim_reverse_sample_loop(
                lit.ema_model, x, model_kwargs={"cond": cond})["sample"]
            rec = render(lit, sampler, x_T, cond, 1.0)
        zs_all.append([z.cpu() for z in zs]); xT_all.append(x_T.cpu())
        recon_pred.append(score(rec, preds, dev))
    recon_pred = np.concatenate(recon_pred)

    # floor: what the autoencoding round trip already costs, per factor
    floor = {n: float((recon_pred[:, j] != factual[:, j]).mean())
             for j, n in enumerate(LAYOUT_COLUMNS)}
    # base: the error a constant (majority-class) predictor would make on this cohort
    base = {}
    for j, n in enumerate(LAYOUT_COLUMNS):
        cnt = np.bincount(factual[:, j], minlength=CARDINALITY[n])
        base[n] = float(1.0 - cnt.max() / len(factual))
    print(f"recon floor  {({k: round(v, 4) for k, v in floor.items()})}")
    print(f"variation base {({k: round(v, 4) for k, v in base.items()})}\n", flush=True)

    # one random non-factual target per source per factor, fixed across guidance values
    targets = {}
    for var in MODELED:
        K = CARDINALITY[var]
        targets[var] = (fac[var] + 1 + rng.randint(0, K - 1, args.n)) % K

    out = {"checkpoint": os.path.basename(args.ckpt), "global_step": step, "n": args.n,
           "T": args.T, "seed": args.seed, "protocol": "one random non-factual target per source",
           "floor": floor, "base": base, "guidance": {}}

    for g in gs:
        per = {}
        for var in MODELED:
            val = targets[var]
            # PATH 1 -- the SCM the model is conditioned on.
            S2, Ho2, C2, Hf2 = intervene(S, Ho, C, Hf, var, val)
            cf_fac = {"shape": S2, "object_hue": Ho2, "scale": C2, "floor_hue": Hf2}
            ref = np.stack([S2, Ho2, C2, Hf2, Hw, R], axis=1)   # exogenous preserved
            # PATH 2 -- the ground-truth structural equations, read off the true CF cell.
            gt_cells = counterfactual_index(S, Ho, C, Hf, Hw, R, var, val)
            gHf, gHw, gHo, gC, gS, gR = decode_index(gt_cells)
            ref_gt = np.stack([gS, gHo, gC, gHf, gHw, gR], axis=1)
            scm_disagree = {n: float((ref[:, j] != ref_gt[:, j]).mean())
                            for j, n in enumerate(LAYOUT_COLUMNS)}
            pred = []
            for i, (a, b) in enumerate(batches(args.n)):
                zs = [z.to(dev) for z in zs_all[i]]
                yc = y_of({k: v[a:b] for k, v in cf_fac.items()}, specs, dev)
                with torch.no_grad():
                    img = render(lit, sampler, xT_all[i].to(dev),
                                 lit.ema_model.make_cond(zs, yc), g)
                pred.append(score(img, preds, dev))
            pred = np.concatenate(pred)

            tgt_set = (var,) + DESCENDANTS[var]
            cc, fc, detail = {}, {}, {}
            for j, n in enumerate(LAYOUT_COLUMNS):
                err = float((pred[:, j] != ref[:, j]).mean())          # vs what it was told
                err_gt = float((pred[:, j] != ref_gt[:, j]).mean())    # vs the true CF
                move = float((ref[:, j] != factual[:, j]).mean())
                detail[n] = {"err_vs_modeled": round(err, 5), "err_vs_gt": round(err_gt, 5),
                             "accuracy_vs_modeled": round(1 - err, 5),
                             "accuracy_vs_gt": round(1 - err_gt, 5),
                             "scm_disagree": round(scm_disagree[n], 5),
                             "move": round(move, 5)}
                if n in tgt_set:
                    v = norm(err, floor[n], move)
                    if v is not None:
                        cc[n] = v
                else:
                    v = norm(err, floor[n], base[n])
                    if v is not None:
                        fc[n] = v
                detail[n]["score"] = None if v is None else round(v, 5)
            CC = float(np.mean(list(cc.values()))) if cc else float("nan")
            FO = float(np.mean([v for k, v in fc.items() if k in MODELED])) if \
                any(k in MODELED for k in fc) else float("nan")
            FU = float(np.mean([v for k, v in fc.items() if k in UNOBS])) if \
                any(k in UNOBS for k in fc) else float("nan")
            hm = lambda x, z: 0.0 if (np.isnan(x) or np.isnan(z) or x + z == 0) else 2 * x * z / (x + z)
            tot_scm = float(np.mean([scm_disagree[n] for n in LAYOUT_COLUMNS]))
            tot_mdl = float(np.mean([detail[n]["err_vs_modeled"] for n in LAYOUT_COLUMNS]))
            tot_e2e = float(np.mean([detail[n]["err_vs_gt"] for n in LAYOUT_COLUMNS]))
            per[var] = {"CC": CC, "FC_obs": FO, "FC_unobs": FU,
                        "attribution": {"scm_side_error": round(tot_scm, 6),
                                        "model_side_error": round(tot_mdl, 6),
                                        "end_to_end_error": round(tot_e2e, 6),
                                        "paths_identical": bool(tot_scm == 0.0)},
                        "CF1_obs": hm(CC, FO), "CF1_unobs": hm(CC, FU),
                        "cc_terms": {k: round(v, 5) for k, v in cc.items()},
                        "fc_terms": {k: round(v, 5) for k, v in fc.items()},
                        "per_factor": detail}
            print(f"g={g:<5g} do({var}):  CC {CC:.4f}  FC_obs {FO:.4f}  FC_unobs {FU:.4f}  "
                  f"CF1_obs {per[var]['CF1_obs']:.4f}   | attribution: scm {tot_scm:.4f}  "
                  f"model {tot_mdl:.4f}  e2e {tot_e2e:.4f}", flush=True)
        agg = {k: float(np.mean([per[v][k] for v in MODELED]))
               for k in ("CC", "FC_obs", "FC_unobs", "CF1_obs", "CF1_unobs")}
        out["guidance"][str(g)] = {"mean": agg, "per_intervention": per}
        print(f"g={g:<5g} MEAN:     CC {agg['CC']:.4f}  FC_obs {agg['FC_obs']:.4f}  "
              f"FC_unobs {agg['FC_unobs']:.4f}  CF1_obs {agg['CF1_obs']:.4f}  "
              f"CF1_unobs {agg['CF1_unobs']:.4f}\n", flush=True)

    path = args.out if not args.tag else args.out.replace(".json", f"_{args.tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")
    print(f"\n{'g':>5} {'CC':>8} {'FC_obs':>8} {'FC_unobs':>9} {'CF1_obs':>8} {'CF1_unobs':>10}")
    for g in gs:
        m = out["guidance"][str(g)]["mean"]
        print(f"{g:>5g} {m['CC']:8.4f} {m['FC_obs']:8.4f} {m['FC_unobs']:9.4f} "
              f"{m['CF1_obs']:8.4f} {m['CF1_unobs']:10.4f}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
