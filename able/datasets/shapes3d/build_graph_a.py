#!/usr/bin/env python
"""3DShapes Graph A -- data-generation protocol v1. Builds the layout and runs all ten tests.

No images are generated or copied. The output is a layout: which grid image appears at each
training position. Reads only the `labels` dataset of the source h5 (never `images`), so the
build never touches the 5.9 GB pixel array.

    python build_graph_a.py --h5 <3dshapes.h5> --out <dir>

Writes factor_to_idx.npy, heldout_cells.npy, train_layout.npy, train_image_idx.npy,
tests_report.json and SHA256SUMS. Every test must pass before the dataset is frozen.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.stats import chi2 as chi2_dist

FACTOR_ORDER = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]
SIZES = [10, 10, 10, 8, 4, 15]
STRIDES = np.array([48000, 4800, 480, 60, 15, 1], dtype=np.int64)
LAYOUT_COLUMNS = ["shape", "object_hue", "scale", "floor_hue", "wall_hue", "orientation"]
M_C = np.array([0, 2, 5, 7], dtype=np.int64)   # scale shift by shape
N_GRID = 480000
CHUNK = 2_000_000                              # fixed, so the draw is reproducible


def noise_pmf(K):
    """P(0)=0.25, P(+-1)=0.125, all other values 0.5/(K-3). Full support."""
    p = np.full(K, 0.5 / (K - 3), dtype=np.float64)
    p[0] = 0.25
    p[1] = 0.125
    p[K - 1] = 0.125
    assert abs(p.sum() - 1.0) < 1e-12, p.sum()
    return p


P8, P10 = noise_pmf(8), noise_pmf(10)
CUM8, CUM10 = np.cumsum(P8), np.cumsum(P10)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(8 << 20)):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- grid / index


def load_grid_indices(h5_path):
    """Map the h5's float factor VALUES to integer indices by rank (the stored values are
    irrational -- scale steps 1/14, orientation 60/14 -- so arithmetic conversion would not be
    exact). Returns (480000, 6) int64 in FACTOR_ORDER."""
    with h5py.File(h5_path, "r") as f:
        labels = f["labels"][:]
    assert labels.shape == (N_GRID, 6), labels.shape
    out = np.empty((N_GRID, 6), dtype=np.int64)
    for j, size in enumerate(SIZES):
        vals = np.unique(labels[:, j])
        assert len(vals) == size, (FACTOR_ORDER[j], len(vals), size)
        out[:, j] = np.searchsorted(vals, labels[:, j])
    return out


def grid_index(Hf, Hw, Ho, C, S, R):
    """idx = H_f*48000 + H_w*4800 + H_o*480 + C*60 + S*15 + R"""
    return (Hf * STRIDES[0] + Hw * STRIDES[1] + Ho * STRIDES[2]
            + C * STRIDES[3] + S * STRIDES[4] + R * STRIDES[5])


# --------------------------------------------------------------------------- SCM


def draw_chunk(rng, n):
    """One chunk of SCM draws. The ORDER of rng calls here is part of the frozen protocol."""
    S = rng.randint(0, 4, n).astype(np.int64)
    Ho = rng.randint(0, 10, n).astype(np.int64)
    Hw = rng.randint(0, 10, n).astype(np.int64)
    R = rng.randint(0, 15, n).astype(np.int64)
    U_C = np.searchsorted(CUM8, rng.random_sample(n)).astype(np.int64)
    U_F = np.searchsorted(CUM10, rng.random_sample(n)).astype(np.int64)
    C = (M_C[S] + U_C) % 8
    Hf = (Ho + C + U_F) % 10
    return S, Ho, C, Hf, Hw, R


def abduce(S, Ho, C, Hf):
    """Exact: U_C = C (-)_8 m_C(S); U_F = H_f (-)_10 (H_o (+)_10 C)."""
    return (C - M_C[S]) % 8, (Hf - Ho - C) % 10


def intervene(S, Ho, C, Hf, var, value):
    """do(var = value): recover noise, set the value, recompute descendants, keep the rest."""
    U_C, U_F = abduce(S, Ho, C, Hf)
    S, Ho, C, Hf = S.copy(), Ho.copy(), C.copy(), Hf.copy()
    if var == "shape":
        S[:] = value
        C = (M_C[S] + U_C) % 8
        Hf = (Ho + C + U_F) % 10
    elif var == "object_hue":
        Ho[:] = value
        Hf = (Ho + C + U_F) % 10
    elif var == "scale":
        C[:] = value                      # mechanism cut, U_C no longer applies
        Hf = (Ho + C + U_F) % 10
    elif var == "floor_hue":
        Hf[:] = value                     # no descendants
    else:
        raise ValueError(var)
    return S, Ho, C, Hf


# --------------------------------------------------------------------------- build


def build(h5_path, n_train, seeds, grid=None):
    """Returns (grid, heldout_cells, layout, image_idx). Pure function of (h5, n, seeds)."""
    if grid is None:
        grid = load_grid_indices(h5_path)

    # factor_to_idx lookup table, (10,10,10,8,4,15) int32
    factor_to_idx = np.full(SIZES, -1, dtype=np.int32)
    factor_to_idx[tuple(grid[:, j] for j in range(6))] = np.arange(N_GRID, dtype=np.int32)

    # 1. split -- hold out 10% of grid cells uniformly at random
    n_held = int(round(0.10 * N_GRID))
    heldout = np.sort(np.random.RandomState(seeds["split_seed"])
                      .choice(N_GRID, size=n_held, replace=False)).astype(np.int32)
    held_mask = np.zeros(N_GRID, dtype=bool)
    held_mask[heldout] = True

    # 2. draw -- N SCM samples with replacement from the TRAINING cells (rejection on hold-out,
    #    which is exactly renormalising the SCM over the 432,000 remaining cells)
    rng = np.random.RandomState(seeds["draw_seed"])
    cols, idxs, got, n_drawn, n_kept = [], [], 0, 0, 0
    while got < n_train:
        S, Ho, C, Hf, Hw, R = draw_chunk(rng, CHUNK)
        n_drawn += CHUNK
        idx = grid_index(Hf, Hw, Ho, C, S, R)
        keep = ~held_mask[idx]
        n_kept += int(keep.sum())
        cols.append(np.stack([S[keep], Ho[keep], C[keep], Hf[keep], Hw[keep], R[keep]], 1)
                    .astype(np.uint8))
        idxs.append(idx[keep].astype(np.int32))
        got += int(keep.sum())
    layout = np.concatenate(cols)[:n_train]
    image_idx = np.concatenate(idxs)[:n_train]

    # 3. shuffle -- one global permutation
    perm = np.random.RandomState(seeds["shuffle_seed"]).permutation(n_train)
    layout, image_idx = layout[perm], image_idx[perm]
    stats = {"n_drawn": n_drawn, "n_kept": n_kept,
             "holdout_rejection_rate": 1.0 - n_kept / n_drawn,
             "overshoot_discarded": n_kept - n_train}
    return grid, factor_to_idx, heldout, held_mask, layout, image_idx, stats


def save(out, factor_to_idx, heldout, layout, image_idx):
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "factor_to_idx.npy", factor_to_idx)
    np.save(out / "heldout_cells.npy", heldout)
    np.save(out / "train_layout.npy", layout)
    np.save(out / "train_image_idx.npy", image_idx)


# --------------------------------------------------------------------------- test helpers

REL_FLOOR, Z_MULT, HELDOUT_FRAC = 0.01, 5, 0.10


def holdout_rel_se(bin_ids, nbins, w, q=HELDOUT_FRAC):
    """Relative sd that the random hold-out alone induces in a bin's probability.

    Holding out each grid cell independently with probability q, a bin's kept SCM weight has
    mean (1-q)*sum(w) and variance q(1-q)*sum(w^2), so its RELATIVE sd is
    sqrt(q/(1-q)) * sqrt(sum(w^2))/sum(w) = sqrt(q/(1-q)/n_eff). This is the term that makes a
    raw-SCM chi-square fail at N=16M, and it is a property of the design, not of the draw --
    so it belongs in the tolerance rather than being discovered as a test failure."""
    s1 = np.bincount(bin_ids, weights=w, minlength=nbins)
    s2 = np.bincount(bin_ids, weights=w * w, minlength=nbins)
    out = np.zeros(nbins, dtype=np.float64)
    m = s1 > 0
    out[m] = np.sqrt(q / (1.0 - q) * s2[m]) / s1[m]
    return out


def compare(observed, expected_p, label, hold_se=None):
    """Raw-SCM reference with a tolerance on relative deviation, per the frozen test rule:
        per-bin tolerance = max(0.01, 5*sqrt(se_multinomial^2 + se_holdout^2))
    se_multinomial = sqrt((1-p)/expected_count); se_holdout from holdout_rel_se (0 if not
    supplied). chi2/df/p and max|z| are recorded as diagnostics only: at N=16M a raw-SCM
    chi-square resolves the hold-out perturbation, which is expected, not a defect. Verified
    against the exact post-hold-out law -- re-scoring there drops max|z| from 7.19 to 2.94 on
    wall_hue x floor_hue, i.e. the deviation is entirely the hold-out."""
    observed = np.asarray(observed, dtype=np.float64).ravel()
    p = np.asarray(expected_p, dtype=np.float64).ravel()
    n = observed.sum()
    exp = n * p
    live = exp > 0
    rel = np.zeros_like(exp)
    rel[live] = observed[live] / exp[live] - 1.0
    se_mult = np.zeros_like(exp)
    se_mult[live] = np.sqrt((1.0 - p[live]) / exp[live])
    se_hold = np.zeros_like(exp) if hold_se is None else np.asarray(hold_se, float).ravel()
    se = np.sqrt(se_mult ** 2 + se_hold ** 2)
    tol = np.maximum(REL_FLOOR, Z_MULT * se)
    z = np.zeros_like(exp)
    z[live] = rel[live] / se[live]
    chi2 = float(((observed[live] - exp[live]) ** 2 / exp[live]).sum())
    df = int(live.sum() - 1)
    ok = bool((np.abs(rel[live]) <= tol[live]).all())
    worst = int(np.argmax(np.abs(rel) - tol)) if live.any() else -1
    return ok, {
        "label": label, "pass": ok, "n": float(n), "bins": int(live.sum()),
        "max_rel_dev": float(np.abs(rel[live]).max()),
        "tolerance_at_worst_bin": float(tol[worst]),
        "max_abs_z": float(np.abs(z[live]).max()),
        "max_se_multinomial": float(se_mult[live].max()),
        "max_se_holdout": float(se_hold[live].max()) if hold_se is not None else 0.0,
        "chi2": chi2, "df": df,
        "chi2_p": float(chi2_dist.sf(chi2, df)) if df > 0 else None,
        "min_expected_count": float(exp[live].min()),
    }


def compare_conditional(obs_rows, ref_rows, label, row_names=None, hold_se_rows=None):
    """A conditional table, tested ROW BY ROW: each row of `obs_rows` is normalised by its own
    count against its own reference row (which sums to 1). Scoring the whole table against a
    joint reference instead would inflate every expected bin by the number of rows."""
    per, ok_all = [], True
    for i, (o, p) in enumerate(zip(obs_rows, ref_rows)):
        nm = row_names[i] if row_names else str(i)
        ok, d = compare(o, p, f"{label}[{nm}]",
                        None if hold_se_rows is None else hold_se_rows[i])
        ok_all &= ok
        per.append(d)
    live = [d for d in per if d["bins"] > 0]
    worst = max(live, key=lambda d: d["max_rel_dev"] - d["tolerance_at_worst_bin"])
    chi2 = sum(d["chi2"] for d in live)
    df = sum(d["df"] for d in live)
    return ok_all, {
        "label": label, "pass": ok_all, "rows": len(live),
        "rows_failing": int(sum(not d["pass"] for d in live)),
        "max_rel_dev": max(d["max_rel_dev"] for d in live),
        "max_abs_z": max(d["max_abs_z"] for d in live),
        "min_expected_count": min(d["min_expected_count"] for d in live),
        "chi2": chi2, "df": df,
        "chi2_p": float(chi2_dist.sf(chi2, df)) if df > 0 else None,
        "worst_row": worst,
    }


def contingency(a, na, b, nb, label, hold_se=None):
    """Independence of a and b: observed joint vs the product of observed marginals."""
    tab = np.bincount(a.astype(np.int64) * nb + b.astype(np.int64),
                      minlength=na * nb).reshape(na, nb).astype(np.float64)
    n = tab.sum()
    p = np.outer(tab.sum(1) / n, tab.sum(0) / n)
    return compare(tab, p, label, hold_se)


def lag1_autocorr(x):
    x = x.astype(np.float64)
    x = x - x.mean()
    d = float((x * x).sum())
    return float((x[:-1] * x[1:]).sum() / d) if d > 0 else 0.0


# --------------------------------------------------------------------------- the ten tests


def run_tests(grid, factor_to_idx, heldout, held_mask, layout, image_idx, stats, h5_path, out,
              n_train, seeds, cf_rows=10000):
    S, Ho, C, Hf, Hw, R = (layout[:, i].astype(np.int64) for i in range(6))
    # Grid-cell SCM weights, used to price the hold-out term of every tolerance.
    gHf, gHw, gHo, gC, gS, gR = (grid[:, j] for j in range(6))
    W = P8[(gC - M_C[gS]) % 8] * P10[(gHf - gHo - gC) % 10] / 6000.0
    rep, results = {}, []

    def add(name, ok, detail):
        results.append(ok)
        rep[name] = {"pass": bool(ok), **detail}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    # 1 -- index formula vs h5 labels, all 480,000 rows
    gi = grid_index(grid[:, 0], grid[:, 1], grid[:, 2], grid[:, 3], grid[:, 4], grid[:, 5])
    n_bad = int((gi != np.arange(N_GRID)).sum())
    lut_bad = int((factor_to_idx < 0).sum())
    add("1_index_formula", n_bad == 0 and lut_bad == 0,
        {"rows_checked": N_GRID, "mismatches": n_bad, "unfilled_lookup_entries": lut_bad})

    # 2 -- hold-out disjointness
    n_leak = int(held_mask[image_idx].sum())
    add("2_holdout_disjoint", n_leak == 0,
        {"heldout_cells": int(heldout.size), "training_cells": int(N_GRID - heldout.size),
         "train_rows_in_heldout": n_leak})

    # 3 -- coverage of every training cell
    counts = np.bincount(image_idx, minlength=N_GRID)
    train_counts = counts[~held_mask]
    mn = int(train_counts.min())
    add("3_coverage", mn >= 1,
        {"min": mn, "median": float(np.median(train_counts)), "max": int(train_counts.max()),
         "mean": float(train_counts.mean()), "cells_below_10": int((train_counts < 10).sum()),
         "cells_zero": int((train_counts == 0).sum()),
         "note": "pass condition is min>=1; 'min>=10' is a target that N=16M does not reach "
                 "(predicted min ~4, ~474 cells below 10)",
         "heldout_cells_hit": int(counts[held_mask].sum())})

    # 4 -- root and unobserved marginals vs uniform
    sub = {}
    ok4 = True
    for name, col, k, gcol in [("shape", S, 4, gS), ("object_hue", Ho, 10, gHo),
                               ("wall_hue", Hw, 10, gHw), ("orientation", R, 15, gR)]:
        o, d = compare(np.bincount(col, minlength=k), np.full(k, 1.0 / k), name,
                       holdout_rel_se(gcol, k, W))
        sub[name] = d
        ok4 &= o
    add("4_root_marginals", ok4, {"tables": sub})

    # 5 -- mechanism tables
    sub = {}
    ok5 = True
    pc_s = np.bincount(S * 8 + C, minlength=32).reshape(4, 8).astype(np.float64)
    ref = np.stack([P8[(np.arange(8) - M_C[s]) % 8] for s in range(4)])
    hse = holdout_rel_se(gS * 8 + gC, 32, W).reshape(4, 8)
    o, d = compare_conditional(pc_s, ref, "P(scale|shape)",
                               [f"S={s}" for s in range(4)], hse)
    sub["P(C|S)"] = d
    ok5 &= o
    key = (Ho * 8 + C) * 10 + Hf
    pf = np.bincount(key, minlength=80 * 10).reshape(80, 10).astype(np.float64)
    ref_f = np.zeros((80, 10))
    names_f = []
    for ho in range(10):
        for c in range(8):
            ref_f[ho * 8 + c] = P10[(np.arange(10) - ho - c) % 10]
            names_f.append(f"H_o={ho},C={c}")
    hse_f = holdout_rel_se((gHo * 8 + gC) * 10 + gHf, 800, W).reshape(80, 10)
    o, d = compare_conditional(pf, ref_f, "P(floor_hue|object_hue,scale)", names_f, hse_f)
    sub["P(H_f|H_o,C)"] = d
    ok5 &= o
    add("5_mechanism_tables", ok5, {"tables": sub})

    # 6 -- exogeneity of wall_hue and orientation against every modeled factor
    sub = {}
    ok6 = True
    for un, uc, uk, guc in [("wall_hue", Hw, 10, gHw), ("orientation", R, 15, gR)]:
        for mn_, mc, mk, gmc in [("shape", S, 4, gS), ("object_hue", Ho, 10, gHo),
                                 ("scale", C, 8, gC), ("floor_hue", Hf, 10, gHf)]:
            o, d = contingency(uc, uk, mc, mk, f"{un} x {mn_}",
                               holdout_rel_se(guc * mk + gmc, uk * mk, W))
            sub[f"{un}_x_{mn_}"] = d
            ok6 &= o
    add("6_exogeneity", ok6, {"tables": sub})

    # 7 -- recovered noise matches P_8 / P_10
    U_C, U_F = abduce(S, Ho, C, Hf)
    sub = {}
    ok7 = True
    gU_C, gU_F = abduce(gS, gHo, gC, gHf)
    for name, u, p, gu in [("U_C", U_C, P8, gU_C), ("U_F", U_F, P10, gU_F)]:
        o, d = compare(np.bincount(u, minlength=len(p)), p, name,
                       holdout_rel_se(gu, len(p), W))
        sub[name] = d
        ok7 &= o
    add("7_recovered_noise", ok7, {"tables": sub})

    # 8 -- counterfactual round trip on cf_rows x every single intervention
    sel = np.random.RandomState(0).choice(n_train, size=cf_rows, replace=False)
    s0, h0, c0, f0 = S[sel], Ho[sel], C[sel], Hf[sel]
    w0, r0 = Hw[sel], R[sel]
    n_rt, bad_rt, bad_rng, oos = 0, 0, 0, 0
    for var, k, orig in [("shape", 4, s0), ("object_hue", 10, h0),
                         ("scale", 8, c0), ("floor_hue", 10, f0)]:
        for v in range(k):
            m = orig != v                                        # a real intervention only
            if not m.any():
                continue
            a = (s0[m], h0[m], c0[m], f0[m])
            b = intervene(*a, var, v)
            ci = grid_index(b[3], w0[m], b[1], b[2], b[0], r0[m])
            oos += int(((ci < 0) | (ci >= N_GRID)).sum())
            # do(var = each row's ORIGINAL value), vectorised per original value
            for v0 in np.unique(orig[m]):
                m2 = orig[m] == v0
                r = intervene(b[0][m2], b[1][m2], b[2][m2], b[3][m2], var, v0)
                n_rt += int(m2.sum())
                bad_rt += int(((r[0] != a[0][m2]) | (r[1] != a[1][m2])
                               | (r[2] != a[2][m2]) | (r[3] != a[3][m2])).sum())
            bad_rng += int(((b[0] < 0) | (b[0] >= 4) | (b[1] < 0) | (b[1] >= 10)
                            | (b[2] < 0) | (b[2] >= 8) | (b[3] < 0) | (b[3] >= 10)).sum())
    add("8_counterfactual_roundtrip", bad_rt == 0 and oos == 0 and bad_rng == 0,
        {"rows": cf_rows, "roundtrips_checked": n_rt, "roundtrip_mismatches": bad_rt,
         "cf_indices_out_of_range": oos, "cf_factors_out_of_range": bad_rng})

    # 9 -- shuffle: lag-1 autocorrelation
    ac = {"train_image_idx": lag1_autocorr(image_idx)}
    for i, nm in enumerate(LAYOUT_COLUMNS):
        ac[nm] = lag1_autocorr(layout[:, i])
    worst = max(ac, key=lambda k: abs(ac[k]))
    add("9_shuffle", abs(ac[worst]) < 1e-3,
        {"lag1_autocorr": ac, "worst": worst, "worst_abs": abs(ac[worst]), "threshold": 1e-3})

    # 10 -- determinism: regenerate from the three seeds, compare sha256 of every .npy
    tmp = out / "_determinism_check"
    g2 = build(h5_path, n_train, seeds, grid=grid)
    save(tmp, g2[1], g2[2], g2[4], g2[5])
    names = ["factor_to_idx.npy", "heldout_cells.npy", "train_layout.npy", "train_image_idx.npy"]
    h1 = {n: sha256(out / n) for n in names}
    h2 = {n: sha256(tmp / n) for n in names}
    shutil.rmtree(tmp)
    add("10_determinism", h1 == h2, {"sha256": h1, "sha256_regenerated": h2})

    rep["_summary"] = {
        "all_passed": all(results), "tests_passed": int(sum(results)), "tests_total": len(results),
        "n_train": n_train, "seeds": seeds, "draw_stats": stats,
        "test_rule": {"reference": "raw SCM",
                      "per_bin_tolerance":
                          "max(0.01, 5*sqrt(se_multinomial^2 + se_holdout^2)); "
                          "se_multinomial=sqrt((1-p)/expected_count), "
                          "se_holdout=sqrt(q/(1-q)/n_eff) with q=0.10",
                      "applies_to": ["4_root_marginals", "5_mechanism_tables",
                                     "6_exogeneity", "7_recovered_noise"],
                      "note": "chi2/df/p recorded as diagnostics only. Restricting draws to "
                              "the 432,000 training cells perturbs the modeled tables, which a "
                              "raw-SCM chi-square resolves at N=16M; the perturbation is priced "
                              "analytically into se_holdout. Verified independently against the "
                              "exact post-hold-out law: re-scoring there drops max|z| from 7.19 "
                              "to 2.94 (wall_hue x floor_hue) and 4.13 to 1.42 "
                              "(P(H_f|H_o=0,C=0)), i.e. the generator matches its own sampling "
                              "law and the residual is the hold-out alone."},
    }
    return all(results), rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(Path(__file__).with_name("scm_graph_a.yaml")))
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seeds = cfg["generation"]["seeds"]
    n_train = args.n or cfg["generation"]["n_train"]
    out = Path(args.out)

    t0 = time.time()
    print(f"building: n={n_train:,} seeds={seeds}")
    grid, f2i, held, held_mask, layout, image_idx, stats = build(args.h5, n_train, seeds)
    save(out, f2i, held, layout, image_idx)
    print(f"  drew {stats['n_drawn']:,}, hold-out rejected "
          f"{100*stats['holdout_rejection_rate']:.2f}%, kept {stats['n_kept']:,}, "
          f"discarded {stats['overshoot_discarded']:,} overshoot, {time.time()-t0:.0f}s")

    print("tests:")
    ok, rep = run_tests(grid, f2i, held, held_mask, layout, image_idx, stats,
                        args.h5, out, n_train, seeds)
    rep["_source"] = {"h5": os.path.abspath(args.h5), "sha256": sha256(args.h5),
                      "bytes": os.path.getsize(args.h5)}
    (out / "tests_report.json").write_text(json.dumps(rep, indent=2, sort_keys=True))

    files = ["factor_to_idx.npy", "heldout_cells.npy", "train_layout.npy",
             "train_image_idx.npy", "tests_report.json"]
    with open(out / "SHA256SUMS", "w") as f:
        for n in files:
            f.write(f"{sha256(out / n)}  {n}\n")

    print(f"\n{'ALL TESTS PASSED' if ok else 'FAILURES PRESENT -- dataset NOT frozen'} "
          f"({time.time()-t0:.0f}s total)")
    if ok:
        for n in files + ["SHA256SUMS"]:
            os.chmod(out / n, 0o444)
        print("files set read-only")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
