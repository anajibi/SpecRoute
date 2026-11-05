# 3DShapes - Graph A - v1 - FROZEN 2026-11-05

Protocol v1, all ten tests passed, files set read-only, hashes committed. Do not modify.

Repo: `git@github.com:anajibi/SpecRoute.git`  branch `shapes3d-graph-a`  (see git log for the commit)
Generator: `able/datasets/shapes3d/build_graph_a.py`   Spec: `able/datasets/shapes3d/scm_graph_a.yaml`

## Graph

    shape --> scale --> floor_hue <-- object_hue        unobserved: wall_hue, orientation

Modeled: shape(4), object_hue(10), scale(8), floor_hue(10).
Unobserved exogenous, never conditioned or intervened on, must be preserved: wall_hue(10), orientation(15).

| variable | mechanism | noise |
|---|---|---|
| shape, object_hue, wall_hue, orientation | identity | uniform |
| scale `C` | `m_C(S) (+)_8 U_C`, `m_C = (0,2,5,7)` | `P_8` |
| floor_hue `H_f` | `(H_o (+)_10 C) (+)_10 U_F` | `P_10` |

`P_K`: P(0)=0.25, P(+-1)=0.125, else 0.5/(K-3). Full support => positivity.
Abduction exact: `U_C = C (-)_8 m_C(S)`, `U_F = H_f (-)_10 (H_o (+)_10 C)`.

## Source

`3dshapes.h5`, 480,000 images, 64x64x3, full grid over
floor_hue(10) x wall_hue(10) x object_hue(10) x scale(8) x shape(4) x orientation(15).
URL `https://storage.googleapis.com/3d-shapes/3dshapes.h5`, 267,573,662 bytes,
md5 `099a2078d58cec4daad0702c55d06868` (verified). Mirrored at `able/shapes3d/source/3dshapes.h5`.

No images are generated or copied. The output is a layout over the existing grid.

## Seeds (frozen)

    split_seed   11052026
    draw_seed    11052027
    shuffle_seed 11052028

## Files

| file | dtype / shape | bytes | sha256 |
|---|---|---:|---|
| `train_image_idx.npy` | int32 (16,000,000,) | 64,000,128 | `8d3ea80ea99a6193a996d085ba70cf9bb47ee08e9aa84ce1c8e0f71ac46a1afe` |
| `train_layout.npy` | uint8 (16,000,000, 6) | 96,000,128 | `58f5ebe37899dde51be86c93662da7ad5bbc9a0d1504a10a9e15ae91c583b34e` |
| `factor_to_idx.npy` | int32 (10,10,10,8,4,15) | 1,920,128 | `bb7a59a8aed381ae25b9715544f9a5df6cf497ef082d595aa16a55feaffa44c5` |
| `heldout_cells.npy` | int32 (48,000,) | 192,128 | `7175b59deb4f4713e01a00194c4c1934654773368a5899bb513273cf54b001d6` |
| `tests_report.json` | - | 12,797 | `131bc88f4cb3411b932ff62019e1f723cd01e871acd45119a21eee3152144855` |

`train_layout.npy` column order: **shape, object_hue, scale, floor_hue, wall_hue, orientation**.
`factor_to_idx.npy` axis order is the h5's: **floor_hue, wall_hue, object_hue, scale, shape, orientation**.

Grid index (h5 row order, orientation fastest-varying):
`idx = H_f*48000 + H_w*4800 + H_o*480 + C*60 + S*15 + R`

## Loader

```python
import h5py, numpy as np
images = h5py.File("3dshapes.h5", "r")["images"][:]      # 480000x64x64x3 uint8, 5.9 GB in RAM
image_idx = np.load("train_image_idx.npy")               # what the loader reads, one pass
# batch: images[image_idx[i:i+B]]
```

Every method trains on the same `train_image_idx.npy`, one pass, N = 16,000,000.

## Test results (all ten passed)

| # | test | outcome |
|---|---|---|
| 1 | index formula vs h5 labels, all 480,000 rows | 0 mismatches, lookup fully populated |
| 2 | hold-out disjointness | 0 of 16,000,000 training rows in the 48,000 held-out cells |
| 3 | coverage of the 432,000 training cells | min 4, median 27, max 229, mean 37.0; 0 empty; 481 below 10 |
| 4 | root + unobserved marginals vs uniform | max rel dev 0.0052, max abs z 2.60 |
| 5 | mechanism tables `P(C|S)`, `P(H_f|H_o,C)` | max rel dev 0.0052 / 0.0652, max abs z 1.51 / 4.01 |
| 6 | exogeneity, wall_hue and orientation x 4 modeled | max rel dev 0.0225, max abs z 2.92 |
| 7 | recovered noise `U_C`, `U_F` vs `P_8`, `P_10` | max rel dev 0.0025, max abs z 1.45 |
| 8 | counterfactual round trip | 10,000 rows x all single interventions = 280,000 round trips, 0 mismatches, 0 out of range |
| 9 | shuffle, lag-1 autocorrelation | worst 4.55e-4 (wall_hue), threshold 1e-3 |
| 10 | determinism | regenerated from the three seeds, sha256 identical for all four .npy |

### Why the pass rule for tests 4-7 is a tolerance, not a chi-square p-value

The reference is the **raw SCM**, and the per-bin condition is

    |observed/expected - 1| <= max(0.01, 5 * sqrt(se_multinomial^2 + se_holdout^2))

with `se_multinomial = sqrt((1-p)/expected_count)` and `se_holdout = sqrt(q/(1-q)/n_eff)`, q = 0.10.

Restricting draws to the 432,000 training cells perturbs the modeled tables by ~0.1-2% relative.
That perturbation is a property of the design, not of the draw, and at N = 16M a chi-square
resolves it easily: the recorded `chi2_p` values run as low as 1e-126 even though the generator
is correct. `se_holdout` prices that term analytically, so the tolerance stays sensitive to real
defects (a wrong `m_C` or `P_K` lands hundreds of sigma out) without flagging the design.

Verified independently: re-scoring the same data against the **exact post-hold-out law** (closed
form over the 480,000 cells from the realised mask) drops max abs z from 7.19 to 2.94 on
`wall_hue x floor_hue` and from 4.13 to 1.42 on `P(H_f|H_o=0,C=0)` - i.e. the data matches its
own sampling law and the whole residual is the hold-out. chi2, df, p and both se components are
recorded per table in `tests_report.json`.

### Known, accepted deviation from the protocol document

Test 3's pass condition (every training cell >= 1) holds. Its stated *target* of min >= 10 is not
reached at N = 16M and cannot be: the rarest cells have expected count 21.2, so ~474 of the
189,000 rarest were predicted to land below 10 and 481 did. Reaching min >= 10 needs N ~ 24M.
Decision on record: keep N = 16M.
