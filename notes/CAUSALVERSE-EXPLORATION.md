# CausalVerse — initial exploration (2026-08-24)

Machine: exouser box, A100-40GB. Branch `wip-from-826a88e` @ 826a88e.
Nothing in the repo was modified; this is a findings note only.

## Dataset

`CausalVerse/CausalVerse_Image` on HuggingFace — public, Apache-2.0, ungated, parquet.
198,764 rows / **136.14 GB** / 139 shards. Paper: arXiv:2510.14049 (NeurIPS 2025 D&B).
Schema is uniform: `image`, `render_path`, `metavalue` (JSON string, split-specific keys).

| split      | rows   | size     | resolution      | factors |
|------------|--------|----------|-----------------|---------|
| fall       | 40,000 | 10.92 GB | 800x600 RGB     | id, h1, r, u, h2, view |
| refraction | 40,000 | 10.71 GB | 800x600 RGB     | id, theta1, theta2, n1, view |
| slope      | 40,000 | 16.69 GB | 800x600 RGB     | id, roughness, theta, mu_2, mu_1, l, v1, v0, view |
| spring     | 40,000 | 15.43 GB | 800x800 RGB     | id, h, r, m, k, l, view |
| scene1     | 11,736 | 19.94 GB | 1024x1024 RGBA  | 20 human factors |
| scene2     | 11,736 | 17.01 GB | 1024x1024 RGBA  | same 20 |
| scene3     | 11,736 | 22.46 GB | 1024x1024 RGBA  | same 20 |
| scene4     |  3,556 | 22.98 GB | 1024x1024 RGBA  | same 20 |

scene factors: domain, age, gender, muscle, weight, proportions, cupsize, firmness,
race, skin, pose, eye_texture, eyes, hair, eyelashes, eyebrows, suit, shirts, pants, shoes

## Verified ground-truth mechanisms

### fall — exactly deterministic
Pooled 2,777 unique (h1, r, u) combos across shards 0/3/7/10:

    h2 = 553.3986 * h1^0.5000 * r^3.0000 * u^-1.0000
    R^2 = 1.00000000, max abs err = 1.71e-14

Exponents are exact (1/2, 3, -1). Zero combos show any variation in h2
(max within-combo std = 0.00e+00).

DAG:  h1 -> h2 <- r,  u -> h2.   `view` (4 cameras) is a pure nuisance
variable: one unique h2 per `id` across all 4 views. `id` is the scene instance.

### scene1 — hard structural constraints
- `gender -> cupsize`: deterministic. gender=0 => cupsize == 0.0 exactly (100% of rows);
  gender=1 => 293 distinct continuous values.
- `proportions` is an EXACT ALIAS of `gender` (identical in every row) — a degenerate
  duplicate column. Do not condition on both.
- `suit` XOR (`shirts`,`pants`): suit==-1 in 473 rows, shirts==-1 in 114, never both,
  never neither. `shirts==-1` iff `pants==-1`. So -1 is a "not worn" SENTINEL.

## Gotchas

1. **Shards are sorted by factor value — one shard is a biased sample.**
   fall shard 0 has u=1.0 ONLY (zero variance); shard 3 has {1.4,1.6}; shard 10 has {3.0}.
   `r` widens from 8 distinct values (shard 0) to 14 (shard 10).
   Any sampler must cross shards or it silently drops a whole factor.

2. **The -1 sentinel is the same bug class this repo has hit four times**
   (see TODO-List: `to_cond_values` hue bug — raw value mistaken for class index).
   Feeding suit=-1 into a categorical embedder or lo/hi binner repeats it.

3. **Size/resolution.** 136 GB total vs ~38 GB free here. Downsampling 1024x1024 -> 64x64
   (the HDAE pipeline size) will destroy fine scene factors: eyelashes, eyebrows,
   eye_texture. RGBA on scene splits needs flattening to RGB.

## Why the physics splits matter for this project

`configs/causal_graph.yaml` has shipped with `edges: []` since item 2 landed, so CC
reduces algebraically to a plain flip rate and `propagate()` has never run on real data
(`verify_scm.py` fabricates a toy Male->Young->Smiling chain to test it). `fall` supplies
a non-trivial DAG whose answer is a closed form known to machine precision — it turns the
SCM from verified-but-dormant into something with a checkable ground truth.

## Local state

- `experiments/hdae/data/causalverse_sample/data/fall-00000-of-00011.parquet`   (949 MB)
- `experiments/hdae/data/causalverse_sample/data/scene1-00000-of-00020.parquet` (951 MB)
Both gitignored by the existing `**/data/**/*` rule.
