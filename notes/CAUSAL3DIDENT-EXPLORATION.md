# Causal3DIdent — exploration (2026-08-24)

Machine: exouser box, A100-40GB. Branch `wip-from-826a88e` @ 826a88e.
Repo code NOT modified. Reference repo cloned to `/home/exouser/refs/ssl_identifiability`.

Paper: von Kügelgen et al., "Self-Supervised Learning with Data Augmentations Provably
Isolates Content from Style", NeurIPS 2021 (arXiv:2106.04619).
Data: Zenodo 10.5281/zenodo.4784282, CC-BY-4.0, 9.125 GB
      (trainset.tar.gz 8.296 GB, testset.tar.gz 0.829 GB).

NOTE: this is TODO-List item 3's "Causal3DIdent", which was listed but never started.

## Layout

    {train,test}set/
      images_{0..6}/NNNN.png     224x224 RGB
      raw_latents_{0..6}.npy     (N, 10) float32, all in [-1, 1]
      latents_{0..6}.npy         (N, 10) float32  -- rescaling of raw, see below

7 object classes, verified visually:
  0 Teapot   1 Hare   2 Dragon   3 Cow   4 Armadillo   5 Horse   6 Head

Test: 3,600 per class = 25,200. Train: 36,000 per class = **252,000**
(README says 250,000; the actual file count is 252,000).
`datasets/clevr_dataset.py::CausalDataset` loads `raw_latents_{i}.npy`.

## The 10 latents (index order from clevr_dataset.py change_list)

  0 pos_x    1 pos_y    2 pos_z          object position
  3 rot_a    4 rot_b    5 rot_g          object rotation
  6 pos_spl                              spotlight position
  7 hue_obj  8 hue_spl  9 hue_bg         hues

## latents vs raw_latents — exact, no information difference

    latents[:, 0:3] = raw_latents[:, 0:3] * 2.0        max abs err 0.00e+00
    latents[:, 3:10] = raw_latents[:, 3:10] * pi/2     max abs err 0.00e+00

Pure per-dim rescaling (positions x2, angles/hues to [-pi/2, pi/2] radians).
Use `raw_latents` — that's what the official loader uses.

## Causal structure — DERIVED EMPIRICALLY (testset, 25,200 samples)

### 1. Object class is a parent of pos_y, rot_a/b/g, hue_obj
Per-class means differ sharply (e.g. pos_y: class0 +0.005, class1 -0.422, class3 +0.409;
hue_obj: class4 +0.461, class5 -0.478). Environment latents pos_spl/hue_spl/hue_bg have
mean ~0 and std ~0.577 (= 1/sqrt(3), i.e. Uniform[-1,1]) for EVERY class -> independent
of class, as the paper states.

### 2. pos_x -> pos_spl, present in classes 1-6, ABSENT in class 0, SIGN FLIPS BY CLASS

    class 0:  r = +0.018   (no coupling)
    class 1:  r = -0.771
    class 2:  r = -0.780
    class 3:  r = +0.774
    class 4:  r = +0.762
    class 5:  r = -0.769
    class 6:  r = +0.765

### 3. The "camouflage" hue edges exist in ONLY 2 of 7 classes, with opposite signs

    class 1 (Hare):   hue_obj~hue_spl = +0.393   hue_obj~hue_bg = +0.413
    class 2 (Dragon): hue_obj~hue_spl = -0.413   hue_obj~hue_bg = -0.427
    classes 0,3,4,5,6: all |r| < 0.04

The paper's remark about object hue "centered about background & spotlight hue ...
biological camouflage in certain hare species" is class-1-specific. Class 2 is the
deliberate anti-camouflage mirror.

### 4. Class 0 (Teapot) is a fully-independent control
All within-class |r| <= 0.039. It also has a visibly different pos_x marginal:
bell-shaped (hist 323/861/1192/878/346, std 0.437) vs uniform for classes 1-6
(hist ~700 each, std ~0.58).

### 5. Two pooled correlations are Simpson's-paradox artifacts — do not model them
    pos_y ~ rot_g   : pooled -0.229, within-class [+0.03,-0.02,+0.00,+0.01,+0.00,-0.02,-0.00]
    pos_y ~ hue_obj : pooled +0.191, within-class all |r| <= 0.012
Both vanish within class; they arise purely from pooling classes with different means.

## Replication on the full trainset (252,000 samples) — all of the above holds

    cls   px~spl   hobj~hspl   hobj~hbg   max|r| of all other pairs
      0   -0.006      -0.001     -0.001   0.012
      1   -0.766      +0.399     +0.400   0.015
      2   -0.764      -0.398     -0.399   0.015
      3   +0.765      -0.011     +0.000   0.015
      4   +0.768      -0.003     -0.000   0.014
      5   -0.766      -0.006     -0.001   0.015
      6   +0.765      +0.007     +0.003   0.020

Every other latent pair is |r| <= 0.020 in every class. The rescaling identity
`latents = raw * [2,2,2,pi/2,...]` also holds on train (max err 1.03e-07, float32 eps).

Integrity: both tarballs MD5-verified against Zenodo
(testset c5d9d32d3737e241a2b12b968275fcb8, trainset acd98fda30eee75856dbbc7c54a27e45).

## Implication for this codebase

`configs/causal_graph.yaml` is a single static edge list. Causal3DIdent's DAG is
**class-modulated**: which edges exist AND their sign depend on object class. A flat
edge list cannot express "hue_bg -> hue_obj only for classes 1,2, with opposite signs",
nor "pos_x -> pos_spl for classes 1-6 but not 0". Options: one graph per class, or a
class-conditional SCM. This is a design decision to make before wiring the SCM up.

Separately: unlike CausalVerse's `fall` (h2 exactly determined by h1,r,u), Causal3DIdent's
dependencies are STOCHASTIC (r ~ 0.77, residual std ~0.37), not deterministic. Good for a
flow-based SCM, which models conditional distributions rather than exact functions.

## Local state

  experiments/hdae/data/causal3dident/{trainset,testset}.tar.gz
  experiments/hdae/data/causal3dident/testset/  (latents + 4 sample images per class)
Gitignored by the existing `**/data/**/*` rule.
