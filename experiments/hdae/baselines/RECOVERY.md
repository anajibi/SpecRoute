# Rebuilding this project from S3 alone

Everything needed to resume is in `s3://najibi-research-7f2a/hdae-handoff/`. Nothing below requires
access to the original machine. Verified 4 November 2026.

Read `AGENT.md` first for what the project is and where things stand; this file is only the restore
procedure.

---

## 1. Code

    aws s3 cp s3://najibi-research-7f2a/hdae-handoff/code/hdae_repo.bundle .
    git clone hdae_repo.bundle SpecRoute
    cd SpecRoute && git checkout celebahq-ladder

The bundle carries the **complete history and every branch**, so GitHub is not needed. If GitHub is
reachable, `git clone git@github.com:anajibi/SpecRoute.git` is equivalent.

## 2. Environment

Python **3.11.16**.

    python3.11 -m venv .venv
    .venv/bin/pip install -r <(aws s3 cp s3://.../code/pip_freeze_exact.txt -)

`code/pip_freeze_exact.txt` pins all 102 packages as they were (torch 2.2.2+cu121). The looser
`code/requirements_hdae.txt` and `code/requirements_diffae_upstream.txt` are also there.

The vendored DiffAE upstream is inside the repo (`diffae_upstream/`, 68 tracked files), so no
separate download.

> `sys.path` order matters: the patched `choices.py` must precede `diffae_upstream/`, or
> `ModelName.hier_autoenc` raises `AttributeError`. `scripts/train.py` already does this.

## 3. Data

    aws s3 cp s3://.../data_packed/celebahq_256_pack.tar.gz - | tar xz -C experiments/hdae/data/packed/

Gives `celebahq_256.lmdb/` (data.mdb, meta.json) and `celebahq_256_attrs.npz`. **Verified**: 41
columns, 30,000 rows, partitions {0,1,2}, derived `Beard` at column 40 (prevalence 0.1891), with
`Eyeglasses` 15, `Male` 20, `Smiling` 31, `Young` 39, `Bald` 4 — the indices every config expects.

The source JPEGs are **not** in S3 and are not needed: the packed LMDB is the training input. To
re-pack from scratch, CelebA-HQ is public and `experiments/hdae/scripts/pack_celebahq_256.py` is the
packer. The derived `Beard` column is `-No_Beard` appended as column 40; `.orig40` in the repo
history is the pre-derivation copy.

MorphoMNIST: `data_packed/morphomnist_70k.h5` and `morphomnist.h5`.
Causal3DIdent raw data: not in S3; public on Zenodo (record 4784282, 250k/25k at 224px).
The trained C3DI models **are** in S3 — see step 6.

## 4. Pretrained initialisation

    aws s3 cp s3://.../celebahq256/ffhq256_autoenc_model.pt experiments/hdae/pretrained/

Required by every phase-1 config. 95% of the model transfers from it.

## 5. Fitted causal models

    aws s3 cp --recursive s3://.../scm/ experiments/hdae/outputs/scm/

Four files: `celebahq_scm.pt` (graph A), `morpho_scm.pt`, `causal3dident_scm.pt` and
`causal3dident_scm_spline.pt`. These were **local-only until 4 November** and are needed for
interventions, not for training.

Graph B's SCM does not exist yet — two independent binary marginals, still to fit.

## 6. Models

Final checkpoints, one per rung (see `AGENT.md` §2 for the full table):

    celebahq256/k1_p2/epoch=124-step=46875.ckpt          graph A, K=1
    celebahq256/k5_p2/epoch=124-step=46875.ckpt          graph A, K=5
    celebahq256/k11_cd015r/epoch=124-step=46875.ckpt     graph A, K=11  (the evaluated one)
    celebahq256/b_k1_p2/epoch=124-step=46875.ckpt        graph B, K=1
    celebahq256/b_k11_p2/epoch=124-step=46875.ckpt       graph B, K=11
    morpho_ladder/{k1,k5,k11}_s42/, _s43/                MorphoMNIST baseline arm
    morpho_adv/{k1,k5,k11}_s42/                          MorphoMNIST adversarial arm

Each CelebA directory also holds that stage's config, `logs/`, and the intermediate checkpoints.

Causal3DIdent models are under a **different prefix**:

    wip-from-826a88e/checkpoints/c3di-v2-validated-20260827/hdae/c3di_k1_final-step217936.ckpt
    wip-from-826a88e/checkpoints/c3di-v2-validated-20260827/hdae/k5/c3di_k5_final-step192936.ckpt
    wip-from-826a88e/checkpoints/c3di-v2-validated-20260827/hdae/c3di_k11_final-step192936.ckpt

Verify any checkpoint after download with `experiments/hdae/ops/verify_s3.sh`; size equality is not
proof of a complete upload.

## 7. Evaluation assets

    celebahq256/readers40_per_attribute.tar.gz    the 40 per-attribute readers
    celebahq256/reader_keep.json                  graph A basis, 28 FC_unobs
    celebahq256/reader_keep_graphB.json           graph B basis, 30 FC_unobs
    celebahq256/baseline_kit.tar.gz               cohort_2048.json, scorer, protocol, reference numbers
    attr_predictors_70k.tar.gz                    MorphoMNIST readers
    c3di/attr_predictors_c3di.tar.gz              Causal3DIdent readers

`cohort_2048.json` is graph A's cohort (all-male, Bald flipped). **There is no graph-B cohort** —
it still has to be defined.

## 8. Results

`celebahq256/results/` and `results/` hold the metric JSONs; `c3di/c3di_results_and_analysis.tar.gz`
holds the Causal3DIdent ladder; `morpho_ladder/sweeps/` and `morpho_adv/analysis/` the MorphoMNIST
sweeps.

## 9. Running

`ops_scripts.tar.gz` and `ops_README.md`, also committed at `experiments/hdae/ops/`. Read
`configs/LADDER_PROTOCOL.md` before training. `AGENT.md` §6 lists the four rules that each came from
a failure — the `init_from`-on-converged-weights NaN in particular.

## 10. What is NOT in S3, deliberately

| | why |
|---|---|
| CelebA-HQ source JPEGs (2.7 GB) | public dataset; the packed LMDB is the training input |
| Causal3DIdent raw data (13 GB) | public on Zenodo |
| local venv | rebuilt from `pip_freeze_exact.txt` |
| per-run one-off shell scripts | superseded by `ops/supervise.sh` and `ops/watch.sh` |
