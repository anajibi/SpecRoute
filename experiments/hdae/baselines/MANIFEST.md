# ABLE — S3 manifest

Bucket `najibi-research-7f2a`, prefix `able/`. Generated from the live listing, 4 November 2026.
**1538 objects, 391.00 GB.**

Read `AGENT.md` for what the project is and where it stands, `RECOVERY.md` to rebuild from scratch.

| prefix | size | objects |
|---|---|---|
| `able/celebahq/graph_a/` | 142.25 GB | 68 |
| `able/celebahq/graph_b/` | 84.02 GB | 50 |
| `able/morphomnist/tuning/` | 43.23 GB | 63 |
| `able/causal3dident/v2_validated/` | 25.30 GB | 25 |
| `able/morphomnist/ladder/` | 24.49 GB | 54 |
| `able/morphomnist/adv/` | 18.45 GB | 77 |
| `able/morphomnist/final/` | 12.16 GB | 8 |
| `able/morphomnist/scale/` | 7.35 GB | 5 |
| `able/causal3dident/v1_probes/` | 7.22 GB | 26 |
| `able/legacy/outputs/` | 6.71 GB | 902 |
| `able/data/` | 5.65 GB | 4 |
| `able/celebahq/assets/` | 5.54 GB | 8 |
| `able/baselines/causaldiffae_celebahq64_mybb/` | 2.73 GB | 17 |
| `able/baselines/dc64_sem1/` | 2.09 GB | 8 |
| `able/baselines/dc64_sem0/` | 1.13 GB | 6 |
| `able/causal3dident/bundles/` | 0.74 GB | 2 |
| `able/(root)/` | 0.69 GB | 10 |
| `able/morphomnist/assets/` | 0.69 GB | 3 |
| `able/data/celebahq_64.lmdb/` | 0.50 GB | 3 |
| `able/legacy/root_checkpoints/` | 0.03 GB | 120 |
| `able/baselines/dc64_code/` | 0.02 GB | 7 |
| `able/code/` | 0.01 GB | 4 |
| `able/causal3dident/results/` | 0.01 GB | 25 |
| `able/notes/` | 0.00 GB | 5 |
| `able/results/` | 0.00 GB | 6 |
| `able/scm/` | 0.00 GB | 4 |
| `able/celebahq/results/` | 0.00 GB | 10 |
| `able/configs/` | 0.00 GB | 17 |
| `able/causal3dident/v3_ladder/` | 0.00 GB | 1 |

## Final models

| model | key |
|---|---|
| CelebA graph A, K=1 | `able/celebahq/graph_a/k1_p2/epoch=124-step=46875.ckpt` |
| CelebA graph A, K=5 | `able/celebahq/graph_a/k5_p2/epoch=124-step=46875.ckpt` |
| CelebA graph A, K=11 | `able/celebahq/graph_a/k11_cd015r/epoch=124-step=46875.ckpt` |
| CelebA graph B, K=1 | `able/celebahq/graph_b/k1_p2/epoch=124-step=46875.ckpt` |
| CelebA graph B, K=11 | `able/celebahq/graph_b/k11_p2/epoch=124-step=46875.ckpt` |
| Causal3DIdent K=1 | `able/causal3dident/v2_validated/hdae/c3di_k1_final-step217936.ckpt` |
| Causal3DIdent K=5 | `able/causal3dident/v2_validated/hdae/k5/c3di_k5_final-step192936.ckpt` |
| Causal3DIdent K=11 | `able/causal3dident/v2_validated/hdae/c3di_k11_final-step192936.ckpt` |
| MorphoMNIST | `able/morphomnist/ladder/{k1,k5,k11}_s42/` and `_s43/`, `able/morphomnist/adv/` |

All eight single-file finals verified complete (multipart ETag plus zip end-of-central-directory
read back from S3) after the move.

## Notes

`k11_cd015` is the run that went non-finite, kept as evidence of the cold-Adam collapse; the usable
K=11 model is `k11_cd015r`. `able/legacy/` holds superseded earlier work. Baseline runs by the
second agent also write to `agentic_perception_dev/checkpoints/able-baselines/` and
`able-baselines-handoff/`, which were left in place so as not to break that agent's uploads.
