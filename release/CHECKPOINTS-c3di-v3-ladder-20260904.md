# c3di-v3-ladder-20260904

The Causal3DIdent depth ladder is complete: k=1, k=5 and k=11 all trained to epoch 50 on an
identical recipe. This tag marks k=5 finishing and the ladder closing.

## The ladder

| model | taps | per-tap dims | steps | samples | lr | status |
|-------|------|--------------|------:|--------:|----|--------|
| k=1   | `[mid]` | 512 | 217,936 | 12,348,000 | **1e-4** | epoch 50 |
| k=5   | `[0,4,8,12,mid]` | 2x103 + 3x102 | 192,936 | 12,348,000 | 2e-4 | epoch 50 |
| k=11  | `[0,2,4,6,8,10,12,14,15,16,mid]` | 6x47 + 5x46 | 192,936 | 12,348,000 | 2e-4 | epoch 50 |

All three share seed 42, batch 64, arm C conditioning (Fourier + per-attribute RMSNorm),
`concat_film` fusion, and a 512-d semantic budget. k=5's taps are a strict SUBSET of k=11's, so
the ladder is nested rather than three unrelated tap layouts.

**The one confound, carried forward:** k=1 ran at lr 1e-4, not the configured 2e-4 -- its run
resumed from an earlier checkpoint and Lightning restored that optimiser state. k=5 vs k=11 is a
CLEAN comparison; k=1 is the rung with the caveat.

## What is in S3 under this tag's prefix

k=5's checkpoints live under the `c3di-v2-validated-20260827/hdae/k5/` prefix (they were archived
as they were produced, before this tag existed):

    hdae/k5/c3di_k5_final-epoch=12-step=48234.ckpt    2,091,983,572
    hdae/k5/c3di_k5_final-epoch=25-step=96468.ckpt    2,091,983,700
    hdae/k5/c3di_k5_final-epoch=37-step=144702.ckpt   2,091,983,700
    hdae/k5/c3di_k5_final-step192936.ckpt             2,091,983,700   <- epoch 50, the release

## Verification of the k=5 release checkpoint

The training node (gpu1, 149.165.153.184) became unreachable after the run finished -- it answers
ICMP but port 22 is closed -- so the usual source-vs-S3 ETag comparison was not possible. The S3
object was instead verified by downloading and inspecting it directly:

    loads cleanly            yes, 2,091,983,700 bytes
    global_step              192936  (config max_steps 192937)
    epoch                    50
    optimiser lr             2e-4, matching the config
    encoder tap heads        5 in `model.`, 5 in `ema_model.`
    per-tap dims             [103, 103, 102, 102, 102], sum 512 -- matches the config exactly
    EMA weights              present

That establishes the object is a complete, loadable, architecturally-correct k=5 checkpoint at
the intended step. It does NOT establish bit-identity with the file on gpu1, which cannot be
checked while that host is down. If gpu1 returns, recompute its multipart ETag and compare.

## Reproduce

    git checkout c3di-v3-ladder-20260904
    aws s3 cp s3://najibi-research-7f2a/wip-from-826a88e/checkpoints/c3di-v2-validated-20260827/hdae/k5/c3di_k5_final-step192936.ckpt .
    # config: experiments/hdae/configs/c3di_hier_k5_final.yaml

See REPRODUCE.md for the full result-to-commit ledger, and
release/CHECKPOINTS-c3di-v2-validated-20260827.md for the k=1 / k=11 results and the
over-training findings that set the epoch-50 policy.
