# Running a graph-A model evaluation on a worker node

Everything a second machine needs. No step depends on the originating node.

**This node is running `k1`.** A worker should take **`k5` and `k11`**.

---

## 1. Code

```bash
git clone <repo> && cd SpecRoute
git checkout celebahq-ladder          # evaluation code lives here
```

Needs `.venv` with torch, torchvision, pytorch-lightning, lmdb, numpy, scipy.

## 2. Pull everything from S3

```bash
B=s3://najibi-research-7f2a/able/evaluators

# source images + attribute table (5.7 GB) -- NOT in git, and the eval cannot run without it
aws s3 cp $B/data/celebahq_256.lmdb experiments/hdae/data/packed/celebahq_256.lmdb --recursive
aws s3 cp $B/data/celebahq_256_attrs.npz experiments/hdae/data/packed/

# 64px readers: the bank routes each attribute to the architecture that won on held-out F1
aws s3 cp $B/celebahq64/ experiments/hdae/outputs/attr_predictors_celebahq64/ --recursive

# cohorts (which source, which target, after SCM propagation)
aws s3 cp $B/cohorts/celebahq/ outputs/cohorts/celebahq/ --recursive

# per-attribute guidance, PER MODEL
aws s3 cp $B/guidance/ outputs/eval_graphA/ --recursive

# model checkpoints
A=s3://najibi-research-7f2a/able/celebahq/graph_a
mkdir -p experiments/hdae/outputs/eval_ckpts
aws s3 cp "$A/k5_p2/epoch=124-step=46875.ckpt"      experiments/hdae/outputs/eval_ckpts/k5.ckpt
aws s3 cp "$A/k11_cd015r/epoch=124-step=46875.ckpt" experiments/hdae/outputs/eval_ckpts/k11.ckpt
```

## 3. Run

```bash
python able/datasets/celebahq/run_eval.py \
  --config experiments/hdae/configs/celebahq256_k5_p2.yaml \
  --ckpt   experiments/hdae/outputs/eval_ckpts/k5.ckpt \
  --tag    k5 \
  --guidance outputs/eval_graphA/g_k5.json \
  --out    outputs/eval_graphA \
  --batch 24 --block 512 --T 100 --save-256 1
```

Then the same with `k11` / `celebahq256_k11_cd015r.yaml` / `g_k11.json`.

Or supervised (retries, resumes, halves batch on OOM):

```bash
export CFG_k5=experiments/hdae/configs/celebahq256_k5_p2.yaml
export CFG_k11=experiments/hdae/configs/celebahq256_k11_cd015r.yaml
export G_k5=$PWD/outputs/eval_graphA/g_k5.json
export G_k11=$PWD/outputs/eval_graphA/g_k11.json
export BATCH=24
# edit the `for m in ...` line to `k5 k11`
bash able/datasets/celebahq/supervise_eval.sh
```

**Resumable.** One npz per 512-source block under `outputs/eval_graphA/<tag>_blocks/`; a rerun
skips blocks already present. A crash or OOM costs one block.

## 4. Settings, and why

| | value |
|---|---|
| batch | **24** — measured 97% GPU at 13 GB of 20 GB; larger risks OOM for no gain once compute-bound |
| T | 100 |
| precision | fp16 autocast, channels_last, TF32 |
| guidance | **per attribute per model**, from the files above |

Per-attribute guidance (each measured on that model's own final checkpoint):

| | Male | Young | Beard | Bald |
|---|---|---|---|---|
| k1 | 6 | 6 | 6 | 5 |
| k5 | 2 | 3 | 2 | 6 |
| k11 | 5 | 3 | 2 | 6 |

**Throughput measured here: ~20 s per source, so ~17–20 h per model** at the full 3,000-source
cohort and T=100. If the worker is slower, the two knobs that scale linearly are `--T 50`
(halves it) and a reduced cohort. Both must then be applied to *every* model or the ladder
comparison breaks.

## 5. What comes back

Per model, under `outputs/eval_graphA/<tag>_blocks/b####.npz`:

- `src64`, `rec64`, `cf64_<attr>` — uint8 images, the scored resolution
- `rec256`, `cf256_<attr>` — retained full-resolution output, never scored
- `read_src_<attr>`, `read_recon_<attr>`, `read_cf_<attr>_<reader>` — the reader bank's verdicts
- `row`, `factual`, `target`, `guidance`

Ship the whole `<tag>_blocks/` directory back to
`s3://najibi-research-7f2a/able/evaluators/results/<tag>/`. Metrics (CC, fc_obs, fc_unobs, CF1
per attribute) are computed from the readings afterwards, so the worker does not need to
compute them.

## 6. Protocol

`docs/EVALUATION_PROTOCOL.md` (also at `$B/EVALUATION_PROTOCOL.md`). Rationale and measured
failure cases in `EVALUATION_RATIONALE.md`.
