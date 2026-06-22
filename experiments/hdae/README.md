# HDAE: hierarchical semantic latents for DiffAE

This experiment leaves the DiffAE denoising decoder, diffusion objective, stochastic DDIM code, EMA, and optimizer/schedule unchanged. It replaces only the semantic encoder with the same upstream BeatGANs down path plus configurable intermediate pooling heads. Latents are ordered **coarse-to-fine** (ascending tap resolutions), concatenated, and projected to upstream `style_ch`.

`per_resolution` is recognized but deliberately raises `NotImplementedError`: phase 1 uses the safer unchanged-decoder `concat_proj` path.

```bash
# 0. preprocess ONCE (resize CelebA-HQ -> packed LMDB @ 64 + aligned attrs)
python experiments/hdae/scripts/preprocess_data.py --config experiments/hdae/configs/celeba64_hier_k3.yaml
# smoke
python experiments/hdae/scripts/smoke_test.py --config experiments/hdae/configs/celeba64_hier_k3.yaml
# full train (2 GPUs)
python experiments/hdae/scripts/train.py --config experiments/hdae/configs/celeba64_hier_k3.yaml
# reconstruction eval
python experiments/hdae/scripts/reconstruct.py --config experiments/hdae/configs/celeba64_hier_k3.yaml --ckpt <path>
# unit tests
pytest experiments/hdae/tests
```

The preprocessor detects direct filename alignment (Case A) or requires a CelebA-HQ-to-CelebA mapping (Case B); it never silently joins by row index. The smoke test needs the configured source data and writes `smoke_grid.png` under `output_dir`.

## Latent probing

After training a reconstruction model, extract per-level semantic latents and train one linear binary classifier for each `(latent level, attribute)` pair:

```bash
python experiments/hdae/latent_probing/extract_latents.py \
  --config experiments/hdae/configs/celeba64_hier_k3.yaml \
  --ckpt <path> \
  --output experiments/hdae/outputs/celeba64_hier_k3/latent_probing/latents.npz

python experiments/hdae/latent_probing/train_linear_probes.py \
  --latents experiments/hdae/outputs/celeba64_hier_k3/latent_probing/latents.npz \
  --output-dir experiments/hdae/outputs/celeba64_hier_k3/latent_probing/probes
```

For 3 latent levels and 40 CelebA attributes this trains 120 independent linear classifiers and writes one metrics row per classifier.

## Learned null-token ablations

Each HDAE latent level owns one learned null token. During training, every level is independently replaced by its null token with probability `conditioning.latent_drop_prob` (default `0.12`). The null tokens are model parameters, so they are saved and loaded with checkpoints. At test time, use them to ablate selected semantic levels:

```bash
python experiments/hdae/latent_probing/reconstruct_with_nulls.py \
  --config experiments/hdae/configs/celeba64_hier_k3.yaml \
  --ckpt <path> \
  --null-levels 0,2 \
  --output experiments/hdae/outputs/celeba64_hier_k3/null_levels_0_2.png
```

## One-command pipeline and pseudo-counterfactuals

Run the full sequence (preprocess, train, reconstruct, extract latents, train linear probes, train an image attribute classifier, and evaluate a latent-direction pseudo-counterfactual):

```bash
python experiments/hdae/scripts/run_full_pipeline.py \
  --config experiments/hdae/configs/celeba64_hier_k3.yaml \
  --attribute Smiling \
  --cf-level best \
  --cf-strength 2.0

# The pipeline skips completed stages by default; add --force to rerun them.
```

The counterfactual stage uses the selected linear-probe direction (for example `Smiling`) to edit one latent level, decodes pseudo-counterfactual images, then scores all 40 attributes with an image-space CelebA attribute classifier. It reports the target-attribute change and non-target preservation metrics so hierarchy sizes can be compared.

## Swap/null diagnostic grid and probe analysis plots

Generate a grid with source/donor rows, swap rows, and null-token rows. The script reads the configured number of levels `K` and renders every single-level swap (`Z1` ... `ZK`), every adjacent-pair swap (`Z1+Z2` ... `Z{K-1}+ZK`), and every single-level null-token ablation:

```bash
python experiments/hdae/latent_probing/swap_null_grid.py \
  --config experiments/hdae/configs/celeba64_hier_k3.yaml \
  --ckpt <path> \
  --output experiments/hdae/outputs/celeba64_hier_k3/latent_probing/swap_null_grid.png
```

Generate an abductive `Z`/`x_T` reveal grid. This abducts the original semantic latents and DDIM stochastic code once, then decodes an all-null row, forward cumulative rows (`Z0`, `Z0+Z1`, ...), and reverse cumulative rows (`Z-1`, `Z-1+Z-2`, ...) with unrevealed levels replaced by learned null tokens:

```bash
python experiments/hdae/latent_probing/abduct_xt_z_grid.py \
  --config experiments/hdae/configs/celeba64_hier_k3.yaml \
  --ckpt <path> \
  --output experiments/hdae/outputs/celeba64_hier_k3/latent_probing/abduct_xt_z_grid.png
```

Analyze probe results with heatmaps and best-level summaries:

```bash
python experiments/hdae/latent_probing/analyze_probe_results.py \
  --probe-metrics experiments/hdae/outputs/celeba64_hier_k3/latent_probing/probes/probe_metrics.csv \
  --output-dir experiments/hdae/outputs/celeba64_hier_k3/latent_probing/analysis
```
