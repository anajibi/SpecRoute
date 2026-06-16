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
