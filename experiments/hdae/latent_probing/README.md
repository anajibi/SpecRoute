# HDAE latent probing

This folder contains the first probing experiment: for every semantic latent level and every CelebA attribute, train an independent linear binary classifier.

For K latent levels and 40 attributes, the run trains `K × 40` classifiers. With `celeba64_hier_k3.yaml`, that is 120 classifiers.

## Workflow

```bash
# 1. Extract semantic latents from a trained HDAE checkpoint.
python experiments/hdae/latent_probing/extract_latents.py \
  --config experiments/hdae/configs/celeba64_hier_k3.yaml \
  --ckpt <path-to-hdae.ckpt> \
  --output experiments/hdae/outputs/celeba64_hier_k3/latent_probing/latents.npz

# 2. Train one linear probe for every (level, attribute) pair.
python experiments/hdae/latent_probing/train_linear_probes.py \
  --latents experiments/hdae/outputs/celeba64_hier_k3/latent_probing/latents.npz \
  --output-dir experiments/hdae/outputs/celeba64_hier_k3/latent_probing/probes
```

Outputs:

- `latents.npz`: `z_level_0`, ..., `attrs`, `partitions`, `indices`, `attribute_names`.
- `probe_metrics.csv`: one row per classifier with validation/test accuracy and balanced accuracy.
- `weights/*.pt`: one serialized linear classifier per level/attribute, including standardization statistics.
- `summary.json`: number of levels, attributes, and classifiers.

The probes use partition labels from the packed dataset. If a packed dataset has incomplete partition labels, training falls back to a deterministic 80/10/10 split, matching the probe code path.

## Null-token reconstruction ablations

To inspect what happens when one or more semantic levels are removed, force those levels to use their learned null tokens and save an original/reconstruction grid:

```bash
python experiments/hdae/latent_probing/reconstruct_with_nulls.py \
  --config experiments/hdae/configs/celeba64_hier_k3.yaml \
  --ckpt <path-to-hdae.ckpt> \
  --null-levels 1 \
  --output experiments/hdae/outputs/celeba64_hier_k3/latent_probing/null_level_1.png
```

## Pseudo-counterfactual preservation experiment

The counterfactual workflow chooses a linear-probe direction for a target attribute, edits that latent level, decodes pseudo-counterfactual images, and measures all 40 attributes with an image classifier:

```bash
python experiments/hdae/counterfactuals/train_attr_classifier.py \
  --config experiments/hdae/configs/celeba64_hier_k3.yaml \
  --output experiments/hdae/outputs/celeba64_hier_k3/counterfactuals/attr_classifier.pt

python experiments/hdae/counterfactuals/run_counterfactual_eval.py \
  --config experiments/hdae/configs/celeba64_hier_k3.yaml \
  --ckpt <path-to-hdae.ckpt> \
  --probe-metrics experiments/hdae/outputs/celeba64_hier_k3/latent_probing/probes/probe_metrics.csv \
  --probe-weights-dir experiments/hdae/outputs/celeba64_hier_k3/latent_probing/probes/weights \
  --attr-classifier experiments/hdae/outputs/celeba64_hier_k3/counterfactuals/attr_classifier.pt \
  --attribute Smiling \
  --level best \
  --output-dir experiments/hdae/outputs/celeba64_hier_k3/counterfactuals/Smiling
```
