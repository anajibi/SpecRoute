# Experiments

## CelebA pseudo-counterfactuals

This experiment runs CelebA/CelebA-HQ component ablations and DiffAE-style semantic edits in `z_sem` while keeping `x_T` fixed.

### Dataset setup (CelebA-HQ)

- Images: `/home/anajibi/HDM/diffae_latent_probe/data/raw_images/celeba-hq`
- Attributes: `/home/anajibi/HDM/diffae_latent_probe/data/celeba-hq/list_attr_celeba.txt`
- Splits: `/home/anajibi/HDM/diffae_latent_probe/data/celeba-hq/list_eval_partition.txt`

### Run

```bash
python experiments/run_celeba_pseudo_counterfactuals.py \
  --config configs/experiments/celeba_pseudo_counterfactuals.yaml
```

### Outputs

All outputs are written under `outputs/celeba_pseudo_counterfactuals/` including:

- `latents/` latent bundle for CelebA subset
- `component_ablation/` metrics + grids
- `attribute_directions/` learned directions
- `pseudo_counterfactuals/` edited images, grids, and preservation metrics

### Notes

These are pseudo-counterfactual edits in DiffAE latent space, not guaranteed causal counterfactuals.
