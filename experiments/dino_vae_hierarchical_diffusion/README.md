# DINO–VAE Hierarchical Latent Diffusion

An isolated, no-attribute experiment combining a frozen Stable Diffusion VAE, frozen DINOv2 evidence, stochastic top-down K=3/K=5 hierarchies, conditional diffusion priors, and a final latent diffusion decoder.

## Quick start

```bash
pip install -r experiments/dino_vae_hierarchical_diffusion/requirements.txt
python experiments/dino_vae_hierarchical_diffusion/scripts/train_stage1_autoencoder.py --config experiments/dino_vae_hierarchical_diffusion/config/k3.yaml --max_images 100 --epochs 2
python experiments/dino_vae_hierarchical_diffusion/scripts/extract_latents.py --config experiments/dino_vae_hierarchical_diffusion/config/k3.yaml --max_images 100
python experiments/dino_vae_hierarchical_diffusion/scripts/train_stage2_priors.py --config experiments/dino_vae_hierarchical_diffusion/config/k3.yaml --epochs 2
```

Set `dataset.root` to an image folder. Images are center-cropped and normalized to `[-1,1]`. The scripts automatically use CPU if CUDA is unavailable. Downloads of pretrained weights require network access. Edits produced here are **latent-space interventions/pseudo-counterfactuals**, not SCM counterfactuals.
