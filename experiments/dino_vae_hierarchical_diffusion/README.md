# DINO–VAE Hierarchical Latent Diffusion

An isolated, no-attribute experiment combining a frozen Stable Diffusion VAE, frozen DINOv2 evidence, stochastic top-down K=3/K=5 hierarchies, conditional diffusion priors, and a final latent diffusion decoder.

## Quick start

```bash
pip install -r experiments/dino_vae_hierarchical_diffusion/requirements.txt
python experiments/dino_vae_hierarchical_diffusion/scripts/train_stage1_autoencoder.py --config experiments/dino_vae_hierarchical_diffusion/config/k3.yaml --max_images 100 --epochs 2
python experiments/dino_vae_hierarchical_diffusion/scripts/extract_latents.py --config experiments/dino_vae_hierarchical_diffusion/config/k3.yaml --max_images 100
python experiments/dino_vae_hierarchical_diffusion/scripts/train_stage2_priors.py --config experiments/dino_vae_hierarchical_diffusion/config/k3.yaml --epochs 2
```

Configure CelebA metadata paths, explicit image paths, or a fallback `dataset.root` image folder. Images are resized to the configured square resolution and normalized to `[-1,1]`. The scripts automatically use CPU if CUDA is unavailable. Downloads of pretrained weights require network access. Edits produced here are **latent-space interventions/pseudo-counterfactuals**, not SCM counterfactuals.

## Dataset configuration

The image loader supports official CelebA attribute/partition files, explicit image paths, recursively discovered image folders, and deterministic synthetic images. The default configs use:

```yaml
dataset:
  image_dir: /home/anajibi/HDM/diffae_latent_probe/data/raw_images/celeba-hq
  attr_path: /home/anajibi/HDM/diffae_latent_probe/data/celeba-hq/list_attr_celeba.txt
  partition_path: /home/anajibi/HDM/diffae_latent_probe/data/celeba-hq/list_eval_partition.txt
  split: train
  image_size: 256
stage1:
  batch_size: 32
```

Although attributes are returned by the CelebA dataset for probes and analysis, Stage 1 and Stage 2 do **not** condition the hierarchy, priors, or decoder on them. Set `dataset.synthetic: true` for loader-level tests without image files. Latent extraction writes a batched `latents.pt` cache that can be read by `LatentDataset` during Stage 2.

## Stage 1 completion and checkpointing

`train_stage1_autoencoder.py` is a finite training command: after the requested epochs it saves `outputs/<k>/checkpoints/stage1.pt` and exits. Checkpoint preparation can briefly look idle because all trainable weights must synchronize from GPU to CPU before they are written. The script now prints separate **preparing**, **writing**, and **complete** messages, reports the final checkpoint size/path, and prints the next latent-extraction command. Checkpoints are written atomically so an interrupted save does not leave a partial `stage1.pt`.
