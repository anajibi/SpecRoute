# Hierarchical Latent Diffusion

Self-contained experiment for hierarchical, manipulable DINOv2-derived latents with a frozen SD-VAE image bridge. Configurations cover K=3/5/8 both with free capacity and a fixed 768-dimensional total budget. The fixed-budget K=8 dimensions are strictly decreasing (unlike the non-monotonic illustrative dimensions in the original recipe).

## Install and data

```bash
pip install -r experiments/hierarchical_latent_diffusion/requirements.txt
```

Set `dataset.root` in a config to a directory of recursively discoverable images. DINOv2 and the SD-VAE are downloaded on first use. All runtime artifacts stay below this experiment's `outputs/` directory.

## Full run

```bash
bash experiments/hierarchical_latent_diffusion/run_full_sweep.sh
```

Or invoke the five scripts in order: `train_stage1_encoder_decoder.py`, `extract_latents.py`, `train_stage2_priors.py`, `run_preservation_probe.py`, and `run_counterfactual_probe.py`, each with `--config .../config/k3_levels.yaml` (or another config).

## Smoke test

For a GPU smoke test, copy a config, set `dataset.synthetic: true`, reduce image/latent dimensions for speed, and run:

```bash
python experiments/hierarchical_latent_diffusion/scripts/train_stage1_encoder_decoder.py --config /tmp/hld_smoke.yaml --epochs 2 --max-images 100
python experiments/hierarchical_latent_diffusion/scripts/extract_latents.py --config /tmp/hld_smoke.yaml --max-images 100
python experiments/hierarchical_latent_diffusion/scripts/train_stage2_priors.py --config /tmp/hld_smoke.yaml --epochs 2
python experiments/hierarchical_latent_diffusion/scripts/run_preservation_probe.py --config /tmp/hld_smoke.yaml --max-images 10 --steps 5
```

Unit tests avoid model downloads by injecting tiny frozen backbone doubles:

```bash
pytest -q experiments/hierarchical_latent_diffusion/tests
```

## Notes

* Backbone wrappers remain in evaluation mode even if a parent module calls `.train()`.
* SD-VAE scaling is read from the model config and applied symmetrically.
* Probe CSVs are append-only and written into the selected output directory.
* Attribute classifiers and ArcFace are intentionally not bundled or altered; downstream users can join their frozen-model metrics to probe outputs by `image_id`.
