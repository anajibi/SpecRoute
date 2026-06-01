# No-Z Classifier-Guided Diffusion Editing POC

This experiment is a proof-of-concept baseline for CelebA/CelebA-HQ facial attribute editing with an unconditional pretrained diffusion model and classifier gradients during DDIM sampling. It intentionally does not train or fine-tune the diffusion model and does not use the prior semantic-latent editing machinery.

## Scientific purpose

The baseline tests whether direct classifier guidance during denoising can edit target facial attributes while preserving non-target attributes. It is intended as an ablation against the semantic-latent editing experiment. Edit strength is controlled only by `guidance_scale`; it should not be compared numerically with any latent-editing strength parameter. For fair reporting, sweep each method's edit strength and compare preservation at matched target-attribute success rates.

## Components

- `config/no_z_classifier_guided_poc.yaml`: default proof-of-concept settings.
- `scripts/train_attribute_classifier.py`: trains or loads the project CelebA attribute classifier.
- `scripts/run_guided_editing_poc.py`: end-to-end DDIM inversion, classifier-guided editing, prediction, metric, and visualization driver.
- `src/diffusion_backbone.py`: Hugging Face Diffusers loader for `google/ddpm-celebahq-256`; its DDIM schedulers disable intermediate predicted-x0 clipping by default because clipping during inversion/reconstruction is lossy.
- `src/ddim_inversion.py`: deterministic DDIM inversion helpers.
- `src/classifier_guidance.py`: classifier-guided DDIM sampler.
- `src/datasets.py`: CelebA/CelebA-HQ attribute parsing and image datasets.
- `src/evaluation.py`: prediction CSV and preservation metric utilities.
- `src/visualization.py`: image saving and comparison grids.
- `src/utils.py`: config, seeding, path, and tensor helpers.

## Quick start

Update the dataset paths in `config/no_z_classifier_guided_poc.yaml`, then run a small smoke test:

```bash
python experiments/no_z_classifier_guided_diffusion/scripts/run_guided_editing_poc.py \
  --config experiments/no_z_classifier_guided_diffusion/config/no_z_classifier_guided_poc.yaml \
  --num-images 4 \
  --target-attributes Smiling \
  --guidance-scales 0.0 1.0
```

The script writes outputs under `experiments/no_z_classifier_guided_diffusion/outputs/poc` by default, including:

- `edited_image_records.csv`
- `attribute_predictions_original.csv`
- `attribute_predictions_edited.csv`
- `edit_metrics.csv`
- `preservation_summary.csv`
- `images/`, `reconstructions/`, `guidance_diagnostics/`, and `grids/`

Classifier checkpoints are cached separately under `experiments/no_z_classifier_guided_diffusion/outputs/attribute_classifier/<attribute>/` unless `classifier.retrain: true` is set. Each classifier is a single-output binary predictor trained only for the one attribute currently being edited, not the full CelebA attribute set. Validation metrics are written to `outputs/attribute_classifier/<attribute>/validation_metrics.csv` after training. The default DDIM inversion/sampling step counts are intentionally higher than a minimal smoke test to reduce reconstruction artifacts; lower them only for quick debugging.

For reconstruction-only debugging, edit `diffusion.reconstruction_step_counts` in the YAML. The driver re-runs inversion and denoising at each listed step count and saves extra files such as `*_ddim_reconstruction_500steps.png` under `reconstructions/`, without classifier guidance for those diagnostic images. Keep `diffusion.clip_sample: false` unless you are deliberately testing scheduler clipping artifacts; the final saved images are still clamped to the valid display range.

For guidance debugging, leave `editing.save_guidance_diagnostics: true`. The sampler writes per-timestep CSVs with the classifier loss, target probability, gradient norm, update norm, and predicted-x0 range. These files should make a no-op edit immediately visible: either `gradient_rms`/`update_rms` is zero, or the classifier probability is already saturated. The classifier input uses a straight-through clamp when `editing.clamp_x0: true`, so out-of-range predicted-x0 values are clipped for the classifier forward pass without zeroing the guidance gradient.
