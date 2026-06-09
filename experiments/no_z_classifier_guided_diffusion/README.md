# No-Z Classifier-Guided Diffusion Editing POC

This experiment is a proof-of-concept baseline for CelebA/CelebA-HQ facial attribute editing with an unconditional pretrained diffusion model and classifier gradients during DDIM sampling. It intentionally does not train or fine-tune the diffusion model and does not use the prior semantic-latent editing machinery.

## Scientific purpose

The baseline tests whether direct classifier guidance during denoising can edit target facial attributes while preserving non-target attributes. It is intended as an ablation against the semantic-latent editing experiment. Edit strength is controlled only by `guidance_scale`; it should not be compared numerically with any latent-editing strength parameter. For fair reporting, sweep each method's edit strength and compare preservation at matched target-attribute success rates.

## Components

- `config/no_z_classifier_guided_poc.yaml`: default proof-of-concept settings.
- `src/pretrained_attribute_classifier.py`: downloads and wraps the off-the-shelf pretrained CelebA attribute classifier used for guidance/evaluation.
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

The editing driver no longer trains project-local per-attribute classifiers. By default, `classifier.provider: hf_torchvision_state_dict` downloads the pretrained `pymlex/celeba-gan-xai` CelebA attribute classifier from Hugging Face Hub and wraps its shared multi-output ResNet18 as a single-logit module for each target attribute. The configured guidance window uses fractions (`editing.guidance_start_fraction` and `editing.guidance_end_fraction`) that are resolved against `diffusion.num_inference_steps`; with the default 700 DDIM steps, the 30%-90% window becomes steps `[210, 630)`. Lower the DDIM step count only for quick debugging because runtime scales roughly linearly with the number of steps.

For reconstruction-only debugging, edit `diffusion.reconstruction_step_counts` in the YAML. The driver re-runs inversion and denoising at each listed step count and saves extra files such as `*_ddim_reconstruction_500steps.png` under `reconstructions/`, without classifier guidance for those diagnostic images. Keep `diffusion.clip_sample: false` unless you are deliberately testing scheduler clipping artifacts; the final saved images are still clamped to the valid display range.

For guidance debugging, leave `editing.save_guidance_diagnostics: true`. The sampler writes per-timestep CSVs with the classifier loss, target probability, gradient norm, update norm, predicted-x0 range, alpha-cumprod value, and any `skip_reason`. These files should make a no-op edit immediately visible: either `gradient_rms`/`update_rms` is zero, the classifier probability is already saturated, or an unstable high-noise step was skipped. The classifier input uses a straight-through clamp when `editing.clamp_x0: true`, so out-of-range predicted-x0 values are clipped for the classifier forward pass without zeroing the guidance gradient. The normalized guidance gradient is computed with float32 statistics, low-pass smoothed with `editing.gradient_smoothing_kernel`, multiplied by `editing.guidance_step_size`, capped by `editing.max_guidance_update_rms`, skipped until `editing.min_guidance_alpha_cumprod`, and bounded by `editing.max_guided_sample_abs`; without these safeguards, predicted-x0 classifier gradients can turn into adversarial color/texture artifacts instead of semantic edits. Image saving now fails fast on non-finite tensors instead of silently casting them to black/garbage pixels.
