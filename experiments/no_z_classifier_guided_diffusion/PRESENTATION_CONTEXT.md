# Presentation Context: `no_z_classifier_guided_diffusion` Branch of Work

## 1. One-slide executive summary

This branch builds a no-latent-editing baseline for facial attribute editing. Instead of using DiffAE semantic latents (`z_sem`) or learned latent directions, it uses a pretrained unconditional diffusion model and applies classifier gradients directly during DDIM denoising. The purpose is to answer: if we do not use DiffAE's semantic latent space at all, can direct classifier guidance edit attributes such as `Smiling`, `Male`, `Young`, and `Eyeglasses` while preserving non-target attributes?

The branch adds a self-contained proof-of-concept under `experiments/no_z_classifier_guided_diffusion/` with:

- A Hugging Face Diffusers unconditional CelebA-HQ DDPM/DDIM backbone (`google/ddpm-celebahq-256`).
- DDIM inversion to map a real image into a terminal noisy latent/sample `x_T`.
- Deterministic DDIM reconstruction from the inverted `x_T`.
- A one-attribute-at-a-time ResNet attribute classifier.
- A classifier-guided DDIM sampler that modifies the denoising trajectory using gradients from the classifier.
- Evaluation CSVs for target success, target probability change, non-target preservation, non-target flip rate, image MSE, and reconstruction-to-edit deltas.
- Visual grids and optional timestep-level guidance diagnostics.

## 2. Scientific purpose

This is an ablation/baseline against the DiffAE latent-editing branch. The key contrast is:

| Branch | Editing mechanism | Representation used for editing |
|---|---|---|
| `diffae_latent_probe` | Add a learned direction to `z_sem` and decode with fixed `x_T`. | DiffAE semantic latent space. |
| `no_z_classifier_guided_diffusion` | Apply classifier gradients during DDIM denoising. | No explicit semantic latent; edits occur directly in diffusion sample space / predicted image space. |

The scientific question is whether direct classifier guidance can achieve attribute edits without DiffAE's semantic representation, and how much non-target drift or image artifacting it causes.

A key reporting principle documented in the README is that `guidance_scale` is not numerically comparable to DiffAE's `alpha` edit strength. For fair comparison, each method should sweep its own edit strength and compare preservation at matched target-attribute success rates.

## 3. Repository layout introduced by this branch

All code lives under:

```text
experiments/no_z_classifier_guided_diffusion/
```

Important files:

- `README.md`: explains the proof-of-concept, quick start, components, output files, and guidance-debugging notes.
- `requirements.txt`: additional dependencies for the POC.
- `config/no_z_classifier_guided_poc.yaml`: default experiment configuration.
- `scripts/train_attribute_classifier.py`: trains or loads a binary classifier for a single target attribute.
- `scripts/run_guided_editing_poc.py`: full end-to-end driver: load data, load backbone, invert images, reconstruct, run guided edits, predict attributes, compute metrics, and save visualizations.
- `src/datasets.py`: CelebA attribute parser, partition parser, dataset classes, subset helper, and CSV writer.
- `src/diffusion_backbone.py`: loads the unconditional CelebA-HQ diffusion model and builds DDIM schedulers.
- `src/ddim_inversion.py`: deterministic DDIM inversion and reconstruction helpers.
- `src/classifier_guidance.py`: core classifier-guided DDIM sampling implementation.
- `src/evaluation.py`: attribute prediction and edit/preservation metric computation.
- `src/visualization.py`: original/reconstruction/edit comparison grids.
- `src/utils.py`: YAML loading, seed/device handling, path setup, tensor/image conversion, finite tensor validation, and autocast helper.
- `tests/test_classifier_guidance.py`: unit tests for guidance math/safeguards.
- `tests/test_image_utils.py`: unit tests for image saving and validation helpers.

## 4. Default experiment configuration

The default YAML config is `config/no_z_classifier_guided_poc.yaml`.

### 4.1 Dataset

The config points to the same local CelebA-HQ-style dataset used by the DiffAE branch:

- Image root: `/home/anajibi/HDM/diffae_latent_probe/data/raw_images/celeba-hq`
- Attribute file: `/home/anajibi/HDM/diffae_latent_probe/data/celeba-hq/list_attr_celeba.txt`
- Partition file: `/home/anajibi/HDM/diffae_latent_probe/data/celeba-hq/list_eval_partition.txt`
- Split: `test`
- Image size: `256`

The default run uses:

- Seed: `42`
- Number of images: `16`
- Batch size: `4`
- Device: `cuda` if available
- Output root: `experiments/no_z_classifier_guided_diffusion/outputs/poc`

### 4.2 Diffusion backbone

The branch uses:

- Model ID: `google/ddpm-celebahq-256`
- Inference steps: `700`
- Inversion steps: `700`
- Reconstruction diagnostic step counts: `[500]`
- FP16 and AMP enabled by default on CUDA.

A very important implementation detail is `clip_sample: false`. Intermediate DDIM predicted-`x0` clipping is disabled because clipping during inversion/reconstruction was observed/documented as lossy and capable of causing severe reconstruction artifacts. The final saved images are still clamped to the displayable range.

### 4.3 Classifier

The branch uses a ResNet attribute classifier from the `diffae_latent_probe` code:

- Backbone: `resnet18`
- Pretrained: `false`
- Epochs: `15`
- Learning rate: `0.001`
- Batch size: `64`
- Retrain: `false`

Each classifier is a **single-output binary predictor for one edited attribute**, not a single multi-output classifier for all CelebA attributes. Checkpoints are cached under:

```text
experiments/no_z_classifier_guided_diffusion/outputs/attribute_classifier/<attribute>/
```

The cached checkpoint is reused only if its stored attribute list matches the currently requested target attribute. Validation metrics are written to `validation_metrics.csv`.

### 4.4 Editing targets and guidance parameters

Default target attributes:

- `Smiling` -> target value `1`
- `Male` -> target value `1`
- `Young` -> target value `0`
- `Eyeglasses` -> target value `0`

Default guidance scales:

- `0.0`
- `0.1`
- `0.25`
- `0.5`
- `1.0`

Guidance is applied over a controlled denoising window:

- `guidance_start_step: 50`
- `guidance_end_step: 450`
- `num_guidance_steps_per_timestep: 2`

Other safeguards:

- `guidance_step_size: 0.005`
- `max_guidance_update_rms: 0.02`
- `gradient_smoothing_kernel: 7`
- `max_guided_sample_abs: 4.0`
- `min_guidance_alpha_cumprod: 0.01`
- `skip_nonfinite_guidance: true`
- `clamp_x0: true`
- `guidance_on_x0_pred: true`

The README explains the motivation: raw classifier gradients can become adversarial color/texture perturbations, especially at very noisy or very late pixel-level timesteps. The branch therefore normalizes, smooths, caps, windows, and finite-checks the guidance updates.

## 5. End-to-end experiment flow

The main run command is:

```bash
python experiments/no_z_classifier_guided_diffusion/scripts/run_guided_editing_poc.py \
  --config experiments/no_z_classifier_guided_diffusion/config/no_z_classifier_guided_poc.yaml
```

A small smoke-test command from the README is:

```bash
python experiments/no_z_classifier_guided_diffusion/scripts/run_guided_editing_poc.py \
  --config experiments/no_z_classifier_guided_diffusion/config/no_z_classifier_guided_poc.yaml \
  --num-images 4 \
  --target-attributes Smiling \
  --guidance-scales 0.0 1.0
```

The driver performs these steps:

1. **Load and override config**
   - Reads YAML.
   - Applies CLI overrides for number of images, target attributes, guidance scales, and output root.
   - Sets seeds and device.

2. **Load dataset subset**
   - Parses CelebA attributes and partitions.
   - Selects the configured split and number of images.
   - Produces tensors normalized for the diffusion model.

3. **Load diffusion backbone**
   - Loads `google/ddpm-celebahq-256` via Diffusers.
   - Builds a DDIM scheduler for forward denoising.
   - Builds a DDIM inverse scheduler for image inversion.
   - Disables scheduler clipping when configured.

4. **For each target attribute**
   - Trains or loads the one-attribute classifier.
   - Moves it to the selected device in eval mode.

5. **For each image**
   - Saves/copies the original image.
   - DDIM-inverts the image into `x_T`.
   - DDIM-reconstructs the image without guidance to establish a reconstruction baseline.
   - Optionally saves reconstruction-only diagnostic images at additional step counts.

6. **For each guidance scale**
   - Starts from the same inverted `x_T`.
   - Runs classifier-guided DDIM denoising.
   - Saves the edited image.
   - Records image ID, paths, target attribute, target value, guidance scale, reconstruction-to-edit MSE, max absolute delta, and step counts.
   - Optionally writes per-timestep guidance diagnostics.

7. **Generate visual grids**
   - Saves side-by-side grids of original, reconstruction, and guided edits across guidance scales.

8. **Predict attributes and compute metrics**
   - Runs classifier prediction CSVs for originals and edits.
   - Computes edit metrics and preservation summaries.

## 6. DDIM inversion and reconstruction

The inversion module contains two concepts:

- `ddim_invert`: maps a real normalized image to a terminal noisy DDIM sample `x_T` using the inverse scheduler.
- `ddim_reconstruct`: maps `x_T` back to an image using the standard scheduler with no classifier guidance.

The reconstruction is important because edited outputs should be compared not only to the original image, but also to the model's own reconstruction. If inversion/reconstruction quality is poor, attribute edits cannot be interpreted cleanly.

The branch added reconstruction diagnostics because early experiments showed that bad reconstructions or scheduler clipping can make editing quality look worse than the guidance method itself.

## 7. Classifier-guided DDIM sampling details

The core sampler is `classifier_guided_ddim_sample`.

For each DDIM timestep:

1. Decide whether guidance should run at this step:
   - Guidance scale must be greater than zero.
   - Step index must be inside `[guidance_start_step, guidance_end_step)`.
   - Number of guidance inner steps must be positive.
   - Alpha cumulative product must be above `min_guidance_alpha_cumprod`.

2. If guidance is active, repeat for `num_guidance_steps_per_timestep` inner steps:
   - Enable gradients on the current sample.
   - Predict noise with the diffusion UNet.
   - Convert the current sample/noise prediction into predicted clean image `x0` if `guidance_on_x0_pred: true`.
   - Clamp the classifier input with a straight-through clamp if `clamp_x0: true`, so the classifier sees valid image-like values but gradients still flow.
   - Compute the classifier logit for the target attribute.
   - Compute binary cross-entropy loss against the desired target value.
   - Backpropagate loss with respect to the current DDIM sample.
   - Smooth the gradient spatially.
   - Normalize the gradient to unit RMS.
   - Scale it by `guidance_scale * guidance_step_size`.
   - Cap update RMS with `max_guidance_update_rms`.
   - Subtract the update from the current sample.
   - Clamp sample magnitude to `max_guided_sample_abs` if configured.
   - Record diagnostics if requested.

3. Run the ordinary DDIM scheduler step using the UNet noise prediction.

4. At the end, clamp the returned sample to `[-1, 1]` for image saving.

The gradient update is a descent step on the target binary cross-entropy. For target value `1`, it tries to increase the classifier's probability; for target value `0`, it tries to decrease it.

## 8. Guidance diagnostics

When `editing.save_guidance_diagnostics: true`, each guided image can get a CSV containing timestep-level rows. Diagnostics include:

- Step index and timestep.
- Inner guidance step index.
- Classifier loss.
- Target probability.
- Gradient RMS.
- Update RMS.
- Predicted-`x0` range.
- Alpha cumulative product.
- Skip reason, if guidance was skipped.

These diagnostics were added because early no-op or artifacting cases can be hard to debug from final images alone. A failed/no-op edit should become visible as one of these cases:

- Gradient RMS is zero or non-finite.
- Update RMS is capped to a tiny value.
- Classifier probability is already saturated.
- High-noise steps are skipped by alpha-cumprod gating.
- Non-finite loss or gradient causes guidance to be skipped.

## 9. Evaluation outputs

The default output root is:

```text
experiments/no_z_classifier_guided_diffusion/outputs/poc
```

Expected output files include:

- `edited_image_records.csv`: one row per edited image, including original/reconstruction/edited paths and guidance settings.
- `attribute_predictions_original.csv`: classifier probabilities for original images.
- `attribute_predictions_edited.csv`: classifier probabilities for edited images.
- `edit_metrics.csv`: per-image edit and preservation rows.
- `preservation_summary.csv`: grouped metrics by target attribute, target value, and guidance scale.
- `images/`: edited images.
- `reconstructions/`: DDIM reconstructions and optional reconstruction-step diagnostics.
- `guidance_diagnostics/`: optional per-timestep CSVs.
- `grids/`: side-by-side visual summaries.

Per-image metrics include:

- `target_success`: whether the target probability moved in the desired direction.
- `target_prob_original`
- `target_prob_edited`
- `target_prob_delta`
- `non_target_mean_abs_delta`
- `non_target_flip_rate`
- `image_mse`, when enabled.
- Reconstruction-to-edited MSE and max absolute delta in `edited_image_records.csv`.

Grouped summary metrics include:

- Number of images.
- Target success rate.
- Mean target probability delta.
- Mean non-target absolute probability delta.
- Mean non-target flip rate.
- Mean image MSE, if enabled.

## 10. What happened during development / experimental findings to mention carefully

The git history and README comments indicate this branch went through several rounds of debugging because direct classifier guidance was difficult to make useful:

- Earlier runs produced edits that did not work or were visually unusable.
- Reconstruction quality was improved by increasing DDIM inversion/sampling step counts.
- Intermediate scheduler clipping was identified as harmful for inversion/reconstruction, so the config keeps `clip_sample: false`.
- Guidance gradients needed normalization, smoothing, step-size reduction, RMS caps, denoising-window limits, alpha-cumprod gating, and finite checks.
- Even with these safeguards, the branch should be presented as a proof-of-concept/baseline rather than the final preferred method.

The fair interpretation is:

- The no-Z branch is valuable because it tests whether editing can be achieved without a semantic latent representation.
- It highlights the instability and artifact risk of direct classifier guidance.
- It provides a baseline that can be swept over guidance strength and compared with DiffAE latent edits at matched target success rates.

## 11. How to explain this branch in the presentation

A good presentation flow for this branch is:

1. **Introduce the baseline.** "What if we remove DiffAE's `z_sem` entirely and edit directly with classifier gradients during diffusion sampling?"
2. **Show the method diagram.** Real image -> DDIM inversion -> `x_T` -> guided DDIM sampling -> edited image.
3. **Explain classifier guidance.** The classifier tells the sampler how to change the current predicted image to increase/decrease a target attribute probability.
4. **Show the safeguards.** Direct gradients are unstable, so the implementation windows, normalizes, smooths, caps, clamps, and diagnoses them.
5. **Show outputs.** Original/reconstruction/edit grids across guidance scales.
6. **Show metrics.** Target success rate versus non-target preservation and image MSE.
7. **Position relative to DiffAE.** This is the no-`z_sem` control condition; it tests whether the semantic latent representation is actually helping.

## 12. Limitations and caveats

- The method depends heavily on classifier quality. If the classifier learns shortcuts, guidance can exploit them as adversarial artifacts.
- DDIM inversion quality matters. Poor reconstructions make edits hard to interpret.
- `guidance_scale` is not directly comparable to DiffAE edit `alpha`.
- Direct classifier guidance may change low-level color/texture instead of high-level semantics; this is why the branch added smoothing, RMS caps, and timestep windows.
- The model is unconditional and pretrained; no diffusion fine-tuning is performed.
- Classifiers are trained one attribute at a time, so multi-attribute interactions are measured only through separately predicted probabilities and preservation metrics.
- Data paths are machine-local and must be changed for another environment.

## 13. Short slide-ready phrasing

> In `no_z_classifier_guided_diffusion`, I built a no-`z_sem` diffusion-editing baseline. I invert each real CelebA-HQ image with DDIM, reconstruct it with an unconditional pretrained CelebA-HQ diffusion model, and then rerun DDIM sampling with gradients from a target-attribute classifier. The branch saves original/reconstruction/edit grids across guidance scales and computes target success and non-target preservation metrics. It is mainly a proof-of-concept baseline showing both the potential and instability of direct classifier-guided diffusion editing without a semantic latent space.
