# Presentation Context: `diffae_latent_probe` Branch of Work

## 1. One-slide executive summary

This branch builds a DiffAE-centered experimental pipeline for studying how much information is carried by DiffAE's two latent components and whether facial attributes can be edited by moving only the semantic latent `z_sem` while holding the stochastic/noise latent `x_T` fixed. The work is framed as a pseudo-counterfactual study: the edited images are not causal counterfactuals in the formal sense, but they are controlled latent-space interventions that ask, "what image does the pretrained DiffAE decoder produce if we keep this image's stochastic code fixed and move its semantic code along a learned attribute direction?"

The branch adds:

- A reusable `diffae_tools` utility package for loading DiffAE checkpoints, preparing images, storing latent bundles, training lightweight probes, computing metrics, and building visualizations.
- A CelebA/CelebA-HQ experiment pipeline that encodes images into `z_sem` and `x_T`, runs component ablations, learns attribute directions in `z_sem`, generates pseudo-counterfactual edits, evaluates target-edit success and non-target preservation, and saves grids/CSV summaries.
- Metric code for reconstruction fidelity, high-frequency residuals, target success, non-target attribute preservation, and classifier-based attribute prediction.
- Scripted stages for setup checking, image preparation, dataset encoding, reconstruction, latent swapping, component ablation, and the main end-to-end CelebA pseudo-counterfactual experiment.

## 2. Scientific question and motivation

The main scientific goal is to understand the division of labor inside DiffAE's latent representation:

- `z_sem` is expected to encode semantic/high-level information such as identity, face structure, and attributes.
- `x_T` is expected to encode stochastic or residual appearance information needed for reconstruction, including fine details and texture.

The branch investigates three related questions:

1. **Component information:** What happens if the decoder gets the correct `z_sem` but random or averaged `x_T`, or the correct `x_T` but averaged/zero/mismatched `z_sem`?
2. **Editability:** Can a linear classifier direction in `z_sem` be used as an attribute edit vector for attributes such as `Young`, `Male`, `Eyeglasses`, and `Smiling`?
3. **Preservation:** When one attribute is edited, how much do other attributes change, and can semantic exclusion groups avoid unfairly penalizing correlated attributes?

The intended presentation narrative is that this branch is the structured latent-intervention baseline: it uses the pretrained DiffAE representation, performs edits in `z_sem`, and uses fixed or controlled `x_T` to isolate what changes.

## 3. Repository layout introduced by this branch

### 3.1 `diffae_latent_probe/diffae_tools/`

This package contains reusable utilities for the DiffAE latent-probe workflow:

- `config_io.py`: YAML loading, path resolution, output-directory creation, git hash capture, and temporary addition of the upstream DiffAE repo to `sys.path`.
- `image_io.py`: image discovery, image-ID sanitization, PIL/tensor conversion, center-crop resize, saving tensors, optional official FFHQ alignment, and fallback alignment.
- `latent_codec.py`: save/load helpers for semantic and stochastic latent arrays, latent metadata, and latent feature flattening for probes.
- `latent_dataset.py`: CSV-backed label dataset utilities that infer label columns, subset labels by image IDs, produce label matrices, and split data.
- `metrics.py`: binary classification and regression metric summaries for probe outputs.
- `model_loader.py`: wrapper around the official DiffAE repository/checkpoint that instantiates the upstream model, loads checkpoint weights, exposes semantic encoding, stochastic encoding, decoding, and reconstruction.
- `plotting.py`: plots for attribute probe results and reconstruction/swap panels.
- `probe_models.py`: train/test splits, latent feature construction, binary/regression probes, and probe-suite summaries over semantic/stochastic latent variants.

### 3.2 `diffae_latent_probe/src/`

This is the newer experiment implementation layer:

- `src/models/latent_bundle.py`: a dataclass storing `image_ids`, `z_sem`, `x_t`, and metadata, with save/load/subset helpers.
- `src/models/diffae_wrapper.py`: thin adapter exposing `encode_semantic`, `encode_stochastic`, and `decode` calls against a DiffAE model implementation.
- `src/models/attribute_classifier.py`: ResNet-based multi-label attribute classifier training and prediction helpers. It is reused by the no-Z branch as well.
- `src/experiments/component_ablation.py`: component ablation driver that decodes controlled combinations of `z_sem` and `x_T` and logs reconstruction/texture metrics.
- `src/experiments/attribute_directions.py`: logistic-regression training over `z_sem` to learn linear attribute directions.
- `src/experiments/pseudo_counterfactuals.py`: generates edits by adding scaled directions to `z_sem` and decoding with either fixed real `x_T` or Gaussian `x_T`.
- `src/experiments/preservation_analysis.py`: compares original and edited attribute predictions to estimate target success, target flips, non-target probability drift, non-target flip rate, and preservation accuracy.
- `src/experiments/other_attribute_classifier.py`: trains or loads a classifier for non-target attributes so preservation can be measured beyond only the directly edited targets.
- `src/metrics/`: MSE, SSIM, LPIPS wrapper, high-frequency filtering, target success, target flip, and non-target preservation metrics.
- `src/visualization/`: labeled edit grids, failure-case grids, high-pass/residual visualization helpers, and general grid creation.
- `src/utils/`: YAML IO, logging setup, directory helpers, and random seed control.

### 3.3 `diffae_latent_probe/scripts/`

The script folder provides smaller pipeline stages:

- `00_check_setup.py`: environment/setup smoke check.
- `01_prepare_images.py`: prepares/crops/aligns input images for DiffAE.
- `02_encode_dataset.py`: encodes a dataset into `z_sem` and `x_T` latent bundles.
- `03_reconstruct_dataset.py`: reconstructs images from saved latents.
- `04_swap_latents.py`: swaps semantic/stochastic latents between images for qualitative analysis.
- `run_component_ablation.py`: standalone component-ablation entry point.

### 3.4 `diffae_latent_probe/experiments/`

This contains the main presentation-relevant experiment:

- `experiments/run_celeba_pseudo_counterfactuals.py`: orchestrates the full CelebA/CelebA-HQ pipeline.
- `experiments/README.md`: quick-start notes for the CelebA pseudo-counterfactual experiment.
- `experiments/ablation_terms.md` and `experiments/ABlation_terms.md`: notes about ablation terminology.

## 4. Main experiment: CelebA pseudo-counterfactuals

The core experiment is configured by `configs/experiments/celeba_pseudo_counterfactuals.yaml` and run with:

```bash
python experiments/run_celeba_pseudo_counterfactuals.py \
  --config configs/experiments/celeba_pseudo_counterfactuals.yaml
```

### 4.1 Dataset and model configuration

The current config targets CelebA-HQ-style face images:

- Image root: `/home/anajibi/HDM/diffae_latent_probe/data/raw_images/celeba-hq`
- Attribute labels: `/home/anajibi/HDM/diffae_latent_probe/data/celeba-hq/list_attr_celeba.txt`
- Split file: `/home/anajibi/HDM/diffae_latent_probe/data/celeba-hq/list_eval_partition.txt`
- Split: `test`
- Image size: `256`
- Number of images: `500`
- Batch size: `8`
- Seed: `0`

The DiffAE model points to the upstream DiffAE repo and checkpoint:

- Repo root: `/home/anajibi/HDM/diffae_upstream`
- Checkpoint: `/home/anajibi/HDM/diffae_upstream/checkpoints/ffhq256_autoenc/last.ckpt`

The target attributes are:

- `Young`
- `Male`
- `Eyeglasses`
- `Smiling`

The edit scales are:

- `-1`
- `-0.5`
- `0`
- `0.5`
- `1`

Attribute directions are trained with logistic regression, balanced class weights, and normalized direction vectors.

### 4.2 End-to-end pipeline steps

The experiment driver is designed to run these steps:

1. **Initialize reproducibility and logging**
   - Reads YAML config.
   - Creates output directories.
   - Sets seed.
   - Writes experiment summaries and metadata.

2. **Load CelebA/CelebA-HQ data**
   - Loads images, attribute labels, and split definitions.
   - Optionally limits the experiment to the configured subset size.
   - Computes or filters attribute prevalence so very rare or very common attributes can be excluded from preservation analysis.

3. **Load DiffAE**
   - Wraps the pretrained DiffAE model.
   - Uses the wrapper to call semantic encoding, stochastic encoding, and decoding.

4. **Encode and cache latents**
   - Produces a latent bundle containing:
     - `image_ids`
     - `z_sem`
     - `x_t`
     - metadata
   - Saves the latent bundle under the configured latent output directory.
   - Reuses cached latents when `latents.recompute: false`.

5. **Component ablation**
   - Tests what the decoder reconstructs under different combinations of semantic and stochastic components.
   - Saves per-setting reconstructions, grids, optional per-image metrics, and summary metrics.

6. **Learn attribute directions**
   - For each target attribute, trains a logistic regression classifier using `z_sem` as the feature vector.
   - The classifier coefficient vector becomes the edit direction.
   - Saves `directions.pt` and `classifier_metrics.csv`.

7. **Generate pseudo-counterfactual edits**
   - For each image, each target attribute, and each alpha value, edits:
     - `z_edit = z_sem + alpha * w_attribute`
   - Decodes with either the original/fixed `x_T` or, for an ablation, Gaussian `x_T`.
   - Saves edited images, random-direction controls, grids, and a CSV manifest.

8. **Train/evaluate attribute classifiers**
   - Trains or loads target and other-attribute classifiers.
   - Uses predictions on original and edited images to quantify edit success and preservation.

9. **Preservation analysis**
   - Computes target success, target flip rate, non-target mean absolute probability change, non-target flip rate, and preservation accuracy.
   - Applies prevalence filtering and optional semantic exclusion groups so attributes that are conceptually tied to the edit can be excluded from preservation scoring.

10. **Write outputs for presentation and debugging**
    - Saves summary Markdown/JSON/CSV files.
    - Saves image grids for component ablations, attribute edits, high-frequency residuals, and failure cases.

## 5. Component ablation details

The component ablation is central to the presentation because it explains what each DiffAE latent part appears to control. For each image, the decoder is run under several settings:

| Setting | Decoder input | Purpose |
|---|---|---|
| `full` | original `z_sem`, original `x_T` | Best available reconstruction baseline. |
| `z_only` | original `z_sem`, random Gaussian `x_T` | Tests whether semantics alone can create a plausible face and whether details vanish/change. |
| `z_only_marginal_avg` | original `z_sem`, multiple random `x_T` samples averaged | Estimates the marginal image implied by semantics after stochastic variation is averaged out. |
| `xt_only_mean` | mean `z_sem`, original `x_T` | Tests how much image detail or identity remains when semantic content is collapsed to the dataset mean. |
| `xt_only_zero` | zero `z_sem`, original `x_T` | Similar to mean semantic ablation, but with a zero semantic vector. |
| `z_swap` / `xT-mismatch` | mismatched `z_sem`, original `x_T` | Tests what happens when another image's semantic code is paired with this image's stochastic code. |
| `xt_swap` | original `z_sem`, another image's `x_T` | Tests how stochastic details transfer while semantics stay fixed. |

Metrics for these ablations include:

- LPIPS, if enabled and available.
- SSIM, with torchmetrics when available or CPU fallback logic.
- MSE in image space.
- High-frequency MSE and high-frequency L1 after Gaussian high-pass filtering.

Visual outputs include per-image grids with columns such as `original`, `full`, `z-only`, `z-only avg`, `xT-mean`, `xT-zero`, `xT-mismatch`, and `xT-swap`. High-frequency grids compare residual detail in the original and reconstructed variants.

## 6. Attribute-direction learning

Attribute directions are learned using logistic regression over the semantic latent vectors:

1. Extract `z_sem` for the image subset.
2. For each attribute, take binary labels from the CelebA attributes table.
3. Split into train/validation with stratification.
4. Train `LogisticRegression(max_iter=200, class_weight="balanced")`.
5. Use the learned coefficient vector `w` as the edit direction.
6. Optionally normalize `w` to unit norm.
7. Save validation accuracy, balanced accuracy, AUROC, and class prevalence.

This supports a clear presentation point: the edit direction is not trained by fine-tuning the generator; it is a linear probe direction in the pretrained DiffAE semantic latent space.

## 7. Pseudo-counterfactual edit generation

Pseudo-counterfactuals are produced by moving `z_sem` along a learned direction:

```text
z_edit = z_sem + alpha * w_attribute
image_edit = DiffAE.decode(z_edit, x_T_fixed)
```

The important design choice is that `x_T` can be held fixed. That gives a controlled intervention where stochastic reconstruction details are intended to stay as constant as possible while the semantic code changes.

The branch also supports a Gaussian `x_T` ablation, controlled by `pseudo_counterfactuals.run_gaussian_xt_ablation: true`, to test whether edit behavior depends on using the original stochastic code versus random stochastic samples.

For each target attribute and alpha value, the pipeline can save:

- Edited image files.
- Random-direction controls using a random vector instead of the learned attribute direction.
- Attribute-level grids across alpha values.
- A records CSV mapping image IDs, attributes, alphas, edit/control kind, and image paths.

## 8. Preservation and evaluation methodology

The preservation analysis compares classifier predictions on original images and edited images.

For each edit:

- **Target success score:** whether the target attribute probability moved in the expected direction.
- **Target flip rate:** whether the target classifier crosses the 0.5 decision threshold.
- **Non-target mean absolute change:** average absolute change in predicted probabilities for attributes that should be preserved.
- **Non-target flip rate:** fraction of non-target attributes whose binary prediction flips.
- **Preservation accuracy:** `1 - non_target_flip_rate`.

The config filters attributes by prevalence:

- `prevalence_min: 0.10`
- `prevalence_max: 0.90`

This prevents extremely rare/common attributes from dominating preservation metrics. It also enables semantic exclusion groups, so directly entangled attributes can be excluded when evaluating preservation for a target edit. For example, an edit to `Male` should not necessarily be penalized for changing highly gender-correlated attributes if those are in the semantic exclusion group.

## 9. Output structure

The configured root is:

```text
outputs/celeba_pseudo_counterfactuals/
```

Expected subfolders and files include:

- `latents/` or configured latent output directory: cached `z_sem`, `x_t`, image IDs, and metadata.
- `component_ablation/`: reconstruction grids, per-image images, summary grids, and metric CSVs.
- `attribute_directions/`: `directions.pt` and logistic-regression classifier metrics.
- `pseudo_counterfactuals/`: edited images, random controls, grids, manifests, and optional Gaussian-`x_T` ablation outputs.
- `attribute_predictions_original.csv`: classifier probabilities on originals.
- `attribute_predictions_edited.csv`: classifier probabilities on edited images.
- `preservation_metrics.csv`: per-image/per-edit preservation rows.
- `preservation_summary_all_filtered.csv`: grouped summary after prevalence filtering.
- `preservation_summary_semantic_excluded.csv`: summary when semantic exclusion groups are active.
- Experiment summary files recording dataset/model/config/step status.

## 10. How to explain this branch in the presentation

A good presentation flow for this branch is:

1. **Start with DiffAE decomposition.** Explain that DiffAE has a semantic latent `z_sem` and stochastic latent `x_T`.
2. **Show component ablation grid.** Demonstrate full reconstruction, semantic-only, stochastic-only, and swapped-latent cases.
3. **Show learned directions.** Explain that each attribute direction is a linear classifier normal vector in `z_sem`.
4. **Show edit grids.** For one image and one attribute, show the alpha sweep from negative to positive values.
5. **Show preservation metrics.** Plot target success versus non-target drift or non-target flip rate.
6. **State limitations.** These are pseudo-counterfactuals, not causal interventions. They depend on classifier quality, DiffAE reconstruction quality, and whether linear directions are disentangled.

## 11. What was learned / likely interpretation

The likely interpretation is:

- DiffAE's full reconstruction requires both `z_sem` and `x_T`.
- `z_sem` carries a large amount of semantic structure and can be probed linearly for face attributes.
- `x_T` contributes reconstruction-specific detail and texture; swapping or randomizing it is a way to test detail preservation.
- Linear semantic edits are interpretable and controlled, but they may still change correlated attributes and may not perfectly preserve identity.
- Preservation analysis must be reported alongside target edit success because a method that changes the target strongly may also cause broad semantic drift.

## 12. Important implementation caveats for the presentation LLM

- Use the phrase **pseudo-counterfactual** rather than **causal counterfactual** unless explicitly describing limitations.
- The branch currently documents and implements a rich pipeline, but in the checked-out code the main experiment imports `src.data.celeba_dataset` and `src.data.celeba_attributes`, while no `diffae_latent_probe/src/data/` directory is present in this checkout. If the presentation mentions runnable status, say that the branch includes the experimental scaffold and analysis code, but the current checkout appears to be missing the `src.data` dataset/attribute modules required by the main driver.
- The DiffAE checkpoint and data paths are absolute machine-local paths and must be updated on a new machine.
- The linear directions are trained from CelebA labels and therefore inherit label noise and classifier/probe bias.
- Attribute-preservation metrics are classifier-based; they measure prediction changes, not guaranteed human-perceived semantic preservation.
- The random-direction controls are useful for arguing that learned directions do something attribute-specific beyond arbitrary movement in latent space.

## 13. Short slide-ready phrasing

> In `diffae_latent_probe`, I built a DiffAE latent-intervention pipeline. I encode CelebA-HQ images into semantic latents `z_sem` and stochastic latents `x_T`, run ablations to isolate what each component controls, learn linear facial-attribute directions in `z_sem`, and decode pseudo-counterfactual edits while keeping `x_T` fixed. The outputs include edit grids, component-ablation grids, reconstruction/high-frequency metrics, target-edit success, and non-target attribute preservation summaries.
