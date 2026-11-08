# CheXpert label-noise audit

Reader: **torchxrayvision densenet121-res224-all** (no training). Reference: **valid.csv, 234 images, radiologist-labelled**.  
Training sweep: **191,027** frontal images.

## Headline table

| label | AUC vs report (explicit rows) | AUC vs report (blank→0) | AUC vs radiologist | gap (plan's defn) | **gap (comparable)** | confident-disagree | radiologist false-flag | excess | uncertain % | blank % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Effusion | 0.9326 | 0.8145 | 0.8818 | -0.0508 | **0.0673** | 1.03% | 1.28% | -0.25% | 5.20% | 40.37% | noisy |
| Cardiomegaly | 0.9037 | 0.7790 | 0.8199 | -0.0838 | **0.0409** | 1.04% | 1.28% | -0.24% | 3.62% | 79.32% | clean |
| Edema | 0.8678 | 0.7805 | 0.8536 | -0.0142 | **0.0731** | 1.73% | 1.28% | 0.45% | 5.81% | 61.53% | noisy |
| Consolidation | 0.9351 | 0.6966 | 0.8954 | -0.0397 | **0.1988** | 6.56% | 1.28% | 5.28% | 12.42% | 68.39% | unreliable |
| Lung Opacity | 0.8469 | 0.6896 | 0.8787 | 0.0318 | **0.1891** | 3.41% | 1.71% | 1.70% | 2.51% | 47.28% | unreliable |
| Atelectasis | 0.7115 | 0.5976 | 0.8141 | 0.1026 | **0.2165** | 3.48% | 1.28% | 2.20% | 15.10% | 69.36% | unreliable |
| Pneumothorax | 0.7448 | 0.7277 | 0.6477 | -0.0971 | **-0.0800** | 3.51% | 1.71% | — | 1.41% | 64.67% | not auditable — not auditable with XRV (n=8 positives in valid) |
| Lung Lesion | 0.7616 | 0.6133 | 0.1974 | -0.5642 | **-0.4159** | 6.58% | 1.28% | — | 0.67% | 94.65% | not auditable — not auditable with XRV (n=1 positives in valid) |
| Fracture | 0.6805 | 0.5295 | — | — | **—** | 0.28% | 1.28% | — | 0.29% | 94.54% | not auditable — not auditable with XRV (n=0 positives in valid) |
| Enlarged Cardiomediastinum | 0.8453 | 0.5814 | 0.8394 | -0.0059 | **0.2580** | 3.09% | 1.71% | 1.38% | 5.55% | 79.93% | unreliable |

`Support Devices`: no XRV output; your own classifier only.

## Model-error floor (fixed 0.9 / 0.1 thresholds)

Reported because the plan asks for it, and labelled a floor rather than a label-error rate: at these thresholds the metric flags this share of the RADIOLOGIST set, where by construction there is no report-label error to find.

| label | train confident-disagree | radiologist false-flag (= reader error) |
|---|---|---|
| Effusion | 2.51% | 9.40% |
| Cardiomegaly | 1.32% | 2.56% |
| Edema | 1.73% | 3.42% |
| Consolidation | 0.70% | 3.42% |
| Lung Opacity | 0.95% | 3.42% |
| Atelectasis | 0.71% | 8.12% |
| Pneumothorax | 0.01% | 0.00% |
| Lung Lesion | 0.41% | 5.13% |
| Fracture | 0.13% | 0.00% |
| Enlarged Cardiomediastinum | 0.51% | 0.00% |

Calibrated per-label thresholds (hi = 99th percentile of probability among radiologist negatives, lo = 1st percentile among radiologist positives) hold the false-flag rate at ≤1.73%, which is what makes the train-side number interpretable.

## S6 self-consistency

- **1_valid_vs_test_agreement** — N/A. not runnable: the 668-image test set's labels are public but its images are behind Stanford AIMI registration and are not in the Kaggle mirror, which carries train and valid only. The audit therefore has ONE reference set.
- **2_ordering** — **FAIL**. the two orderings should broadly agree; a label high on one and low on the other is flagged rather than averaged away
- **3_radiologist_false_flag** — PASS. fixed thresholds fail this for six of eight labels, which is why they are reported only as a model-error floor
- **4_chex_control** — **FAIL**. densenet121-res224-chex learned these very report labels. If the gap is label noise it should beat -all against the reports and not against the radiologists.

## Age (sanity only; age labels do not come from reports)

MAE **8.37 years**, bias -1.34, bin accuracy 0.706 over 191,024 images. Bin confusion (rows = CSV truth, cols = predicted):

| | <45 | 45–65 | >65 |
|---|---|---|---|
| **<45** | 20890 | 14562 | 157 |
| **45–65** | 3514 | 57911 | 13150 |
| **>65** | 511 | 24258 | 56071 |

