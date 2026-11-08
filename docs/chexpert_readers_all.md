# CheXpert — three readers, every attribute, train / test / radiologists

Generated 2026-11-08. Row sets: fit 138,727 · select 15,529 · train 154,256 (= fit + select) · test 17,740 · radiologist_val 200.

**The three columns do not measure the same thing.** `train` and `test` score against report-derived CheXpert labels with `-1` folded to 0, so an AUC there is agreement-with-the-report. `radiologists` scores against labels assigned from the images by radiologists, and is the only column the v2 gate is defined on. Independence also differs by reader: ours trained on the fit rows outright, XRV has CheXpert in its training mix, CheXzero saw none of these labels.

`gap` = test − radiologists. Large and positive means the reader tracks the report better than it tracks the finding.


## effusion  ·  *modeled*

| reader | train | test | radiologists | gap | test n_pos | unc folded |
|---|---|---|---|---|---|---|
| ours (best.ckpt) | 0.8835 [0.882,0.885] | 0.8671 [0.862,0.872] | 0.9049 [0.858,0.944] | -0.0378 | 7,670 | 0 |
| ours (11-head) | 0.8822 [0.880,0.884] | 0.8675 [0.863,0.873] | 0.9061 [0.859,0.944] | -0.0386 | 7,670 | 0 |
| XRV `-all` | 0.8160 [0.814,0.818] | 0.8024 [0.796,0.809] | 0.8803 [0.826,0.926] | -0.0779 | 7,670 | 0 |
| CheXzero (zero-shot) | 0.8544 [0.853,0.856] | 0.8418 [0.837,0.848] | 0.9034 [0.856,0.941] | -0.0616 | 7,670 | 0 |

## edema  ·  *modeled*

| reader | train | test | radiologists | gap | test n_pos | unc folded |
|---|---|---|---|---|---|---|
| ours (best.ckpt) | 0.8414 [0.839,0.843] | 0.8262 [0.820,0.832] | 0.9111 [0.861,0.952] | -0.0849 | 4,603 | 1,107 |
| ours (11-head) | 0.8407 [0.838,0.843] | 0.8281 [0.822,0.835] | 0.9134 [0.865,0.952] | -0.0853 | 4,603 | 1,107 |
| XRV `-all` | 0.7641 [0.761,0.766] | 0.7547 [0.748,0.762] | 0.8675 [0.806,0.925] | -0.1128 | 4,603 | 1,107 |
| CheXzero (zero-shot) | 0.7765 [0.774,0.779] | 0.7640 [0.757,0.771] | 0.9096 [0.850,0.961] | -0.1456 | 4,603 | 1,107 |

Test AUC on certain rows only (`-1` dropped, n=16,633): ours (best.ckpt) 0.8455 · ours (11-head) 0.8464 · XRV `-all` 0.7744 · CheXzero (zero-shot) 0.7885

## cardiomegaly  ·  *out*

| reader | train | test | radiologists | gap | test n_pos | unc folded |
|---|---|---|---|---|---|---|
| ours (best.ckpt) | 0.8926 [0.890,0.895] | 0.8813 [0.874,0.889] | 0.8080 [0.738,0.868] | +0.0733 | 2,217 | 0 |
| ours (11-head) | 0.8909 [0.888,0.893] | 0.8800 [0.873,0.887] | 0.8155 [0.748,0.875] | +0.0645 | 2,217 | 0 |
| XRV `-all` | 0.7776 [0.773,0.781] | 0.7805 [0.770,0.790] | 0.8063 [0.741,0.863] | -0.0258 | 2,217 | 0 |
| CheXzero (zero-shot) | 0.8321 [0.829,0.835] | 0.8390 [0.829,0.848] | 0.8544 [0.797,0.904] | -0.0154 | 2,217 | 0 |

## enlarged_cardiomediastinum  ·  *out*

| reader | train | test | radiologists | gap | test n_pos | unc folded |
|---|---|---|---|---|---|---|
| ours (best.ckpt) | — | — | — | — | — | — |
| ours (11-head) | 0.7467 [0.740,0.752] | 0.7000 [0.681,0.718] | 0.5532 [0.470,0.633] | +0.1468 | 853 | 987 |
| XRV `-all` | 0.5835 [0.577,0.589] | 0.5988 [0.579,0.619] | 0.8292 [0.767,0.883] | -0.2304 | 853 | 987 |
| CheXzero (zero-shot) | 0.5742 [0.568,0.580] | 0.5934 [0.573,0.614] | 0.9160 [0.876,0.950] | -0.3226 | 853 | 987 |

Test AUC on certain rows only (`-1` dropped, n=16,753): ours (11-head) 0.7033 · XRV `-all` 0.5995 · CheXzero (zero-shot) 0.5928

## lung_opacity  ·  *eval-only*

| reader | train | test | radiologists | gap | test n_pos | unc folded |
|---|---|---|---|---|---|---|
| ours (best.ckpt) | 0.7289 [0.726,0.732] | 0.7172 [0.710,0.725] | 0.9027 [0.856,0.943] | -0.1855 | 8,876 | 382 |
| ours (11-head) | 0.7281 [0.726,0.731] | 0.7175 [0.711,0.725] | 0.9057 [0.859,0.944] | -0.1882 | 8,876 | 382 |
| XRV `-all` | 0.6909 [0.688,0.694] | 0.6841 [0.677,0.692] | 0.8819 [0.831,0.926] | -0.1978 | 8,876 | 382 |
| CheXzero (zero-shot) | 0.6941 [0.691,0.697] | 0.6924 [0.685,0.700] | 0.8635 [0.807,0.910] | -0.1711 | 8,876 | 382 |

Test AUC on certain rows only (`-1` dropped, n=17,358): ours (best.ckpt) 0.7211 · ours (11-head) 0.7213 · XRV `-all` 0.6863 · CheXzero (zero-shot) 0.6934

## consolidation  ·  *eval-only*

| reader | train | test | radiologists | gap | test n_pos | unc folded |
|---|---|---|---|---|---|---|
| ours (best.ckpt) | 0.7636 [0.760,0.768] | 0.7194 [0.706,0.733] | 0.9126 [0.866,0.950] | -0.1932 | 1,287 | 2,134 |
| ours (11-head) | 0.7588 [0.755,0.763] | 0.7222 [0.709,0.737] | 0.9033 [0.853,0.945] | -0.1811 | 1,287 | 2,134 |
| XRV `-all` | 0.6863 [0.682,0.692] | 0.6910 [0.677,0.705] | 0.8804 [0.827,0.923] | -0.1894 | 1,287 | 2,134 |
| CheXzero (zero-shot) | 0.7010 [0.697,0.707] | 0.6957 [0.681,0.709] | 0.8839 [0.831,0.925] | -0.1882 | 1,287 | 2,134 |

Test AUC on certain rows only (`-1` dropped, n=15,606): ours (best.ckpt) 0.7307 · ours (11-head) 0.7324 · XRV `-all` 0.6995 · CheXzero (zero-shot) 0.7016

## atelectasis  ·  *out*

| reader | train | test | radiologists | gap | test n_pos | unc folded |
|---|---|---|---|---|---|---|
| ours (best.ckpt) | 0.7074 [0.705,0.711] | 0.6831 [0.673,0.694] | 0.7612 [0.690,0.823] | -0.0781 | 2,734 | 2,699 |
| ours (11-head) | 0.7063 [0.704,0.710] | 0.6822 [0.672,0.692] | 0.7529 [0.682,0.817] | -0.0707 | 2,734 | 2,699 |
| XRV `-all` | 0.5740 [0.570,0.577] | 0.5728 [0.563,0.583] | 0.8054 [0.740,0.863] | -0.2326 | 2,734 | 2,699 |
| CheXzero (zero-shot) | 0.6274 [0.624,0.630] | 0.6325 [0.622,0.642] | 0.7845 [0.719,0.842] | -0.1520 | 2,734 | 2,699 |

Test AUC on certain rows only (`-1` dropped, n=15,041): ours (best.ckpt) 0.7066 · ours (11-head) 0.7051 · XRV `-all` 0.5935 · CheXzero (zero-shot) 0.6636

## pneumothorax  ·  *out*

| reader | train | test | radiologists | gap | test n_pos | unc folded |
|---|---|---|---|---|---|---|
| ours (best.ckpt) | 0.8970 [0.895,0.899] | 0.8694 [0.860,0.878] | 0.8110 [0.601,0.972] | +0.0584 | 1,707 | 206 |
| ours (11-head) | 0.8919 [0.889,0.894] | 0.8632 [0.855,0.872] | 0.7668 [0.545,0.917] | +0.0964 | 1,707 | 206 |
| XRV `-all` | 0.7267 [0.723,0.731] | 0.7228 [0.710,0.736] | 0.6830 [0.521,0.853] | +0.0398 | 1,707 | 206 |
| CheXzero (zero-shot) | 0.7652 [0.762,0.769] | 0.7780 [0.765,0.790] | 0.8926 [0.763,0.982] | -0.1146 | 1,707 | 206 |

Test AUC on certain rows only (`-1` dropped, n=17,534): ours (best.ckpt) 0.8718 · ours (11-head) 0.8657 · XRV `-all` 0.7242 · CheXzero (zero-shot) 0.7799

## support_devices  ·  *out*

| reader | train | test | radiologists | gap | test n_pos | unc folded |
|---|---|---|---|---|---|---|
| ours (best.ckpt) | 0.8441 [0.842,0.846] | 0.8397 [0.834,0.845] | 0.8483 [0.793,0.897] | -0.0086 | 10,260 | 94 |
| ours (11-head) | 0.8436 [0.841,0.845] | 0.8393 [0.833,0.845] | 0.8270 [0.767,0.882] | +0.0123 | 10,260 | 94 |
| XRV `-all` | — | — | — | — | — | — |
| CheXzero (zero-shot) | 0.7576 [0.755,0.759] | 0.7562 [0.749,0.763] | 0.7855 [0.718,0.845] | -0.0293 | 10,260 | 94 |

Test AUC on certain rows only (`-1` dropped, n=17,646): ours (best.ckpt) 0.8428 · ours (11-head) 0.8423 · CheXzero (zero-shot) 0.7585

## lung_lesion  ·  *out*

| reader | train | test | radiologists | gap | test n_pos | unc folded |
|---|---|---|---|---|---|---|
| ours (best.ckpt) | — | — | — | — | — | — |
| ours (11-head) | — | — | — | — | — | — |
| XRV `-all` | 0.6153 [0.609,0.623] | 0.6202 [0.596,0.644] | 0.1508† [0.105,0.201] | +0.4694 | 585 | 99 |
| CheXzero (zero-shot) | 0.5116 [0.504,0.520] | 0.5038 [0.478,0.529] | 0.7337† [0.668,0.789] | -0.2299 | 585 | 99 |

Test AUC on certain rows only (`-1` dropped, n=17,641): XRV `-all` 0.6208 · CheXzero (zero-shot) 0.5040

## fracture  ·  *out*

| reader | train | test | radiologists | gap | test n_pos | unc folded |
|---|---|---|---|---|---|---|
| ours (best.ckpt) | — | — | — | — | — | — |
| ours (11-head) | — | — | — | — | — | — |
| XRV `-all` | 0.5276 [0.521,0.533] | 0.5318 [0.511,0.553] | not scoreable | — | 687 | 37 |
| CheXzero (zero-shot) | 0.4416 [0.435,0.448] | 0.4510 [0.428,0.472] | not scoreable | — | 687 | 37 |

Test AUC on certain rows only (`-1` dropped, n=17,703): XRV `-all` 0.5318 · CheXzero (zero-shot) 0.4505

## pneumonia  ·  *out*

| reader | train | test | radiologists | gap | test n_pos | unc folded |
|---|---|---|---|---|---|---|
| ours (best.ckpt) | — | — | — | — | — | — |
| ours (11-head) | — | — | — | — | — | — |
| XRV `-all` | 0.6283 [0.621,0.637] | 0.6457 [0.619,0.671] | 0.8229 [0.700,0.930] | -0.1772 | 469 | 1,494 |
| CheXzero (zero-shot) | 0.6346 [0.625,0.645] | 0.6451 [0.619,0.674] | 0.8906 [0.785,0.963] | -0.2455 | 469 | 1,494 |

Test AUC on certain rows only (`-1` dropped, n=16,246): XRV `-all` 0.6596 · CheXzero (zero-shot) 0.6603

## pleural_other  ·  *out*

| reader | train | test | radiologists | gap | test n_pos | unc folded |
|---|---|---|---|---|---|---|
| ours (best.ckpt) | — | — | — | — | — | — |
| ours (11-head) | — | — | — | — | — | — |
| XRV `-all` | — | — | — | — | — | — |
| CheXzero (zero-shot) | 0.4501 [0.437,0.463] | 0.4874 [0.453,0.522] | 0.9698† [0.944,0.990] | -0.4824 | 264 | 52 |

Test AUC on certain rows only (`-1` dropped, n=17,688): CheXzero (zero-shot) 0.4875

## sex  ·  *modeled*

| reader | train | test | radiologists | gap | test n_pos | unc folded |
|---|---|---|---|---|---|---|
| ours (best.ckpt) | 0.9963 [0.996,0.997] | 0.9964 [0.996,0.997] | 0.9945 [0.986,1.000] | +0.0019 | 10,322 | 0 |
| ours (11-head) | 0.9965 [0.996,0.997] | 0.9963 [0.996,0.997] | 0.9943 [0.984,1.000] | +0.0020 | 10,322 | 0 |
| XRV `-all` | — | — | — | — | — | — |
| CheXzero (zero-shot) | — | — | — | — | — | — |

## ap_pa  ·  *eval-only*

| reader | train | test | radiologists | gap | test n_pos | unc folded |
|---|---|---|---|---|---|---|
| ours (best.ckpt) | 0.9977 [0.997,0.998] | 0.9962 [0.995,0.997] | 1.0000 [1.000,1.000] | -0.0038 | 2,620 | 0 |
| ours (11-head) | 0.9977 [0.998,0.998] | 0.9964 [0.995,0.997] | 1.0000 [1.000,1.000] | -0.0036 | 2,620 | 0 |
| XRV `-all` | — | — | — | — | — | — |
| CheXzero (zero-shot) | — | — | — | — | — | — |

## age_bin  ·  *modeled*

| reader | train | test | radiologists | gap | test n_pos | unc folded |
|---|---|---|---|---|---|---|
| ours (best.ckpt) | 0.9725 | 0.9299 | 0.9523 | -0.0224 | — | — |
| ours (11-head) | 0.9730 | 0.9292 | 0.9554 | -0.0262 | — | — |

---

† fewer than 5 positives in radiologist_val -- the number is not usable. lung_lesion and pleural_other have 1 positive each; fracture has 0 and cannot be scored against radiologists at all. These three stay out of the protocol on those grounds alone, independently of any AUC.

Validation: all 22 numbers this table shares with the frozen `predictor_report.json` (ours10 on test and on radiologist_val, 11 attributes each) reproduce to 4 decimals, and the fit/select/test sub-split reproduces at 138,727 / 15,529 / 17,740. The CheXzero streaming rewrite reproduces all seven previously reported radiologist_val AUCs to 4 decimals.

