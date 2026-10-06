# Analysis scripts

One-off studies that produced the published CelebA-HQ 256 results. They are kept because the
numbers they generated are cited in the artifacts and in `baselines/AGENT.md`, and several
record findings that are not obvious from the pipeline code alone.

| script | what it established |
|---|---|
| `qual_grids.py` | counterfactual grids across guidance; the source of the "beard leak" observation |
| `t_sweep.py` | reconstruction vs edit quality across DDIM steps; k=11 degrades past T=100, k=1 does not |
| `recon_eval_pretrained.py` | the untuned ffhq256_autoenc transfer floor on CelebA-HQ |
| `guided_inversion.py` | inversion/sampling asymmetry: guided inversion cuts high-frequency excess 78% at g=5 |
| `nullset_xt_variants.py` | the 6 null-set x inverted-latent configurations; the x_T axis is inert |
| `nullset_quantified.py` | predictor-based drift under each null set |
| `verify_null_masking.py` | proves nulling attribute i perturbs only slice i of the concat_film embedding |
| `gsweep_beard_bracket.py` | the g=1.0/1.5 cells that bracketed Beard's optimum below the main grid |
| `reader_f1_eval.py`, `reader_test_eval.py` | reader precision/recall/F1 on held-out data |
| `checkpoint_trajectory.py` | whether more training still helps, holding dropout fixed |
| `guidance_noise_fixes.py` | interval guidance and dynamic thresholding against high-g artefacts |
| `build_artifact.py` | regenerates the published results page from the result JSONs |

These were written against absolute paths in a scratch directory and will need their paths
adjusted. They are committed for provenance, not as a maintained API.
