# c3di-v2-validated-20260827

Both full-length Causal3DIdent runs finished and evaluated. This is the first tag in this
project where the headline claim is backed by a measured, interval-bounded result rather
than a training curve.

## The result

k=11 beats k=1 on every modelled attribute, at a third of the guidance. Measured on a fixed
256-image test cohort, each model at its own best guidance strength, EMA weights, T=100,
11-point strength grid, attributes read back with the ConvNeXt-Tiny predictors:

| intervention | k=1 best g | k=1            | k=11 best g | k=11            | k=11 advantage |
|--------------|-----------:|----------------|------------:|-----------------|----------------|
| do(class)    |          8 | 88.28%         |           3 | **100.00%**     | +11.7 pts      |
| do(pos_spl)  |          2 | 0.0394         |           2 | **0.0214**      | 1.84x          |
| do(pos_obj)  |          8 | 0.1933         |           2 | **0.0300**      | 6.44x          |
| do(rot_obj)  |          8 | 0.1751         |           3 | **0.0790**      | 2.22x          |

`class` is accuracy (higher better); the rest are MAE against the SCM's counterfactual
target (lower better). k=11 wins all 20 cells of the strength grid, not only at its optimum,
and its collateral damage on the three unmodelled hues is lower at every strength.

Reconstruction floors are effectively identical (pos_obj 0.0114 vs 0.0117), so the gap is
about responding to interventions, not reconstruction quality.

## Caveat that travels with this tag

**The learning rates were not matched.** k=11 trained at the configured 2e-4. k=1 ran at
1e-4: its run resumed from an earlier checkpoint and Lightning restored that optimiser
state, which was not caught until after the run. Sample count (12,348,000 each), batch size
(64), seed (42), data, and schedule are identical; the learning rate is not. Some unknown
share of the gap is the extra learning rate rather than the extra taps. The clean version of
this experiment is a k=1 re-run at 2e-4 -- about three days on one A100.

Neither model converged: there is no LR scheduler in either run, and the final 10% of
training was still returning roughly 3% loss reduction.

## Contents

    hdae/c3di_k1_final-step217936.ckpt     k=1,  50 epochs, 2.15 GB, single 512-d tap at `mid`
    hdae/c3di_k11_final-step192936.ckpt    k=11, 50 epochs, 2.09 GB, 11 taps, 6x47 + 5x46 = 512
    configs/c3di_hier_k1_final.yaml
    configs/c3di_hier_k11_final.yaml
    configs/c3di_hier_k5_final.yaml        the middle rung, training as of this tag
    results/sweep_k1_n256.json             every cell of the strength grid, raw values
    results/sweep_k11_n256.json
    results/persample_k1_n256.npz          PER-SAMPLE raw errors -- any aggregation is
    results/persample_k11_n256.npz         post-hoc arithmetic on these, no re-rendering
    results/best_g.json                    optima + paired-bootstrap tied sets
    results/aggregate_metrics.json         CC / FC / CF1 with raw MAE and accuracy retained

Predictors, SCM, and the ablation checkpoints are unchanged from `c3di-v1-20260825` and are
not duplicated here.

## Loading a checkpoint

```python
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule

cfg = load_hdae_config("experiments/hdae/configs/c3di_hier_k11_final.yaml", require_data=False)
m = HDAELitModule.load_from_checkpoint("c3di_k11_final-step192936.ckpt",
                                       conf=cfg.train_conf, map_location="cpu").eval()
net = m.ema_model          # EMA is what every result here was measured with
```

The attribute predictors are saved from a wrapper, so their keys carry a `net.` prefix that
must be stripped before loading into a bare `convnext_tiny` -- see `load_predictor` in
`experiments/hdae/scripts/cfg_sweep_c3di.py`.

## Evaluation settings that matter

T=50, not T=100. Measured equivalent on the same cohort (k=11, g=3): class 100.00% vs
100.00%, pos_spl 0.0276 vs 0.0275, pos_obj 0.0326 vs 0.0321, rot_obj 0.0752 vs 0.0790
(inside that cell's bootstrap CI), at exactly half the cost. T=25 was tested and REJECTED:
pos_obj degrades 49%, pos_spl 41%. `--fp16` gives 0% (the loop is memory-bound, not
matmul-bound) and `--compile` produces NaNs and is slower. Do not re-try either.

---

## Correction, 2026-08-27: k=11 `do(class)` is not 100%

The headline table above reports k=11 reaching 100.00% on `do(class)`. That figure came from a
256-image cohort, and it is a saturation artifact. Re-measured on 1024 images:

| g | n=256, T=100 | n=1024, T=50 | failures at n=1024 |
|--:|-------------:|-------------:|-------------------:|
| 3 |     100.00%  |    **99.32%**|            7 / 1024 |
| 5 |     100.00%  |    **99.51%**|            5 / 1024 |
| 8 |     100.00%  |    **99.80%**|            2 / 1024 |

256 consecutive successes produce a bootstrap CI of exactly [1.0000, 1.0000] -- an interval with
no width, which is not evidence of certainty but evidence the cohort was too small to contain a
failure. The larger draw contains them.

Two consequences:

* k=11's `do(class)` does NOT saturate at g=3. It keeps improving to g=8 (96.97% at g=1 ->
  99.80% at g=8), so the "tied from g=1.5 through 12" result in best_g.json is an artifact of the
  zero-width interval and will change when recomputed.
* The k=11 advantage over k=1 on `do(class)` is +11.0 points, not +11.7.

This is a correction to the measurement, not to the ordering. Everything else held: across 19
matched cells, 15 fell inside the interval the small cohort predicted, and the mean change per
attribute was -0.2% (class), -3.9% (pos_spl), +7.0% (pos_obj) -- opposite signs, so the residual
is cohort sampling, not the T=100 -> T=50 change. T was isolated separately on an identical
cohort and contributes at most ~1.5% on pos_obj and nothing elsewhere.

## Convergence status at this tag: NOT converged

Neither model has a learning-rate scheduler, and k=1's training loss is still falling at the last
step -- mean loss over the final six deciles of training: 0.001585, 0.001543, 0.001503, 0.001476,
0.001453, 0.001412. The last decile is 2.8% below the one before it. Whether that translates into
better counterfactuals is a separate question, and it is answerable from the intermediate
checkpoints already on disk:

    k=1   epochs  8, 22, 36, 50   (steps 54484, 108968, 163452, 217936)
    k=11  epochs 12, 25, 37, 50   (steps 48234, 96468, 144702, 192936)

Evaluating those eight gives the metric-versus-epoch curve to 50 epochs with no further training.

---

## Finding, 2026-08-28: k=1 is OVER-trained at 50 epochs, not under-trained

The convergence question was posed as "how many more epochs do these need". For k=1 the answer
is negative: it needed fewer. Evaluating the four k=1 checkpoints on a fixed 256-image cohort at
T=50, each intervention at k=1's own best guidance strength:

| epoch | class  | pos_spl | pos_obj | rot_obj | recon (pos_obj) |
|------:|-------:|--------:|--------:|--------:|----------------:|
|     8 | 97.66% |  0.0291 |  0.1422 |  0.1505 |          0.0128 |
|    22 | 96.09% |  0.0345 |  0.1662 |  0.1485 |          0.0126 |
|    36 | 92.19% |  0.0364 |  0.1828 |  0.1611 |          0.0127 |
|    50 | 87.11% |  0.0390 |  0.1976 |  0.1767 |          0.0125 |

All four metrics degrade monotonically. `class` loses 10.5 points, `pos_obj` 39%. Sixteen of
sixteen readings move the wrong way, which is far outside n=256 sampling noise.

RECONSTRUCTION IS FLAT over the same span (0.0128 -> 0.0125). Nothing is broken in the
autoencoder; what decays is specifically the response to an intervention. And the TRAINING LOSS
FALLS THROUGHOUT. Loss and counterfactual quality are anti-correlated for this model, so a
loss-plateau stopping criterion cannot see this failure at all.

k=11 over the same span does the opposite -- improves, then plateaus by epoch 25-37:

| epoch  | class   | pos_spl | pos_obj | rot_obj |
|-------:|--------:|--------:|--------:|--------:|
|     12 | 100.00% |  0.0211 |  0.0389 |  0.1098 |
|     25 | 100.00% |  0.0202 |  0.0310 |  0.0841 |
|     37 |  99.61% |  0.0202 |  0.0303 |  0.0799 |
|     50 | 100.00% |  0.0214 |  0.0306 |  0.0752 |

WORKING HYPOTHESIS: conditioning collapse. With a single 512-d tap at `mid`, the semantic latent
absorbs progressively more of the image until the attribute vector is redundant and the
classifier-free-guidance delta shrinks toward nothing. k=11's eleven taps spread that capacity, so
it does not collapse. This is directly testable with the `dL_all` diagnostic used in the original
conditioning ablation -- loss with attributes nulled minus loss with true attributes -- computed
across the four k=1 checkpoints. Not yet run.

CONSEQUENCES FOR EVERYTHING ALREADY PUBLISHED:

* Every k=1 number in this tag comes from k=1's WORST checkpoint. At each model's best epoch the
  k=11 advantage on pos_obj is 0.0306 vs 0.1422 = 4.6x, not the 6.4x reported from epoch 50.
* The matched-epoch protocol was still the fair comparison to run; it just happens to flatter k=11.
* The k=1 extension to 75 epochs was started on the projection above and STOPPED at epoch 51 once
  these curves came in. No k=1 extension checkpoint was ever written; its output directory holds
  only the seeded copy of epoch 50.

k=11's extension to 75 epochs continues, since its curve is flat-to-improving rather than
degrading and the question "does it keep improving past 50" is still open for it.
