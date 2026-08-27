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
