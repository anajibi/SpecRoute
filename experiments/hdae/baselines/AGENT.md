# HDAE — state of the project and where everything lives

Last updated 4 November 2026. Supersedes the 6 October version, which predates the second causal
graph, the training protocol and most of the models.

Read this first, then `MANIFEST.md` at the S3 prefix root for the file-by-file inventory.

---

## 1. What the project is

A hierarchical diffusion autoencoder for counterfactual image generation. A diffusion autoencoder
conditioned on a structural causal model over named attributes, where the semantic latent is
partitioned across `K` taps entering the decoder at different depths instead of one global code
injected everywhere. `K` is the experimental variable; everything else is held fixed.

Code: `git@github.com:anajibi/SpecRoute.git`, branch **`celebahq-ladder`**.
Artifacts: `s3://najibi-research-7f2a/hdae-handoff/`.
Protocol: `experiments/hdae/configs/LADDER_PROTOCOL.md` — read before training anything.

---

## 2. Models — all trained, all archived

Eleven models. Every CelebA-HQ rung ran the same two phases: 31,250 steps at
`cfg_drop 0.10 / attr_drop 0.08` from the ffhq256 transfer, then resumed to 46,875 at `0.15 / 0.12`.
A rung trained straight through at the final dropout is **not** comparable and must not be mixed in.

### CelebA-HQ 256 — graph A · {Male, Young, Beard, Bald}

| K | final checkpoint | evaluated |
|---|---|---|
| 1 | `celebahq256/k1_p2/epoch=124-step=46875.ckpt` | no |
| 5 | `celebahq256/k5_p2/epoch=124-step=46875.ckpt` | no |
| 11 | `celebahq256/k11_cd015r/epoch=124-step=46875.ckpt` | **yes** |

Phase-1 states are in `k1/`, `k5_p1/`, `k11/` at `step=31250` — intermediate, not for evaluation.

### CelebA-HQ 256 — graph B · {Eyeglasses, Smiling}, independent

| K | final checkpoint | evaluated |
|---|---|---|
| 1 | `celebahq256/b_k1_p2/epoch=124-step=46875.ckpt` | no |
| 11 | `celebahq256/b_k11_p2/epoch=124-step=46875.ckpt` | no |

Phase-1 states in `b_k1_p1/`, `b_k11_p1/`. Only K=1 and K=11 were trained for this graph.

### MorphoMNIST++ · {digit, thickness, intensity, hue}

`morpho_ladder/{k1,k5,k11}_s42/` and `_s43/` — baseline arm, two seeds.
`morpho_adv/{k1,k5,k11}_s42/` — adversarial arm (per-tap attribute heads with a confusion loss).
Trained and evaluated. SCM at `morpho_scm.pt`, data at `data_packed/morphomnist_70k.h5`.

### Causal3DIdent · {class, spotlight position, object position, object rotation}

Trained and evaluated. `c3di/c3di_results_and_analysis.tar.gz` holds the ladder metrics, per-factor
breakdowns, cohorts and logs; `c3di/attr_predictors_c3di.tar.gz` holds the seven readers.

> **The C3DI model weights no longer exist.** Confirmed by a full-filesystem search: the output
> directories retain only tensorboard events and `hparams.yaml`, nothing was ever uploaded, and no
> `.ckpt` survives locally. The metrics are safe but those models cannot be re-sampled or re-scored
> without retraining from `configs/c3di_hier_k{1,5,11}_final.yaml`. This blocks adding any new
> metric to C3DI.

---

## 3. Where things stand

Trained but **not evaluated**: graph A K=1 and K=5, graph B K=1 and K=11.

Graph B cannot be evaluated yet. Two items are outstanding:

1. **No fitted SCM.** `configs/causal_graph_celebahq_b.yaml` declares
   `scm_checkpoint: experiments/hdae/outputs/scm/celebahq_b_scm.pt`, which does not exist. It is two
   independent binary marginals, so it is minutes of CPU and no GPU. Training did not need it
   (image logging is off); evaluation does.
2. **No cohort.** `cohort_2048.json` in `baseline_kit.tar.gz` is all-male with Bald flipped — built
   for graph A and wrong for an Eyeglasses/Smiling study. Graph B needs its own frozen cohort and
   the selection rule is a research decision, not a default.

The reader basis **is** handled — see §4.

---

## 4. Evaluation assets, and the one thing that inverts between graphs

| asset | key |
|---|---|
| 40 per-attribute readers | `celebahq256/readers40_per_attribute.tar.gz` |
| graph A reader basis | `celebahq256/reader_keep.json` — 28 FC_unobs |
| graph B reader basis | `celebahq256/reader_keep_graphB.json` — 30 FC_unobs |
| cohort, scorer, protocol, reference numbers | `celebahq256/baseline_kit.tar.gz` |
| pretrained init | `celebahq256/ffhq256_autoenc_model.pt` |
| packed 256px data | `data_packed/celebahq_256_pack.tar.gz` |

**The observed/unobserved split inverts with the graph, and nothing does this automatically.**
Eyeglasses and Smiling sit *inside* graph A's 28-attribute FC_unobs set. Scoring graph B against
`reader_keep.json` would count the intervened attributes as collateral damage, so a successful edit
would be penalised as unobserved drift — a bad score precisely when the model works. Use
`reader_keep_graphB.json` for graph B: Eyeglasses and Smiling move out, Male/Young/Beard/Bald move
in (all four clear the F1 ≥ 0.55 screen).

Attribute **selection** was made on the partition-0 slice, never on the evaluation cohort.
Threshold fitting and tuning on the cohort is fine; selection on it is not, or the surviving readers
are the ones that happen to look good on the evaluation set.

---

## 5. Metric definitions, and a caveat

- **CC** — the intervened attribute now reads as its target value.
- **FC_obs** — among conditioned attributes, non-descendants of the target are unchanged.
  Descendants are expected to move and are excluded, not penalised.
- **FC_unobs** — the same test over attributes outside the graph, where any movement is pure
  collateral damage. The stricter and more informative of the two.
- **CF1** — computed **per intervention and then averaged**, not on pooled rates:
  `CF1_a = 2·CC_a·FC_a/(CC_a+FC_a)`, then `CF1 = mean_a CF1_a`. The two differ, and the pooled form
  lets a frequent intervention dominate a rare one.
- **Composition / FC floor** — intervene on an attribute to the value it already has. Preservation
  under that null intervention is **not** 1.0, so preservation is reported as a ratio to that
  measured ceiling, never as a difference from one.
- Guidance is swept **per attribute**; the optimum is not shared across attributes.
- Null set is **target-only** — only the intervened attribute is masked in the unconditional branch.
  It beat all-four nulling at every attribute tested.

> **CF1 is not comparable across datasets yet.** The three pipelines combine CC with different
> preservation terms: MorphoMNIST uses all measured attributes, Causal3DIdent reports separate
> conditioned and unconditioned variants, CelebA-HQ uses the unconditioned set. Within a dataset the
> comparison across `K` is sound. Unify the definition before any cross-dataset claim.

---

## 6. How to run training

Scripts in `experiments/hdae/ops/`, documented in `ops/README.md`.

    systemd-run --user --scope --unit=hdae-<run> --slice=hdae.slice \
      bash experiments/hdae/ops/supervise.sh <config> <run-name> <target-step> <s3-prefix>

`loginctl enable-linger` must be on. Run one job at a time: five concurrent jobs measured 0.88× the
throughput of running them sequentially.

### Four rules, each from a failure that cost time

**Never use `init_from` to continue converged weights.** It loads weights only, so Adam restarts
with zero second moments and takes sign-like ±lr steps regardless of gradient magnitude. On a
converged model that goes non-finite at optimiser step **423**, reproduced twice. Use `resume_from`,
which restores weights, Adam moments, the GradScaler and `global_step`. `init_from` is safe only
from the pretrained DiffAE checkpoint.

**`max_steps` is absolute on a resume.** Phase 2 is `46875`, not `15625`; the latter exits
immediately.

**`accum_batches` must be in the config, not only on the Trainer.** `conf.accum_batches` gates the
EMA update, and setting one side alone advances the EMA 8× too fast and silently corrupts every
evaluation.

**A non-finite gradient is normal; a non-finite parameter is not.** The AMP scaler legitimately
skips a handful of steps per run. The NaN tracer (`HDAE_NAN_TRACE`, `HDAE_NAN_TRACE_ABORT=1`) aborts
only on a non-finite **parameter** — aborting on gradients would kill healthy runs.

### Storage

Checkpoints are ~2.8 GB and the disk holds ~10 GB free. Run `ckpt_janitor.sh <dir> <s3prefix> 0`
alongside: it ships each checkpoint, recomputes the multipart ETag locally to verify, then deletes
the local copy. Kill stale janitors when a run ends — one left watching a finished directory with
`KEEP=0` deleted a restored checkpoint twice.

**Size equality is not proof of a complete upload.** A truncated file ETag-verifies perfectly; the
ETag proves the transfer was faithful, not that the file was whole. Use `ops/verify_s3.sh`, which
reads the tail back from S3 and checks for the zip end-of-central-directory record.

When listing S3 to check completeness, **sort by step number, not by name** — `step=37500` sorts
after `step=46875` alphabetically, so a missing final checkpoint can look present.

---

## 7. Baselines

`celebahq256/BASELINE_HANDOFF.md` has the protocol; `baseline_kit.tar.gz` has the frozen cohort,
the model-agnostic scorer and the reference numbers. Baseline runs are on a separate cluster.

**Open question.** If the comparison methods run at lower image resolution than ours, resolution is
confounded with model quality and the numbers are not directly comparable. Either match resolutions
for a common-resolution comparison (which needs readers retrained at that resolution, since many
fine-detail attributes degrade badly) or report each at its native resolution with the limitation
stated. A compute estimate for each option exists.

---

## 8. Current blocker

The box rebooted on 4 November into kernel `7.0.0-30-generic` and the **NVIDIA kernel module did not
load** — `nvidia-smi` cannot reach the driver and `torch.cuda.is_available()` is False. The A100 is
on the PCI bus and DKMS has a module built for this kernel, so it looks like a load rather than a
reinstall: `sudo modprobe nvidia`. Nothing can train or evaluate until that is back.
