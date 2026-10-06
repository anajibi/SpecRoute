# CelebA-HQ 256 ladder — training protocol

Fixes every hyperparameter so k=1, 3, 5, 7, 11 differ **only** in the tap ladder. Anything that
varies besides k makes the depth comparison uninterpretable, which is the whole point of the ladder.

k=11 is already trained (`celebahq256_k11_cd015r`, step 46,875). This protocol reproduces its
conditions exactly for the other four.

---

## 1. The ladder

18 decoder blocks, 512 latent dims, split as evenly as the integers allow. Taps are a **nested
subset** of k=11's, so each rung is a coarsening of the one below it rather than a different design.

| k | `hier_tap_block_ids` | `hier_level_dims` (sum 512) | blocks per level |
|---|---|---|---|
| 1 | `[mid]` | `[512]` | 18 |
| 3 | `[0, 5, mid]` | `[171, 171, 170]` | 6, 6, 6 |
| 5 | `[0, 3, 6, 9, mid]` | `[103, 103, 102, 102, 102]` | 4, 4, 4, 3, 3 |
| 7 | `[0, 2, 4, 6, 8, 9, mid]` | `[74, 73, 73, 73, 73, 73, 73]` | 3, 3, 3, 3, 2, 2, 2 |
| 11 | `[0..9, mid]` | `[47×6, 46×5]` | per k=11 config |

`hier_block_to_level`, 18 entries, monotone non-decreasing:

    k=1   [0]*18
    k=3   [0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2]
    k=5   [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3, 4,4,4]
    k=7   [0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4, 5,5, 6,6]
    k=11  [0,0,1,1,2,3,3,4,4,5,6,6,7,7,8,9,9,10]

`n_decoder_output_blocks: 18` for all.

---

## 2. Fixed hyperparameters — do not vary these

### Conditioning
    attr_embed_dim      384          # 96 per attribute under concat_film
    attr_fusion         concat_film  # `sum` was the worst of thirteen configs tested
    style_ch            512
    latent_drop_prob    0.0
    cfg_drop_prob / attr_dropout_prob  -- phase-dependent, see §3a
    attributes          Male, Young, Beard, Bald  (binary, range [-1, 1])

### Optimisation
    batch_size_per_gpu  8
    accum_batches       8            # effective batch 64
    lr                  1.0e-4       # DiffAE's own lr for this template
    ema_decay           0.9999
    precision           16-mixed
    grad_clip           1.0
    T                   1000
    T_eval              100
    use_checkpoint      true

`accum_batches` must be set in the config, not only on the Trainer: `conf.accum_batches` gates the
EMA update, and setting one side alone advances the EMA 8× too fast and silently corrupts every
evaluation.

### Schedule
    checkpoint_every_n_steps 3125
    save_top_k               -1

## 3a. Two phases, because k=11 was trained that way

k=11 was not trained at one dropout setting. It ran 31,250 steps at the lower setting, then was
resumed for 15,625 more at the raised one. Every other rung must replicate that, or the depth
comparison is confounded with the dropout schedule.

| phase | steps (absolute) | cfg_drop_prob | attr_dropout_prob | init |
|---|---|---|---|---|
| p1 | 31,250 | 0.10 | 0.08 | `init_from` ffhq256_autoenc |
| p2 | 46,875 | 0.15 | 0.12 | `resume_from` the p1 checkpoint |

Configs: `celebahq256_k{3,5,7}_p1.yaml` then `..._p2.yaml`.
**k=1 already has p1** (trained to 31,250 at 0.10/0.08), so it needs only `celebahq256_k1_p2.yaml`.
k=11 is complete as `celebahq256_k11_cd015r`.

---

## 3. Initialisation — the part that breaks if you get it wrong

**Phase 1 (k=3, 5, 7):**

    init_from: "experiments/hdae/pretrained/ffhq256_autoenc_model.pt"

95.8% of the model transfers; the tap heads, per-block FiLM and attribute embedding are new. From
scratch without this is DiffAE's own 200M-sample budget, ~96 days on one A100.

**Phase 2 (every rung, and k=1's only phase):**

    resume_from: "experiments/hdae/outputs/<run>_p1/checkpoints/last.ckpt"

**Never use `init_from` to continue converged weights.** It loads weights only, so Adam restarts
with zero second moments and takes sign-like steps of ~±lr regardless of gradient magnitude. On a
converged model that walks it out of its basin: grad norms crept 0.006 → 0.042 over ~400 steps with
the loss flat at 0.012, then went non-finite at optimiser step **423**. Reproduced exactly, twice.
`resume_from` restores weights, Adam moments, the GradScaler and `global_step`.

`max_steps` is **absolute** on a resume. k=1 resuming at step 31,250 and wanting 15,625 more sets
`max_steps: 46875`. Setting 15,625 exits immediately, already past the limit.

---

## 4. Safety rails

    HDAE_NAN_TRACE=<path> HDAE_NAN_TRACE_ABORT=1 python experiments/hdae/scripts/train.py --config ...

Logs pre-clip grad norms every 10 steps and aborts the moment a **parameter** goes non-finite. It
deliberately does not abort on a non-finite *gradient*: the AMP scaler legitimately skips those
(~5 per 15,000 steps at the inherited 2²⁴ scale) and aborting there would kill healthy runs.

Checkpoints are 2.8 GB and the disk holds ~4 GB free. Run `ckpt_janitor.sh <ckptdir> <s3prefix> 0`
alongside: it ships each checkpoint to S3, verifies the multipart ETag, then deletes the local copy.
Size equality is not proof — a truncated file uploads and size-matches perfectly.

Run one training job at a time. Measured: 5 concurrent jobs on this A100 give **0.88×** the
throughput of running them sequentially.

---

## 5. Evaluation — identical for every rung

Use the frozen kit in `experiments/hdae/baselines/`:

- cohort: `cohort_2048.json`, 2,048 held-out males, Bald flipped **both** directions
- readers: the 40 per-attribute ConvNeXt-Small models; FC_unobs over the 28 in `reader_keep.json`
- thresholds: untuned (`logit > 0`); the intervened attribute uses its F1-optimal threshold
- null set: **target-only**. It beat all-four at every attribute (ΔCF1 −0.008 to −0.116)
- guidance: sweep per attribute. The optimum varies 3× across attributes (Beard 2, Bald 6), so a
  single fixed g handicaps some badly
- FC is measured against each model's **own reconstruction**, not the source photograph
- run the null-intervention control (`do(A = its existing value)`) per model and report `FC / floor`

Scripts: `gsweep_attrs_celebahq.py`, `nullctrl_celebahq.py`, `ablation_nullset_celebahq.py`.

---

## 6. Budget

At ~1,360 optimiser-steps/hour measured on one A100:

| rung | phases needed | train h | eval h | total |
|---|---|---|---|---|
| k=1 | p2 only (15,625 steps) | 11.5 | 19 | **30 h** |
| k=3 | p1 + p2 (46,875) | 34.5 | 19 | **54 h** |
| k=5 | p1 + p2 | 34.5 | 19 | **54 h** |
| k=7 | p1 + p2 | 34.5 | 19 | **54 h** |
| | | | | **~192 h = 8 days** |

Evaluation is ~13 h for a 5-point g-sweep over 4 attributes plus ~6 h for the null-intervention
control. Run strictly sequentially: 5 concurrent jobs measured at 0.88x the throughput of one.

If 8 days is too much, drop k=3 or k=7 rather than shortening any run — a ladder of 1/5/11 with
matched conditions is worth more than 1/3/5/7/11 with mismatched ones.

---

## 7. One deliberate exclusion

The strongest single lever identified is **oversampling rare-attribute positives**. Bald is 2.4% of
the training set (582 images), needs g=6, and loses 23.2% of achievable preservation; Beard is 18.9%,
needs g=2, and loses 11.8%. The model never learned Bald well enough to express it without being
pushed hard, and the push is what causes the collateral damage.

It is **not** in this protocol because changing the sampler would make every rung non-comparable with
the published k=11 results. Run it as a separate controlled experiment — k=11 with and without
rebalancing — not folded into the ladder.
