# HDAE YAML schema

| Field | Meaning |
|---|---|
| `base_template` | Callable upstream template in `templates.py`; all unspecified diffusion/network defaults are inherited. |
| `data.image_dir` | Raw CelebA-HQ image directory. |
| `data.attr_path`, `partition_path` | CelebA attributes and official split sources. |
| `data.image_size` | One-time packed resolution and model resolution. |
| `data.lmdb_path`, `attr_npz` | Raw-RGB LMDB directory and aligned array output. |
| `data.flip_aug` | Cheap train-only random horizontal flip. |
| `data.resize_filter` | `bicubic` or `lanczos`; used only while packing. |
| `encoder.type` | `flat` (K=1) or `hierarchical`. |
| `encoder.tap_resolutions` | Real encoder stages, ordered coarse-to-fine (ascending). |
| `encoder.level_dims` | Latent widths corresponding one-to-one with taps. |
| `encoder.pool` | Phase-1 value: `adaptive_avg`. |
| `encoder.proj` | Head type: `linear` or two-layer `mlp`. |
| `conditioning.strategy` | `concat_proj`; `per_resolution` is reserved/experimental. |
| `conditioning.style_ch` | Decoder semantic conditioning width; must equal `sum(level_dims)`. |
| `conditioning.latent_drop_prob` | Per-sample, per-level probability of replacing that latent with its learned null token during training; default `0.12`. |
| `conditioning.cfg_drop_prob` | Per-sample probability of replacing all modeled attributes with the learned null attribute token during training for classifier-free guidance; recommended default `0.1`. |
| `conditioning.cfg_guidance_scale` | Default attribute-CFG inference scale used by PCF when no CLI override is passed; recommended default `2.0` for a moderate edit-strength/preservation tradeoff. |
| `train.batch_size_per_gpu`, `total_batch_size` | Local/global batch; global must equal local times devices. |
| `train.lr`, `ema_decay`, `T`, `T_eval` | Upstream learning rate, EMA, train diffusion steps, and DDIM evaluation steps. |
| `train.max_steps`, `precision`, `grad_clip`, `num_workers` | Trainer/runtime settings. |
| `train.compile` | Reserved optional compile flag; false by default. |
| `lightning.*` | Passed to Lightning Trainer; production configs use two-GPU DDP. |
| `seed`, `output_dir` | Reproducibility seed and artifact root. |
