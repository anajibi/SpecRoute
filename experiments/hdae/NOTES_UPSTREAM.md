# Confirmed upstream integration points

- Autoencoder model/config: `BeatGANsAutoencModel` and `BeatGANsAutoencConfig` in `model/unet_autoenc.py`.
- Semantic encoder model/config: `BeatGANsEncoderModel` and `BeatGANsEncoderConfig` in `model/unet.py`. It builds `input_blocks`, `middle_block`, then the `adaptivenonzero` pool/projection `out`.
- Semantic dimension: `BeatGANsAutoencConfig.enc_out_channels`, populated from `TrainConfig.style_ch` by `TrainConfig.make_model_conf()`.
- Decoder conditioning: `BeatGANsAutoencModel.forward()` obtains `cond`, passes it through `TimeStyleSeperateEmbed`, then threads `cond_emb` through the U-Net blocks. Each conditioned `ResBlock` applies `cond_emb_layers`; `apply_conditions()` combines its scale/shift with the timestep scale/shift.
- Training: upstream already uses Lightning (`experiment.LitModel`). `training_step()` samples timesteps with `T_sampler.sample`, calls `sampler.training_losses(model, x_start, t)`, and averages `loss`. `on_train_batch_end()` performs EMA; `configure_optimizers()` preserves the configured Adam/AdamW and optional warmup schedule.
