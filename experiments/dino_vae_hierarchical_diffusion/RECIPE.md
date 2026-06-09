# Experiment recipe

1. Run `run_reconstruction_eval.py --vae_ceiling` before training to establish the frozen-VAE ceiling.
2. Train the evidence pyramid, stochastic hierarchy, deterministic stabilizer, and S0 decoder with `train_stage1_autoencoder.py`.
3. Cache posterior samples/statistics with `extract_latents.py`, then train hierarchy priors with `train_stage2_priors.py`.
4. Evaluate reconstruction and K=3 level resampling. Inspect swaps before treating level meanings as established.
5. Fit external linear probes and pass normalized probe weights to `run_latent_direction_editing.py` for pseudo-counterfactual edits.

The trainable encoder receives only VAE latents and frozen DINO features. No attributes, LIC objective, adversarial objective, or raw RGB conditioning are used. Level dropout replaces omitted levels with zeros; increase KL pressure if the fine level dominates.
