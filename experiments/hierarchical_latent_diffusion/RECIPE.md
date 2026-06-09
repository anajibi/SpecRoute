# Hierarchical Latent Diffusion Design

Images are encoded by a hard-frozen DINOv2 backbone. A trainable chain encoder creates strictly decreasing latent levels; only the first encoder sees DINO features, while every later encoder sees the concatenation of earlier latents. A decoder consumes every level and predicts the latent of a hard-frozen Stable Diffusion VAE. Independent vector diffusion priors model each conditional distribution `p(Z_l | Z_<l)`.

Training is split into stage 1 (chain encoder and decoder), latent extraction, and stage 2 (priors). Preservation and single-level counterfactual probes resample levels and report DINO similarity and image MSE. Optional external attribute, LPIPS, or identity metrics can be computed from the emitted images/checkpoints without modifying existing experiments.
