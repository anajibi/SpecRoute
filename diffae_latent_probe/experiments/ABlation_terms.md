# Component Ablation Terms

This document describes the component-ablation renderings used in the CelebA pseudo-counterfactual experiments.

## Definitions

- **full**: Reconstruction using both the semantic latent `z` and the stochastic latent `x_T` from the same image.
- **z-only**: Reconstruction using the semantic latent `z` from the image and a freshly sampled `x_T` (Gaussian), holding `z` fixed.
- **z-only avg**: Average reconstruction from multiple `x_T` samples while holding `z` fixed (Monte Carlo average).
- **xT-mean**: Reconstruction using the batch mean of `z` and the original `x_T` (tests how much detail is carried by `x_T`).
- **xT-zero**: Reconstruction using a zero vector for `z` and the original `x_T`.
- **xT-mismatch**: Reconstruction using a *different image’s* `z` with the original `x_T`. This is effectively a **z-swap** (swap `z`, keep `x_T`).
- **xT-swap**: Reconstruction using the original `z` with a *different image’s* `x_T` (swap `x_T`, keep `z`).

## Notes

- `xT-mismatch` and `xT-swap` are the two swap conditions: swapping `z` and swapping `x_T` respectively.
- All saved grids and images are in `[0,1]` RGB space after denormalization to avoid washed-out outputs.

