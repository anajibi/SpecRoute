# Archived

This is the predecessor to `experiments/hdae`. It probes a **frozen,
pretrained, non-hierarchical** DiffAE checkpoint (single `z_sem` + `x_T`,
linear-probe attribute directions in `z_sem`, `z_edit = z_sem + alpha * w`).
`experiments/hdae` supersedes it: same probing/pseudo-counterfactual
methodology, but applied to a **trained-from-scratch hierarchical** semantic
latent (`z_0..z_{K-1}`) with attribute-CFG conditioning.

Moved here (out of `diffae_latent_probe/` at the repo root) because nothing
under `experiments/hdae/` imports this code. `PRESENTATION_CONTEXT.md` in
this directory has a full writeup of the methodology if you need it for
comparison.

**Not moved:** `diffae_latent_probe/data/` (raw CelebA-HQ images/labels,
~19 GB, untracked) is still the live raw-data source referenced by every
`experiments/hdae/configs/*.yaml` — see `experiments/hdae/AGENDA.md` §5.
`diffae_latent_probe/outputs/` and `diffae_latent_probe/configs/outputs/`
(generated run artifacts, untracked) were also left in place, unreferenced
by anything active.

Start at `experiments/hdae/AGENDA.md` for the current, active experiment.
