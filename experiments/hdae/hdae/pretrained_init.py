"""Initialise a hierarchical HDAE from a vanilla DiffAE checkpoint.

95.8% of the 256px hierarchical model is already trained in `ffhq256_autoenc`: the whole UNet
decoder, the middle block, the time embedding, and the encoder backbone. Only the tap projection
heads, the per-block FiLM and the attribute embedding are new. Training those from a cold model
would cost 200M samples (~96 days on one A100); starting here costs a day.

Two key remaps:
  encoder.*  ->  encoder.backbone.*   HierarchicalSemanticEncoder wraps the same BeatGANs encoder
                                      under `.backbone`. All 116 tensors match by shape.
  model. / ema_model. prefixes        the published file is a full Lightning state_dict holding
                                      both copies plus an `x_T` buffer.

This asserts what landed rather than trusting strict=False to be sensible: a silent partial load
is the failure mode that makes a fine-tune look like a slow from-scratch run.
"""
import torch

_EXPECT_MIN_FRAC = 0.90


def load_pretrained_into(model, ckpt_path, prefix="ema_model.", verbose=True):
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    src = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
    if not src:
        avail = sorted({k.split(".")[0] for k in raw})[:8]
        raise ValueError(f"no tensors under prefix {prefix!r}; top-level keys look like {avail}")

    tgt = model.state_dict()
    remapped, skipped = {}, []
    for k, v in src.items():
        cands = [k, f"encoder.backbone.{k[len('encoder.'):]}" if k.startswith("encoder.") else None]
        for c in cands:
            if c and c in tgt and tgt[c].shape == v.shape:
                remapped[c] = v
                break
        else:
            skipped.append(k)

    missing = [k for k in tgt if k not in remapped]
    n_load = sum(remapped[k].numel() for k in remapped)
    n_tot = sum(v.numel() for v in tgt.values())
    frac = n_load / n_tot

    model.load_state_dict({**tgt, **remapped}, strict=True)

    if verbose:
        import collections
        grp = collections.Counter(k.split(".")[0] for k in missing)
        print(f"[pretrained_init] {ckpt_path}")
        print(f"[pretrained_init] loaded {len(remapped)}/{len(tgt)} tensors, "
              f"{n_load/1e6:.1f}M/{n_tot/1e6:.1f}M params ({100*frac:.1f}%)")
        print(f"[pretrained_init] randomly initialised ({(n_tot-n_load)/1e6:.2f}M params): "
              + ", ".join(f"{g}:{n}" for g, n in grp.most_common()))
        if skipped:
            print(f"[pretrained_init] {len(skipped)} source tensors unused (e.g. {skipped[:3]})")
    if frac < _EXPECT_MIN_FRAC:
        raise RuntimeError(
            f"only {100*frac:.1f}% of parameters initialised from {ckpt_path} "
            f"(expected >={100*_EXPECT_MIN_FRAC:.0f}%). A partial load here would look like a "
            f"slow from-scratch run rather than an error -- refusing to start.")
    return dict(loaded_tensors=len(remapped), loaded_params=n_load,
                total_params=n_tot, fraction=frac, new_groups=dict(
                    __import__("collections").Counter(k.split(".")[0] for k in missing)))
