"""Two independent fixes for the high-g noise, and the measurements that motivated each.

Diagnosis (cfg_diag / cfg_diag2 on the CelebA-HQ k=11 model, do(Bald)):

  * NOT variance blow-up. std(eps_hat)/std(eps_cond) stays within 5% of 1.0 at every g and every
    timestep, so the usual CFG rescaling fix has nothing to correct here.
  * NOT a noisy guidance direction. delta = eps_cond - eps_null is SMOOTHER than the signal it is
    added to at every t (HF-energy ratio 0.27-0.65), which is why the edit lands cleanly.
  * The problem is WHERE it is applied. delta's relative magnitude peaks at t~200 (8.2% of signal)
    and its high-frequency content grows 29x from t=900 (0.013) to t=20 (0.373). Multiplying that
    by g and writing it into an almost-finished image, at every remaining step, compounds into
    visible speckle.

IntervalCFG targets that directly: guide where delta is smooth and structural, stop once it is
fine-grained. dynamic_threshold is the orthogonal safety net -- it bounds the x0 estimate instead
of the noise prediction, so it catches overshoot the variance test would not reveal.
"""
import torch


class IntervalCFG(torch.nn.Module):
    """Classifier-free guidance applied only for t in [t_lo, t_hi]; g=1 elsewhere.

    The wrapper sees ORIGINAL-scale timesteps (0-999): SpacedDiffusion maps the spaced index back
    through timestep_map before the model is called, and beatgans_rescale_timesteps is False.
    """

    def __init__(self, base_model, guidance_scale, t_lo=200, t_hi=800):
        super().__init__()
        self.base_model = base_model
        self.guidance_scale = float(guidance_scale)
        self.t_lo, self.t_hi = int(t_lo), int(t_hi)
        self._seen = set()

    def forward(self, x, t, cond, **kw):
        cond_out = self.base_model.forward(x=x, t=t, cond=cond, **kw)
        if self.guidance_scale == 1.0:
            return cond_out
        self._seen.add(int(t.view(-1)[0]))
        # per-sample gate; t is a batch vector, so a scalar test would be wrong at batch edges
        on = ((t >= self.t_lo) & (t <= self.t_hi)).float().view(-1, *([1] * (x.dim() - 1)))
        if float(on.sum()) == 0:
            return cond_out
        null_mask = torch.ones_like(cond["y_idx"], dtype=torch.bool)
        null_cond = {"zs": cond["zs"], "y_idx": cond["y_idx"], "null_mask": null_mask}
        uncond_out = self.base_model.forward(x=x, t=t, cond=null_cond, **kw)
        g_eff = 1.0 + (self.guidance_scale - 1.0) * on
        guided = uncond_out.pred + g_eff * (cond_out.pred - uncond_out.pred)
        return cond_out.__class__(pred=guided, cond=cond)


def dynamic_threshold(percentile=0.995):
    """Imagen-style dynamic thresholding, as a `denoised_fn` for ddim_sample_loop.

    Acts on the x0 ESTIMATE, not the noise prediction. Each image's |x0| percentile s is found;
    if s > 1 the image is clipped to [-s, s] and divided by s, pulling outliers back inside the
    valid range while preserving relative contrast. s <= 1 leaves the image untouched, so this is
    a no-op whenever nothing is overshooting.
    """
    def fn(x0):
        flat = x0.reshape(x0.shape[0], -1).abs()
        s = torch.quantile(flat.float(), percentile, dim=1).clamp(min=1.0)
        s = s.view(-1, *([1] * (x0.dim() - 1)))
        return x0.clamp(-s, s) / s
    return fn


class TargetNullCFG(torch.nn.Module):
    """CFG whose unconditional branch nulls ONLY the attribute being edited.

    The default wrapper nulls all four attributes, so the guidance direction is
    eps(all known) - eps(nothing known): a joint direction whose cosine similarity with a
    single-attribute direction was measured at just 0.59. Roughly 40% of what gets amplified is
    the OTHER attributes' information, which is why raising g drags in unrequested facial hair
    and flattens backgrounds.

    Nulling only the target isolates it. This is in-distribution rather than a hack: training
    already drops attributes independently via attr_dropout_prob=0.08, so "three known, one
    missing" is a state the model has seen many times.

    Cost: the direction is about half the magnitude (3.56% of signal vs 8.24% at t=200), so a
    given edit needs roughly twice the g.
    """

    def __init__(self, base_model, guidance_scale, attr_index, t_lo=None, t_hi=None):
        super().__init__()
        self.base_model = base_model
        self.guidance_scale = float(guidance_scale)
        self.attr_index = int(attr_index)
        self.t_lo, self.t_hi = t_lo, t_hi

    def forward(self, x, t, cond, **kw):
        cond_out = self.base_model.forward(x=x, t=t, cond=cond, **kw)
        if self.guidance_scale == 1.0:
            return cond_out
        null_mask = torch.zeros_like(cond["y_idx"], dtype=torch.bool)
        null_mask[:, self.attr_index] = True
        null_cond = {"zs": cond["zs"], "y_idx": cond["y_idx"], "null_mask": null_mask}
        uncond_out = self.base_model.forward(x=x, t=t, cond=null_cond, **kw)
        g = self.guidance_scale
        if self.t_lo is not None:
            on = ((t >= self.t_lo) & (t <= self.t_hi)).float().view(-1, *([1] * (x.dim() - 1)))
            g = 1.0 + (self.guidance_scale - 1.0) * on
        return cond_out.__class__(pred=uncond_out.pred + g * (cond_out.pred - uncond_out.pred),
                                  cond=cond)
