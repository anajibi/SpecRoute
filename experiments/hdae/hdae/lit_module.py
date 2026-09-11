"""Lightning module for conditional HDAE training."""
from torch.cuda import amp

from choices import TrainMode
import torch

from experiment import LitModel, ema

from .attr_utils import observed_unique, to_cond_values, to_index_space


class _GradReverse(torch.autograd.Function):
    """Identity forward, negated-and-scaled gradient backward.

    The adversary heads must MINIMISE their prediction error (they are honest probes), while
    the encoder must MAXIMISE it. Reversing the gradient at the boundary gives both from one
    backward pass: the heads see +grad, the encoder sees -lambda*grad.
    """

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def _mlp(din, dout, hidden, layers):
    mods, d = [], din
    for _ in range(max(1, layers) - 1):
        mods += [torch.nn.Linear(d, hidden), torch.nn.GELU()]
        d = hidden
    mods.append(torch.nn.Linear(d, dout))
    return torch.nn.Sequential(*mods)


# MorphoMNIST specifics. digit stores its class index; hue stores its BIN CENTRE (0.05..0.95),
# so it needs the same *10-0.5 conversion the evaluator uses -- getting this wrong is what once
# trained a predictor to always answer class 0 while reporting 100% accuracy.
_CAT_CLASSES = {"digit": 10, "hue": 10}


class HDAELitModule(LitModel):
    def setup(self, stage=None):
        """Build the adversary heads HERE, not on the first training step.

        Lightning calls setup() and then configure_optimizers(), which collects
        self.parameters(). Heads created later -- as they were on the first attempt -- are
        never handed to the optimiser: they stay at random init, emit uniform predictions,
        and the confusion loss sits pinned at its floor contributing exactly zero gradient.
        The whole adversarial term was inert for a full 36,000-step run, and the failure was
        SILENT because an untrained probe and a perfectly-fooled probe are indistinguishable
        from the loss value alone -- adv/cat_acc read 0.10 either way. Tap widths come from
        hier_level_dims (verified equal to the actual z widths), so no forward pass is needed.
        """
        e = self.model.hdae_conf.encoder
        if float(getattr(e, "adv_tap_lambda", 0.0) or 0.0) <= 0.0 or hasattr(self, "_adv_heads"):
            return None
        hid = int(getattr(e, "adv_tap_hidden", 128))
        nl = int(getattr(e, "adv_tap_layers", 3))
        attrs = list(e.conditioning_attrs)
        self._adv_heads = torch.nn.ModuleList([
            torch.nn.ModuleDict({a: _mlp(int(d), _CAT_CLASSES.get(a, 1), hid, nl) for a in attrs})
            for d in e.hier_level_dims])
        print(f"adversary heads built in setup(): {len(self._adv_heads)} taps x {len(attrs)} "
              f"attrs, {nl}-layer MLP (hidden {hid}), dims {list(e.hier_level_dims)}", flush=True)
        return None

    def _conditioning_attr_indices(self):
        """Column indices in batch["attr"] for the conditioning attributes, in config order.

        A vector-valued attribute (AttrCondSpec.dim > 1, e.g. Causal3DIdent's 3-D pos_obj)
        occupies `dim` consecutive dataset columns named "<attr>_0..<attr>_{dim-1}"; a scalar
        attribute is looked up by its bare name exactly as before.
        """
        names = self.trainer.datamodule.attribute_names
        e = self.model.hdae_conf.encoder
        dims = {sp.name: int(getattr(sp, "dim", 1)) for sp in (e.cond_specs or [])}
        idx = []
        for name in e.conditioning_attrs:
            d = dims.get(name, 1)
            if d == 1:
                idx.append(names.index(name))
            else:
                idx.extend(names.index(f"{name}_{j}") for j in range(d))
        return idx

    def _batch_y_idx(self, batch):
        e = self.model.hdae_conf.encoder
        raw = batch["attr"][:, self._conditioning_attr_indices()]
        if not hasattr(self, "_logged_attr_values"):
            self._logged_attr_values = True
            print(f"HDAE raw attribute unique values sample: {observed_unique(raw)}")
        if e.cond_specs:
            return to_cond_values(raw, e.cond_specs).to(raw.device)
        return to_index_space(raw, e.attr_input_range).to(raw.device)

    def configure_optimizers(self):
        """Add the adversary heads to the optimiser.

        Upstream LitModel.configure_optimizers builds the optimiser from
        `self.model.parameters()` -- NOT `self.parameters()`. The adversary heads hang off the
        LightningModule rather than off self.model, so they were never eligible no matter when
        they were constructed. Two full 36,000-step runs trained an adversary that was frozen
        at random init before this was found, and both failed silently: a random probe emits
        uniform predictions, which is exactly what a perfectly-fooled probe emits, so
        adv/cat_acc sat at chance either way.
        """
        out = super().configure_optimizers()
        if not hasattr(self, "_adv_heads"):
            return out
        opt = out["optimizer"] if isinstance(out, dict) else out
        params = [q for q in self._adv_heads.parameters() if q.requires_grad]
        # weight_decay 0: the probes should be as strong as they can be. Decaying them would
        # hand the encoder an easier adversary and overstate how well invariance was achieved.
        # 5x the model lr. Measured head gradient norm is ~0.045, far under the 1.0 clip, so
        # the probes are not gradient-limited -- they are step-size-limited, and a WEAK
        # adversary is the dangerous failure here: it would be easy to fool and would overstate
        # how much attribute information the taps actually lost.
        adv_lr = 5.0 * self.conf.lr
        opt.add_param_group({"params": params, "lr": adv_lr, "weight_decay": 0.0})
        n = sum(q.numel() for q in params)
        print(f"optimiser: added {len(params)} adversary tensors ({n/1e6:.2f} M params) "
              f"at lr {adv_lr} (model lr {self.conf.lr})", flush=True)
        return out

    def on_before_optimizer_step(self, optimizer, optimizer_idx: int = 0) -> None:
        """Clip the diffusion model and the adversary heads as SEPARATE gradient vectors.

        Upstream clips every optimiser parameter together to max_norm=1.0. Concatenating a
        92M-parameter diffusion model with a 0.9M-parameter adversary and normalising the pair
        to unit norm leaves the heads with a vanishing share of it -- their effective learning
        rate collapses by orders of magnitude and they never leave random init. That is why
        head_loss sat at exactly its theoretical init value, (2*log10 + 1 + 1)/4 = 1.6513, for
        6,000 steps even after the heads were correctly registered with the optimiser.

        Clipping each group to its own max_norm keeps the model's behaviour identical to
        upstream while letting the probes actually train.
        """
        if self.conf.grad_clip <= 0:
            return
        head_ids = {id(q) for q in self._adv_heads.parameters()} \
            if hasattr(self, "_adv_heads") else set()
        model_params = [q for g in optimizer.param_groups for q in g["params"]
                        if id(q) not in head_ids]
        torch.nn.utils.clip_grad_norm_(model_params, max_norm=self.conf.grad_clip)
        if head_ids:
            # head_grad_norm is logged because a silently-frozen adversary is the failure
            # this whole feature hit four times; a norm near zero is the earliest honest signal.
            gn = torch.nn.utils.clip_grad_norm_(list(self._adv_heads.parameters()),
                                                max_norm=self.conf.grad_clip)
            self.log("adv/head_grad_norm", gn, sync_dist=True)

    def enable_compile(self, mode: str = "default"):
        """Compile a SEPARATE callable that shares self.model's parameters.

        self.model itself is deliberately left uncompiled. torch.compile returns an
        OptimizedModule whose state_dict keys carry a `_orig_mod.` prefix, and two things
        here index state_dict by key: upstream's ema() zips source.state_dict().keys() into
        target.state_dict()[key] (a compiled source against an uncompiled ema_model would
        KeyError on the first EMA step), and Lightning's checkpointing would bake the
        prefix into every saved checkpoint, making it unloadable by uncompiled code.
        Compiling a side handle avoids both: parameters are shared, not copied, so the
        compiled path trains exactly the same weights while state_dict stays clean.
        """
        # object.__setattr__ bypasses nn.Module.__setattr__, which would REGISTER the
        # OptimizedModule as a child module -- it then reappears in state_dict() as
        # `_compiled_model._orig_mod.*`, duplicating every weight in the checkpoint and
        # making strict load_state_dict into an uncompiled module fail on unexpected keys.
        # (Measured: 760 extra keys of 2281 before this fix.) Writing straight to __dict__
        # keeps the handle usable while leaving the module tree untouched.
        object.__setattr__(self, "_compiled_model", torch.compile(self.model, mode=mode))
        print(f"torch.compile enabled (mode={mode}); self.model left uncompiled so "
              f"checkpoints and EMA stay prefix-free", flush=True)
        return self._compiled_model

    @property
    def train_model(self):
        return self.__dict__.get("_compiled_model") or self.model

    def training_step(self, batch, batch_idx):
        if self.conf.train_mode != TrainMode.diffusion:
            return super().training_step(batch, batch_idx)
        with amp.autocast(False):
            x_start = batch["img"]
            y_idx = self._batch_y_idx(batch)
            zs = self.model.encode(x_start)
            t, _weight = self.T_sampler.sample(len(x_start), x_start.device)
            losses = self.sampler.training_losses(
                model=self.train_model,
                x_start=x_start,
                t=t,
                model_kwargs={"cond": self.model.make_cond(zs, y_idx)},
            )
            loss = losses["loss"].mean()
            adv = self._adv_loss(zs, batch)
            if adv is not None:
                loss = loss + adv
            for key in ["loss", "vae", "latent", "mmd", "chamfer", "arg_cnt"]:
                if key in losses:
                    losses[key] = self.all_gather(losses[key]).mean()
            if self.global_rank == 0:
                self.logger.experiment.add_scalar("loss", losses["loss"], self.num_samples)
        self._log_latents()
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx=0) -> None:
        if not self.is_last_accum(batch_idx):
            return
        ema(self.model, self.ema_model, self.conf.ema_decay)

    def _adv_targets(self, batch):
        """Per-attribute regression/classification targets from the RAW attribute columns.

        Continuous attributes are standardised by dataset statistics computed once from the
        training split, not per batch: a per-batch mean makes the target a moving goalpost and
        the adversary chases its own normalisation instead of the attribute.
        """
        e = self.model.hdae_conf.encoder
        names = self.trainer.datamodule.attribute_names
        raw = batch["attr"]
        if not hasattr(self, "_adv_stats"):
            import h5py
            with h5py.File(self.trainer.datamodule.h5_path, "r") as h:
                A = h["attrs"][:].astype("float64")
            self._adv_stats = {a: (float(A[:, names.index(a)].mean()),
                                   float(A[:, names.index(a)].std()) + 1e-8)
                               for a in e.conditioning_attrs if a not in _CAT_CLASSES}
        out = {}
        for a in e.conditioning_attrs:
            v = raw[:, names.index(a)].float()
            if a in _CAT_CLASSES:
                idx = torch.round(v) if a == "digit" else torch.round(v * 10.0 - 0.5)
                out[a] = idx.clamp(0, _CAT_CLASSES[a] - 1).long()
            else:
                mu, sd = self._adv_stats[a]
                out[a] = ((v - mu) / sd).unsqueeze(1)
        return out

    def _adv_loss(self, zs, batch):
        """Confusion-style adversary over every tap x every conditioning attribute.

        NOT plain gradient reversal. A first attempt negated the adversary's cross-entropy into
        the encoder and diverged within 400 steps (k=11 loss 1.79 -> 36.1 and climbing, k=1
        spiking to 505): negated CE is UNBOUNDED ABOVE, so the cheapest way for the encoder to
        win is to make the taps confidently wrong or simply enormous, which destroys the latent
        instead of scrubbing the attribute from it.

        The bounded formulation (Tzeng et al.) splits the two objectives:
          heads   minimise their own error on DETACHED taps -- honest probes, unaffected by the
                  encoder's objective.
          encoder minimises distance to an UNINFORMATIVE prediction: cross-entropy against the
                  uniform distribution for the two categorical attributes (bounded by log 10),
                  and squared error toward 0 -- the standardised mean -- for the continuous
                  ones. Both are bounded below at their optimum, so "attribute is gone" is a
                  reachable minimum rather than a direction to run in forever.

        The encoder term is computed through torch.func.functional_call with detached head
        parameters, so pushing predictions toward uniform updates the ENCODER only and never
        weakens the probes that are meant to keep it honest.
        """
        e = self.model.hdae_conf.encoder
        lam_max = float(getattr(e, "adv_tap_lambda", 0.0) or 0.0)
        if lam_max <= 0.0:
            return None
        attrs = list(e.conditioning_attrs)
        assert hasattr(self, "_adv_heads"), "adversary heads missing -- setup() did not run"
        if not hasattr(self, "_adv_checked"):
            self._adv_checked = True
            opt_ids = {id(q) for g in self.optimizers().param_groups for q in g["params"]} \
                if self.optimizers() is not None else set()
            head_ids = {id(q) for q in self._adv_heads.parameters()}
            missing = len(head_ids - opt_ids)
            if missing:
                raise RuntimeError(
                    f"{missing}/{len(head_ids)} adversary tensors are NOT in the optimiser -- "
                    f"the adversary would train nothing and fail silently (a random probe and "
                    f"a fooled probe both sit at chance accuracy). Refusing to waste the run.")
            print(f"adversary check: all {len(head_ids)} head tensors are in the optimiser",
                  flush=True)
        total_steps = int(getattr(self.trainer, "max_steps", 0) or 0)
        frac = float(getattr(e, "adv_tap_warmup_frac", 0.3))
        warm = max(1, int(total_steps * frac)) if total_steps > 0 else 1
        lam = lam_max * min(1.0, int(self.global_step) / warm)
        tg = self._adv_targets(batch)

        head_loss, conf_loss, accs = 0.0, 0.0, []
        for ti, z in enumerate(zs):
            zf = z.float()
            zd = zf.detach()
            for a in attrs:
                head = self._adv_heads[ti][a]
                p_det = head(zd)                       # trains the head, no encoder gradient
                params = {k: v.detach() for k, v in head.named_parameters()}
                buffers = {k: v.detach() for k, v in head.named_buffers()}
                p_enc = torch.func.functional_call(head, {**params, **buffers}, (zf,))
                if a in _CAT_CLASSES:
                    head_loss = head_loss + torch.nn.functional.cross_entropy(p_det, tg[a])
                    # uniform target == maximise entropy, bounded by log(n_classes)
                    conf_loss = conf_loss - torch.log_softmax(p_enc, dim=1).mean(1).mean()
                    accs.append((p_det.argmax(1) == tg[a]).float().mean().detach())
                else:
                    head_loss = head_loss + torch.nn.functional.mse_loss(p_det, tg[a])
                    conf_loss = conf_loss + (p_enc ** 2).mean()
        n = len(zs) * len(attrs)
        head_loss, conf_loss = head_loss / n, conf_loss / n
        self.log("adv/lambda", lam, sync_dist=True)
        self.log("adv/head_loss", head_loss.detach(), sync_dist=True)
        self.log("adv/conf_loss", conf_loss.detach(), sync_dist=True)
        if accs:
            self.log("adv/cat_acc", torch.stack(accs).mean(), sync_dist=True)
        return head_loss + lam * conf_loss

    def _log_latents(self):
        for i, z in enumerate(self.model.last_zs):
            self.log(f"latent/norm_{i}", z.norm(dim=1).mean(), sync_dist=True)
