"""Lightning module for conditional HDAE training."""
from torch.cuda import amp

from choices import TrainMode
import torch

from experiment import LitModel, ema

from .attr_utils import observed_unique, to_cond_values, to_index_space


class HDAELitModule(LitModel):
    def setup(self, stage=None):
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

    def _log_latents(self):
        for i, z in enumerate(self.model.last_zs):
            self.log(f"latent/norm_{i}", z.norm(dim=1).mean(), sync_dist=True)
