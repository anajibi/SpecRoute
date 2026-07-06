"""Lightning module preserving upstream DiffAE sampler/EMA/optimizer."""
from torch.cuda import amp

from choices import TrainMode
from experiment import LitModel

from .attr_utils import observed_unique, to_index_space


class HDAELitModule(LitModel):
    """Upstream LitModel operating on a config-selected hierarchical model."""

    def setup(self, stage=None):
        """Datasets are supplied by the external packed-data DataModule."""
        return None

    def _conditioning_attr_indices(self):
        e = self.model.hdae_conf.encoder
        if not e.conditioning_attrs:
            return list(range(e.n_attributes))
        names = getattr(getattr(self, "trainer", None), "datamodule", None)
        names = getattr(names, "attribute_names", None)
        if names is None:
            return list(range(e.n_attributes))
        missing = [name for name in e.conditioning_attrs if name not in names]
        if missing:
            raise ValueError(f"conditioning_attrs not found in datamodule attributes: {missing}")
        return [names.index(name) for name in e.conditioning_attrs]

    def _batch_y_idx(self, batch):
        e = self.model.hdae_conf.encoder
        if "attr" not in batch:
            return None
        raw = batch["attr"][:, self._conditioning_attr_indices()]
        if not hasattr(self, "_logged_attr_values"):
            self._logged_attr_values = True
            print(f"HDAE raw attribute unique values sample: {observed_unique(raw)}")
        y_idx = to_index_space(raw, e.attr_input_range).to(raw.device)
        if y_idx.numel() and int(y_idx.max()) > 1:
            raise ValueError("attribute normalization must produce indices <= 1 before CFG dropout")
        return y_idx

    def training_step(self, batch, batch_idx):
        """Run the upstream diffusion loss, adding attribute indices to the model."""
        if self.conf.train_mode != TrainMode.diffusion or "img" not in batch:
            result = super().training_step(batch, batch_idx)
            self._log_latents()
            return result
        with amp.autocast(False):
            x_start = batch["img"]
            t, weight = self.T_sampler.sample(len(x_start), x_start.device)
            model_kwargs = {}
            y_idx = self._batch_y_idx(batch)
            if y_idx is not None:
                model_kwargs["y_idx"] = y_idx
            losses = self.sampler.training_losses(model=self.model, x_start=x_start, t=t,
                                                  model_kwargs=model_kwargs)
            loss = losses["loss"].mean()
            for key in ["loss", "vae", "latent", "mmd", "chamfer", "arg_cnt"]:
                if key in losses:
                    losses[key] = self.all_gather(losses[key]).mean()
            if self.global_rank == 0:
                self.logger.experiment.add_scalar("loss", losses["loss"], self.num_samples)
        self._log_latents()
        return loss

    def _log_latents(self):
        zs = getattr(self.model, "last_zs", None)
        if zs:
            for i, z in enumerate(zs):
                self.log(f"latent/norm_{i}", z.norm(dim=1).mean(), sync_dist=True)
        null_mask = getattr(self.model, "last_null_mask", None)
        if null_mask is not None:
            for i in range(null_mask.shape[1]):
                self.log(f"latent/null_rate_{i}", null_mask[:, i].float().mean(), sync_dist=True)
