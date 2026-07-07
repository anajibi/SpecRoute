"""Lightning module for conditional HDAE training."""
from torch.cuda import amp

from choices import TrainMode
from experiment import LitModel

from .attr_utils import observed_unique, to_index_space


class HDAELitModule(LitModel):
    def setup(self, stage=None):
        return None

    def _conditioning_attr_indices(self):
        names = self.trainer.datamodule.attribute_names
        return [names.index(name) for name in self.model.hdae_conf.encoder.conditioning_attrs]

    def _batch_y_idx(self, batch):
        e = self.model.hdae_conf.encoder
        raw = batch["attr"][:, self._conditioning_attr_indices()]
        if not hasattr(self, "_logged_attr_values"):
            self._logged_attr_values = True
            print(f"HDAE raw attribute unique values sample: {observed_unique(raw)}")
        return to_index_space(raw, e.attr_input_range).to(raw.device)

    def training_step(self, batch, batch_idx):
        if self.conf.train_mode != TrainMode.diffusion:
            return super().training_step(batch, batch_idx)
        with amp.autocast(False):
            x_start = batch["img"]
            y_idx = self._batch_y_idx(batch)
            zs = self.model.encode(x_start)
            t, _weight = self.T_sampler.sample(len(x_start), x_start.device)
            losses = self.sampler.training_losses(
                model=self.model,
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

    def _log_latents(self):
        for i, z in enumerate(self.model.last_zs):
            self.log(f"latent/norm_{i}", z.norm(dim=1).mean(), sync_dist=True)
