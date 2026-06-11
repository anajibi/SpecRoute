"""Lightning module preserving upstream DiffAE loss, sampler, EMA, and optimizer."""
from experiment import LitModel


class HDAELitModule(LitModel):
    """Upstream LitModel operating on a config-selected hierarchical model.

    ``LitModel.training_step``, ``on_train_batch_end`` and
    ``configure_optimizers`` are intentionally inherited unchanged.
    """
    def setup(self, stage=None):
        """Datasets are supplied by the external packed-data DataModule."""
        return None

    def training_step(self, batch, batch_idx):
        result = super().training_step(batch, batch_idx)
        zs = getattr(self.model, "last_zs", None)
        if zs:
            for i, z in enumerate(zs):
                self.log(f"latent/norm_{i}", z.norm(dim=1).mean(), sync_dist=True)
        return result
