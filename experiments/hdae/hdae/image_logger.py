"""Lightning callback: log reconstructions and counterfactuals to TensorBoard during training.

Training loss alone cannot tell you whether conditioning is working -- a model can drive
the diffusion loss down while ignoring the attribute signal entirely, which is exactly the
failure this project already hit once on MorphoMNIST. Rendering a fixed cohort every few
epochs makes that visible while the run is still in progress rather than 35 hours later.

Two image grids are written each time:

  recon/grid   source images beside their reconstruction (DDIM-encode -> decode, no
               guidance). Watches image quality.
  cf/<name>    the same sources rendered under an intervention, with the counterfactual
               attribute vector produced by the trained SCM's abduct -> intervene ->
               predict path, so descendants propagate. Watches conditioning strength.

The cohort is fixed across the whole run (same images, same seed), so successive
TensorBoard steps are directly comparable frame to frame.

Everything runs under no_grad on the EMA weights and restores train/eval mode afterwards,
so it cannot perturb training. Any failure is caught and logged as a warning rather than
killing a long run over a visualisation.
"""
import os
import traceback

import torch

try:
    import pytorch_lightning as pl
    _Base = pl.Callback
except ImportError:
    _Base = object

from .attr_utils import to_cond_values

# model conditioning column layout for Causal3DIdent: class(1) pos_spl(1) pos_obj(3) rot_obj(3)
_COLS = {"class": [0], "pos_spl": [1], "pos_obj": [2, 3, 4], "rot_obj": [5, 6, 7]}
_ORDER = ["class", "pos_spl", "pos_obj", "rot_obj"]

DEFAULT_INTERVENTIONS = [
    ("class_dragon", "class", 2.0),
    ("pos_spl_pos", "pos_spl", 0.9),
    ("pos_spl_neg", "pos_spl", -0.9),
    ("rot_obj_pos", "rot_obj", 0.9),
]


class ImageLogCallback(_Base):
    def __init__(self, h5_path, every_n_epochs=3, n_images=4, T=100, guidance=3.0,
                 scm_path=None, seed=0, interventions=None):
        super().__init__()
        self.h5_path = h5_path
        self.every_n_epochs = int(every_n_epochs)
        self.n_images = int(n_images)
        self.T = int(T)
        self.guidance = float(guidance)
        self.scm_path = scm_path
        self.seed = int(seed)
        self.interventions = interventions or DEFAULT_INTERVENTIONS
        self._cohort = None
        self._scm = None
        self._scm_tried = False

    # -- lazy setup ------------------------------------------------

    def _load_cohort(self, pl_module):
        from experiments.hdae.data.causal3dident import Causal3DIdentPacked
        import numpy as np
        ds = Causal3DIdentPacked(self.h5_path)
        idx = sorted(np.random.RandomState(self.seed)
                     .choice(len(ds), self.n_images, replace=False).tolist())
        batch = [ds[i] for i in idx]
        x = torch.stack([b["img"] for b in batch])
        attr = torch.stack([b["attr"] for b in batch])[:, :8]
        specs = pl_module.model.hdae_conf.encoder.cond_specs
        self._cohort = (x, to_cond_values(attr, specs))

    def _load_scm(self, device):
        """Optional: without it we still log reconstructions, just no counterfactuals."""
        if self._scm_tried:
            return self._scm
        self._scm_tried = True
        if not self.scm_path or not os.path.exists(self.scm_path):
            print(f"[ImageLogCallback] no SCM at {self.scm_path!r}; "
                  f"interventions will set the target column directly without propagating "
                  f"to descendants", flush=True)
            return None
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "causal"))
            from train_scm_causal3dident import SCM, CausalGraph
            blob = torch.load(self.scm_path, map_location=device)
            c = blob["config"]
            scm = SCM(CausalGraph(c["attributes"], c["edges"]), c["nodes"],
                      mechanism=blob.get("mechanism", "gaussian"), bins=blob.get("bins", 16)).to(device)
            scm.load_state_dict(blob["state_dict"]); scm.eval()
            self._scm = scm
        except Exception:
            print("[ImageLogCallback] SCM load failed:\n" + traceback.format_exc(), flush=True)
        return self._scm

    # -- rendering -------------------------------------------------

    def _cf_attrs(self, y, name, col, value, device):
        scm = self._load_scm(device)
        if scm is None:
            y_cf = y.clone()
            y_cf[:, _COLS[col]] = value
            return y_cf
        obs = {k: y[:, _COLS[k]].contiguous() for k in _ORDER}
        iv = {col: torch.full((1, len(_COLS[col])), value, device=device)}
        cf = scm.propagate(scm.abduct(obs), obs, iv)
        y_cf = y.clone()
        for k in _ORDER:
            y_cf[:, _COLS[k]] = cf[k].to(y.dtype)
        return y_cf

    @torch.no_grad()
    def _render(self, pl_module, x_T, cond, guidance):
        from experiments.hdae.counterfactuals.hdae_adapter import AttributeCFGWrapper
        model = pl_module.ema_model
        m = model if guidance == 1.0 else AttributeCFGWrapper(model, guidance).to(x_T.device).eval()
        out = self._sampler(pl_module).sample(model=m, noise=x_T, model_kwargs={"cond": cond})
        return (out + 1) / 2

    def _sampler(self, pl_module):
        return pl_module.conf._make_diffusion_conf(self.T).make_sampler()

    # -- hook ------------------------------------------------------

    def on_train_epoch_end(self, trainer, pl_module):
        ep = trainer.current_epoch + 1
        if ep % self.every_n_epochs != 0:
            return
        logger = getattr(trainer, "logger", None)
        if logger is None or not hasattr(logger, "experiment"):
            return
        try:
            self._log(trainer, pl_module, logger, ep)
        except Exception:
            print(f"[ImageLogCallback] epoch {ep} failed (training continues):\n"
                  + traceback.format_exc(), flush=True)

    def _log(self, trainer, pl_module, logger, ep):
        import torchvision
        device = pl_module.device
        if self._cohort is None:
            self._load_cohort(pl_module)
        x, y = self._cohort
        x, y = x.to(device), y.to(device)
        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                sampler = self._sampler(pl_module)
                zs = [z.clone() for z in pl_module.ema_model.encode(x)]
                cond = pl_module.ema_model.make_cond(zs, y)
                x_T = sampler.ddim_reverse_sample_loop(
                    pl_module.ema_model, x, model_kwargs={"cond": cond})["sample"]
                recon = self._render(pl_module, x_T, cond, 1.0)
                grid = torchvision.utils.make_grid(
                    torch.cat([(x + 1) / 2, recon.clamp(0, 1)]), nrow=self.n_images)
                logger.experiment.add_image("recon/source_top_recon_bottom", grid,
                                            global_step=trainer.global_step)
                for name, col, val in self.interventions:
                    y_cf = self._cf_attrs(y, name, col, val, device)
                    cond_cf = pl_module.ema_model.make_cond(zs, y_cf)
                    img = self._render(pl_module, x_T, cond_cf, self.guidance)
                    g = torchvision.utils.make_grid(img.clamp(0, 1), nrow=self.n_images)
                    logger.experiment.add_image(f"cf/{name}", g, global_step=trainer.global_step)
            print(f"[ImageLogCallback] logged recon + {len(self.interventions)} "
                  f"counterfactual grids at epoch {ep} (step {trainer.global_step})", flush=True)
        finally:
            if was_training:
                pl_module.train()
