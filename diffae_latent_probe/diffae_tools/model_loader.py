from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import logging
import sys

import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class LoadedDiffAE:
    conf: object
    model: object
    checkpoint_state: dict


class DiffAEModelWrapper:
    """
    Thin adapter around the official DiffAE autoencoding API.

    Important:
    - Computation happens on self.device.
    - Outputs stay on GPU by default.
    - The calling script should explicitly call .detach().cpu().numpy()
      only when saving.
    """

    def __init__(self, repo_root, checkpoint_path, device):
        self.repo_root = Path(repo_root)
        self.checkpoint_path = Path(checkpoint_path)

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable. Falling back to CPU.")
            requested_device = torch.device("cpu")

        self.device = requested_device
        self._loaded: Optional[LoadedDiffAE] = None
        self._default_reverse_steps = 250
        self._default_render_steps = 20

    def _ensure_repo_on_path(self):
        repo_root = str(self.repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    def _import_official_modules(self):
        self._ensure_repo_on_path()
        import templates  # type: ignore
        import experiment  # type: ignore

        return templates, experiment

    def _instantiate_official_model(self):
        templates, experiment = self._import_official_modules()
        conf = templates.ffhq256_autoenc()

        # Keep inference side-effect free: we load checkpoint manually.
        conf.pretrain = None
        conf.latent_infer_path = None
        conf.eval_programs = tuple()

        model = experiment.LitModel(conf)
        return conf, model

    def load(self):
        if not self.repo_root.exists():
            raise FileNotFoundError(f"DiffAE repo_root does not exist: {self.repo_root}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"DiffAE checkpoint not found: {self.checkpoint_path}")

        conf, model = self._instantiate_official_model()

        # PyTorch 2.x-safe loading for trusted local research checkpoints.
        try:
            state = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            state = torch.load(self.checkpoint_path, map_location="cpu")

        if "state_dict" not in state:
            raise KeyError(f"Checkpoint does not contain 'state_dict': {self.checkpoint_path}")

        load_result = model.load_state_dict(state["state_dict"], strict=False)
        if load_result is None:
            missing, unexpected = [], []
        else:
            missing, unexpected = load_result

        if missing:
            LOGGER.warning("Missing checkpoint keys: %d", len(missing))
        if unexpected:
            LOGGER.warning("Unexpected checkpoint keys: %d", len(unexpected))

        # CRITICAL FIX:
        # Move the entire Lightning model to GPU, not only ema_model.
        if hasattr(model, "to"):
            model = model.to(self.device)
        else:
            LOGGER.warning("Model does not implement .to(); leaving on default device.")

        # Also move ema_model explicitly, because DiffAE often uses EMA for inference.
        if hasattr(model, "ema_model"):
            if hasattr(model.ema_model, "to"):
                model.ema_model = model.ema_model.to(self.device)
            model.ema_model.eval()

        model.eval()

        # Sanity check.
        if hasattr(model, "parameters"):
            try:
                first_param = next(model.parameters())
                LOGGER.info("[DiffAEModelWrapper] model parameter device: %s", first_param.device)
            except StopIteration:
                LOGGER.warning("[DiffAEModelWrapper] model has no parameters.")
        else:
            LOGGER.warning("[DiffAEModelWrapper] model has no parameters() method.")

        if hasattr(model, "ema_model"):
            if hasattr(model.ema_model, "parameters"):
                try:
                    first_ema_param = next(model.ema_model.parameters())
                    LOGGER.info("[DiffAEModelWrapper] ema_model parameter device: %s", first_ema_param.device)
                except StopIteration:
                    LOGGER.warning("[DiffAEModelWrapper] ema_model has no parameters.")
            else:
                LOGGER.warning("[DiffAEModelWrapper] ema_model has no parameters() method.")

        if self.device.type == "cuda":
            LOGGER.info("[DiffAEModelWrapper] CUDA device: %s", torch.cuda.get_device_name(self.device))

        self._loaded = LoadedDiffAE(conf=conf, model=model, checkpoint_state=state)
        return self

    @property
    def conf(self):
        if self._loaded is None:
            raise RuntimeError("Call load() before using the wrapper.")
        return self._loaded.conf

    @property
    def model(self):
        if self._loaded is None:
            raise RuntimeError("Call load() before using the wrapper.")
        return self._loaded.model

    def _ensure_loaded(self):
        if self._loaded is None:
            raise RuntimeError("DiffAEModelWrapper is not loaded. Call load() first.")

    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        if x.ndim == 3:
            x = x.unsqueeze(0)

        x = x.to(self.device, non_blocking=True)

        # Align floating-point inputs with the model's parameter dtype to prevent matmul mismatches.
        if torch.is_floating_point(x) and self._loaded is not None and hasattr(self.model, "parameters"):
            try:
                model_dtype = next(self.model.parameters()).dtype
            except StopIteration:
                model_dtype = None
            if model_dtype is not None and x.dtype != model_dtype:
                x = x.to(model_dtype)

        return x

    def encode_semantic(self, images, return_cpu: bool = False):
        self._ensure_loaded()
        images = self._to_device(images)

        with torch.inference_mode():
            z_sem = self.model.encode(images)

        if isinstance(z_sem, dict):
            z_sem = z_sem.get("cond", z_sem)

        z_sem = z_sem.detach()

        if return_cpu:
            z_sem = z_sem.cpu()

        return z_sem

    def encode_stochastic(self, images, z_sem=None, return_cpu: bool = False):
        self._ensure_loaded()
        images = self._to_device(images)

        if z_sem is None:
            z_sem = self.encode_semantic(images, return_cpu=False)
        else:
            z_sem = self._to_device(z_sem)

        with torch.inference_mode():
            out = self.model.encode_stochastic(
                images,
                z_sem,
                T=self._default_reverse_steps,
            )

        if isinstance(out, dict):
            stochastic = out.get("sample", out)
        else:
            stochastic = out

        stochastic = stochastic.detach()

        if return_cpu:
            stochastic = stochastic.cpu()

        return stochastic

    def decode_from_latents(self, z_sem, stochastic, return_cpu: bool = False):
        self._ensure_loaded()
        z_sem = self._to_device(z_sem)
        stochastic = self._to_device(stochastic)

        with torch.inference_mode():
            recon = self.model.render(
                stochastic,
                z_sem,
                T=self._default_render_steps,
            )

        recon = recon.detach()

        if return_cpu:
            recon = recon.cpu()

        return recon

    def reconstruct(self, images=None, z_sem=None, stochastic=None, ddim_steps=None, return_cpu: bool = False):
        self._ensure_loaded()

        if images is not None:
            images = self._to_device(images)

        if z_sem is None:
            if images is None:
                raise ValueError("reconstruct requires images or z_sem")
            z_sem = self.encode_semantic(images, return_cpu=False)

        if stochastic is None:
            if images is None:
                raise ValueError("reconstruct requires images or stochastic")
            stochastic = self.encode_stochastic(images, z_sem=z_sem, return_cpu=False)

        if ddim_steps is not None:
            prev_steps = self._default_render_steps
            self._default_render_steps = ddim_steps
            try:
                return self.decode_from_latents(z_sem, stochastic, return_cpu=return_cpu)
            finally:
                self._default_render_steps = prev_steps

        return self.decode_from_latents(z_sem, stochastic, return_cpu=return_cpu)

    def state_summary(self) -> str:
        if self._loaded is None:
            return "unloaded"

        return json.dumps(
            {
                "repo_root": str(self.repo_root),
                "checkpoint_path": str(self.checkpoint_path),
                "device": str(self.device),
                "model_name": getattr(self.conf, "name", None),
            },
            indent=2,
        )

