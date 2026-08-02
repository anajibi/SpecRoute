"""Model-agnostic contract for counterfactual-image generation.

Any generative model that can be scored by CC/FC/CF1 (``run_cf1_eval.py``)
implements ``CFModelAdapter``. A ``CFState`` is an opaque per-model bundle
of whatever the adapter needs to go from "encoded source image" to
"rendered image": semantic latents, abducted x_T noise, conditioning
tensors, etc. ``run_cf1_eval.py`` never inspects a state's contents.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch

CFState = Dict[str, Any]

_ADAPTER_REGISTRY: Dict[str, type] = {}


def register_adapter(model_type: str):
    def _decorator(cls):
        _ADAPTER_REGISTRY[model_type] = cls
        return cls
    return _decorator


def load_adapter(model_type: str, config_path: str, ckpt_path: str, device: str, **kwargs) -> "CFModelAdapter":
    if model_type not in _ADAPTER_REGISTRY:
        raise ValueError(f"unknown model_type {model_type!r}; registered: {sorted(_ADAPTER_REGISTRY)}")
    return _ADAPTER_REGISTRY[model_type].load(config_path, ckpt_path, device, **kwargs)


class CFModelAdapter(ABC):
    """Contract every CF-scorable model must implement.

    ``modeled_attrs`` is the fixed list of CelebA attribute names this
    model can condition on / edit, in a stable order the adapter chooses
    internally (callers must not assume an order and should look up
    columns by name).

    ``edit_strength`` is the model's own edit-strength hyperparameter (HDAE's
    attribute-CFG guidance scale, DiffAEProbeAdapter's alpha, ...) — declared
    here rather than duck-typed so the cross-model comparability column in
    ``cf1_aggregate.csv``/``cf1_per_intervention.csv`` can't silently become
    ``nan`` for an adapter that names its knob something else.
    """

    modeled_attrs: list
    edit_strength: float

    @classmethod
    @abstractmethod
    def load(cls, config_path: str, ckpt_path: str, device: str, **kwargs) -> "CFModelAdapter":
        """Load model weights/config and return a ready-to-use adapter."""

    @abstractmethod
    def encode(self, images: torch.Tensor, attrs_raw: torch.Tensor, attr_names: list) -> CFState:
        """Encode a batch of images (NCHW, dataset-native range [-1, 1]) into a per-model state.

        ``attrs_raw`` is the full CelebA attribute matrix for this batch
        (B, len(attr_names)) in the dataset's native value range (may be
        {-1,1} or {0,1} depending on the source; adapters that care about
        the exact range convert internally, e.g. via ``to_index_space``).
        Adapters that need ground-truth attributes to build a
        conditioning signal (e.g. HDAE) look up their own
        ``modeled_attrs`` columns by name here. Adapters that don't
        condition on attributes at all (e.g. a frozen probe-edited model)
        may ignore ``attrs_raw``/``attr_names`` entirely.

        Must perform both semantic encoding and any x_T abduction the
        model needs, together, since abduction generally requires the
        exact model instance/conditioning that produced the semantic
        code (this mirrors HDAE's current constraint; a single call
        avoids adapters needing to smuggle that instance across a second
        call).
        """

    @abstractmethod
    def intervene(self, state: CFState, attr: str, direction: str, cf_attrs: Dict[str, torch.Tensor]) -> CFState:
        """Return a new state with ``attr`` (and, per the causal graph, its descendants) counterfactually set.

        ``direction`` is ``"positive"`` (set/push the attribute on) or
        ``"negative"`` (set/push it off). ``attr`` must be in
        ``self.modeled_attrs``.

        ``cf_attrs`` is the SCM-propagated (TODO item 2, ``causal/scm.py``)
        counterfactual value for *every* entry in ``self.modeled_attrs`` —
        one ``(batch, 1)`` tensor of {0.0, 1.0} per attribute name, keyed by
        name (order-independent). ``attr`` itself is forced to
        ``direction``'s target side; causal descendants of ``attr`` carry
        their propagated value; everything else equals its original,
        observed value (a no-op for attributes the declared causal graph
        has no edges into/out of — e.g. the whole vector when the graph is
        edgeless). Adapters that don't condition on an explicit attribute
        vector (e.g. a frozen probe-edited model) may ignore ``cf_attrs``
        and use ``attr``/``direction`` only — accept the parameter anyway,
        the caller always passes it.
        """

    @abstractmethod
    def render(self, state: CFState) -> torch.Tensor:
        """Render a state to images, NCHW, values approximately in [0, 1] (unclamped)."""
