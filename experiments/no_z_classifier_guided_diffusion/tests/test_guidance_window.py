from __future__ import annotations

import importlib.util

import pytest

_HAS_DIFFUSERS = importlib.util.find_spec("diffusers") is not None
pytestmark = pytest.mark.skipif(not _HAS_DIFFUSERS, reason="diffusers is required to import the editing driver")

if _HAS_DIFFUSERS:
    from experiments.no_z_classifier_guided_diffusion.scripts.run_guided_editing_poc import _resolve_guidance_window


def test_resolve_guidance_window_uses_fractions() -> None:
    assert _resolve_guidance_window({"guidance_start_fraction": 0.30, "guidance_end_fraction": 0.90}, 700) == (210, 630)


def test_resolve_guidance_window_keeps_step_override_compatibility() -> None:
    assert _resolve_guidance_window({"guidance_start_step": 50, "guidance_end_step": 450}, 700) == (50, 450)


def test_resolve_guidance_window_rejects_invalid_order() -> None:
    with pytest.raises(ValueError, match="Invalid guidance window"):
        _resolve_guidance_window({"guidance_start_fraction": 0.9, "guidance_end_fraction": 0.3}, 700)
