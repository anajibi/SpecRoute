import ast
from pathlib import Path

import torch

from dino_vae_hierarchical_diffusion.src.evidence import EvidencePyramid


EXPERIMENT_ROOT = Path(__file__).parents[1]
STAGE1_PATH = EXPERIMENT_ROOT / "scripts" / "train_stage1_autoencoder.py"


def test_evidence_pyramid_uses_zero_dino_fallback():
    evidence = EvidencePyramid()(torch.randn(2, 4, 32, 32))
    expected_shapes = {
        "D16": (2, 256, 16, 16),
        "D8": (2, 256, 8, 8),
        "D4": (2, 256, 4, 4),
        "Dg": (2, 512),
    }
    for name, shape in expected_shapes.items():
        assert evidence[name].shape == shape
        assert torch.count_nonzero(evidence[name]) == 0


def test_stage1_source_is_vae_only_latent_reconstruction():
    source = STAGE1_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "setup(args, use_dino=False)" in source
    assert 'modules = {"evidence": evidence, "encoder": encoder, "deterministic": deterministic}' in source
    assert "F.l1_loss(z0_deterministic, z0)" in source
    assert "s0.loss" not in source
    assert "vae.decode" not in source
    assert "highpass" not in source
    assert '"s0":' not in source
