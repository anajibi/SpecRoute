from pathlib import Path

import torch

from diffae_tools.model_loader import DiffAEModelWrapper


class FakeEmaModel:
    def eval(self):
        return self

    def to(self, device):
        return self


class FakeLitModel:
    def __init__(self):
        self.ema_model = FakeEmaModel()

    def load_state_dict(self, state_dict, strict=False):
        return None

    def eval(self):
        return self

    def encode(self, x):
        return torch.ones(x.shape[0], 4)

    def encode_stochastic(self, x, cond, T=None):
        return {"sample": torch.zeros(x.shape[0], 3, 256, 256)}

    def render(self, stochastic, cond, T=None):
        return torch.clamp((stochastic + 1.0) / 2.0, 0.0, 1.0)


def test_wrapper_load_encode_and_reconstruct(tmp_path: Path, monkeypatch):
    ckpt = tmp_path / "last.ckpt"
    torch.save({"state_dict": {}}, ckpt)

    wrapper = DiffAEModelWrapper(repo_root=tmp_path, checkpoint_path=ckpt, device="cpu")
    monkeypatch.setattr(wrapper, "_instantiate_official_model", lambda: (object(), FakeLitModel()))
    wrapper.load()

    image = torch.zeros(1, 3, 256, 256)
    z_sem = wrapper.encode_semantic(image)
    stochastic = wrapper.encode_stochastic(image, z_sem=z_sem)
    recon = wrapper.decode_from_latents(z_sem, stochastic)

    assert z_sem.shape == (1, 4)
    assert stochastic.shape == (1, 3, 256, 256)
    assert recon.shape == (1, 3, 256, 256)
    assert torch.isfinite(recon).all()


