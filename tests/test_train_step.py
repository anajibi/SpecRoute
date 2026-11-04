"""Tier 4 -- one optimizer step, end to end.

Skipped while the GPU is down (the box rebooted into a kernel whose NVIDIA module did not load, so
`torch.cuda.is_available()` is False). Written now so it runs the moment the driver is back.

The reason this tier exists: `accum_batches` gates the EMA update via `conf.accum_batches`, so
setting it on the Trainer alone advances the EMA eight times too fast. That corrupts every
evaluation and nothing raises -- the loss curve looks normal. Only watching the EMA actually move
once per accumulation window catches it.
"""
import pytest
import torch
from able import registry as R

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="no CUDA device; nvidia module not loaded"),
]


@pytest.mark.parametrize("name", ["c3di_k5", "celebahq_a_k11", "celebahq_b_k1"])
def test_single_step_produces_a_finite_loss(name):
    """One synthetic batch through the real loss, on one model per dataset and graph."""
    m = R.get(name)
    lit, cfg = R.build(m)
    lit = lit.cuda().train()
    n_attr = cfg.raw["encoder"]["n_attributes"]
    size = cfg.raw["data"]["image_size"]
    batch = {
        "img": torch.randn(2, 3, size, size, device="cuda").clamp(-1, 1),
        "attr": torch.randint(0, 2, (2, n_attr), device="cuda").float() * 2 - 1,
        "index": torch.tensor([0, 1], device="cuda"),
    }
    loss = lit.training_step(batch, 0)
    assert torch.isfinite(loss), f"{name}: loss is {loss.item()}"
    assert loss.item() > 0, f"{name}: loss {loss.item()} should be positive for random input"


def test_ema_advances_once_per_accumulation_window():
    """The trap: with accum_batches=8, the EMA must move on the 8th micro-batch, not on each."""
    m = R.get("celebahq_b_k1")
    lit, cfg = R.build(m)
    accum = cfg.raw["train"]["accum_batches"]
    assert accum == 8, f"expected the protocol's accum_batches=8, config says {accum}"
    assert lit.conf.accum_batches == accum, (
        "conf.accum_batches must match the config: it gates the EMA update, and a Trainer-only "
        "setting advances the EMA 8x too fast while the loss curve still looks normal")

    lit = lit.cuda()
    before = next(iter(lit.ema_model.parameters())).detach().clone()
    for i in range(accum - 1):
        assert not lit.is_last_accum(i), f"micro-batch {i} should not close the window"
    assert lit.is_last_accum(accum - 1), "the 8th micro-batch must close the window"
    after = next(iter(lit.ema_model.parameters())).detach()
    assert torch.equal(before, after), "EMA must not move before the window closes"
