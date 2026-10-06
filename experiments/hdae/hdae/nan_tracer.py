"""Opt-in tracer that pins down where a run first goes non-finite.

The celebahq256_k11_cd015 continuation went NaN at optimiser step ~423 with NO loss ramp
(max clean loss 0.0224, then 0.0123 -> nan in one step), while the parent run logged zero
NaN across 500,250 loss lines with the same code, precision and grad_clip. A loss curve
cannot distinguish "inf appeared in the forward pass" from "inf gradient got applied", so
this records, every optimiser step: the pre-clip grad norm, and the first parameter whose
grad or value is non-finite. Enabled only when HDAE_NAN_TRACE is set.
"""
import os
import torch
from pytorch_lightning.callbacks import Callback


class NanTracer(Callback):
    def __init__(self, path, stop_after=None, abort=True):
        self.path = path
        self.stop_after = stop_after
        self.abort = abort
        self.fired = False
        open(self.path, "w").close()

    def _log(self, msg):
        with open(self.path, "a") as fh:
            fh.write(msg + "\n")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, *a):
        step = trainer.global_step
        tot, bad_g, bad_p = 0.0, None, None
        for n, p in pl_module.model.named_parameters():
            if p.grad is not None:
                g = p.grad.detach()
                if bad_g is None and not torch.isfinite(g).all():
                    bad_g = n
                tot += g.float().pow(2).sum().item()
            if bad_p is None and not torch.isfinite(p.detach()).all():
                bad_p = n
        gn = tot ** 0.5
        if step % 10 == 0 or bad_g or bad_p:
            self._log(f"step={step} grad_norm={gn:.6g} bad_grad={bad_g} bad_param={bad_p}")
        if (bad_g or bad_p) and not self.fired:
            self.fired = True
            self._log(f"*** FIRST NON-FINITE at optimiser step {step}: "
                      f"grad={bad_g} param={bad_p} grad_norm={gn:.6g}")
            if self.abort:
                # A non-finite PARAMETER is unrecoverable -- every later checkpoint is garbage.
                # A non-finite GRAD alone may still be skipped by the AMP scaler, so only a
                # poisoned parameter aborts.
                if bad_p:
                    self._log("*** ABORTING: parameters are non-finite, run is unrecoverable")
                    trainer.should_stop = True
                    raise RuntimeError(
                        f"NanTracer abort: parameter {bad_p} non-finite at step {step}")
        if self.stop_after and step >= self.stop_after:
            self._log(f"reached stop_after={self.stop_after}")
            trainer.should_stop = True


def maybe_tracer():
    p = os.environ.get("HDAE_NAN_TRACE")
    if not p:
        return None
    n = os.environ.get("HDAE_NAN_TRACE_STEPS")
    ab = os.environ.get("HDAE_NAN_TRACE_ABORT", "1") != "0"
    return NanTracer(p, int(n) if n else None, abort=ab)
