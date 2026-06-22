"""Latent probing utilities for HDAE semantic levels."""

from .linear_probe import ProbeJob, make_probe_jobs, train_all_probes

__all__ = ["ProbeJob", "make_probe_jobs", "train_all_probes"]


def __getattr__(name):
    if name in __all__:
        from . import linear_probe
        return getattr(linear_probe, name)
    raise AttributeError(name)
