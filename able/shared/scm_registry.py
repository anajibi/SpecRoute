"""Load any trained SCM in this project, and prove it can still abduct.

WHY THIS EXISTS. The evaluation protocol requires that, for every model under test, the SCM it
was trained against is available and can perform abduction -- otherwise `do(X)` cannot be
propagated to descendants and the counterfactual is not the counterfactual. Two of the saved
SCMs could no longer be loaded by the current code:

  * `causal3dident_scm.pt` was trained when `ContextEncoder`'s hidden width defaulted to 32.
    The default is now 64 and the width is not recorded in the checkpoint, so
    `load_state_dict` fails with a shape mismatch. The weights were never wrong; the loader
    drifted away from them.
  * the two C3DI checkpoints do not record their mechanism (`gaussian` vs `spline`) either.

Both are inferred here from the state_dict rather than guessed or hard-coded, so a checkpoint
saved under any past default still loads. `verify` then checks the property that actually
matters: a NULL intervention must reproduce the factual values. A loader that returns a model
with silently wrong weights would otherwise look like success.
"""
from __future__ import annotations
import os, sys
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _hidden_from_state(sd) -> int | None:
    """Width of the first encoder layer, i.e. the `hidden` the checkpoint was trained with."""
    for k, v in sd.items():
        if k.startswith("encoders.") and k.endswith("net.0.weight"):
            return int(v.shape[0])
    return None


def load_celebahq(path):
    """graph-A style SCM (experiments/hdae/causal/scm.py). Records its own attributes/edges."""
    from experiments.hdae.causal.scm import SCM
    return SCM.load(path)


def load_c3di(path):
    """C3DI SCM (experiments/hdae/causal/train_scm_causal3dident.py).

    `hidden` and `mechanism` are not stored, so both are recovered: hidden from the first
    encoder weight, mechanism by trying each and keeping the one whose shapes match.
    """
    import functools
    import experiments.hdae.causal.train_scm_causal3dident as T
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, sd = ck["config"], ck["state_dict"]
    hidden = _hidden_from_state(sd)
    graph = T.CausalGraph(cfg["attributes"], [tuple(e) for e in cfg["edges"]])
    orig = T.ContextEncoder.__init__
    try:
        if hidden is not None:
            T.ContextEncoder.__init__ = functools.partialmethod(orig, hidden=hidden)
        last = None
        for mech in ("gaussian", "spline"):
            try:
                scm = T.SCM(graph, cfg["nodes"], mechanism=mech)
                scm.load_state_dict(sd)
                scm.eval()
                scm._loaded_with = {"hidden": hidden, "mechanism": mech}
                return scm
            except RuntimeError as e:
                last = e
        raise RuntimeError(f"{path}: no mechanism matched (hidden={hidden}): {last}")
    finally:
        T.ContextEncoder.__init__ = orig


def verify(scm, attrs_or_tensor, attr_index=None, atol=1e-3) -> bool:
    """A null intervention must reproduce the factual values. Returns True/False."""
    if attr_index is None:                                   # C3DI style: dict of tensors
        back = scm.counterfactual(attrs_or_tensor, {})
        return all(torch.allclose(back[k], v, atol=atol)
                   for k, v in attrs_or_tensor.items()
                   if scm.kinds.get(k) != "categorical")
    back = scm.counterfactual(attrs_or_tensor, attr_index, {})
    return all(torch.allclose(torch.as_tensor(back[n]).reshape(-1),
                              attrs_or_tensor[:, i], atol=atol)
               for i, n in enumerate(attr_index)
               if scm.specs[n].kind != "categorical")


if __name__ == "__main__":
    import json
    S = "experiments/hdae/outputs/scm"
    rows = []
    for name, path, kind in [
        ("celebahq graph A", f"{S}/celebahq_scm.pt", "celebahq"),
        ("celebahq graph B", f"{S}/celebahq_b_scm.pt", "celebahq"),
        ("c3di gaussian",    f"{S}/causal3dident_scm.pt", "c3di"),
        ("c3di spline",      f"{S}/causal3dident_scm_spline.pt", "c3di"),
    ]:
        p = os.path.join(REPO, path)
        if not os.path.exists(p):
            rows.append((name, "MISSING", "-", "-")); continue
        try:
            if kind == "celebahq":
                scm = load_celebahq(p)
                at = scm.graph.attributes
                x = torch.zeros(8, len(at))
                ok = verify(scm, x, {n: i for i, n in enumerate(at)})
                extra = f"attrs={at}"
            else:
                scm = load_c3di(p)
                dims = scm.dims
                a = {n: torch.zeros(8, dims[n]) for n in scm.graph.attributes}
                ok = verify(scm, a)
                extra = f"{scm._loaded_with}"
            rows.append((name, "loaded", "consistent" if ok else "INCONSISTENT", extra))
        except Exception as e:
            rows.append((name, f"FAILED {type(e).__name__}", "-", str(e)[:60]))
    w = max(len(r[0]) for r in rows)
    for n, st, cons, extra in rows:
        print(f"  {n:<{w}}  {st:<22s} {cons:<13s} {extra}")
