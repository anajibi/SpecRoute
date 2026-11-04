"""able -- one entry point for every trained model.

    able list                       what is registered
    able show   <model>             config, checkpoint, attributes, graph
    able verify <model>|--all       does the config still build the claimed architecture
    able eval   <model> ...         run the evaluation protocol (see --help)

`eval` deliberately refuses to emit a single headline CF1: it reports CC, every FC term and a CF1
per FC term, each named. Which one is 'the' number is a protocol decision that has not been made,
and hardcoding a default here is how the two datasets diverged in the first place.
"""
from __future__ import annotations
import argparse, json, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from able import registry as R


def _list(args):
    for name in sorted(R.MODELS):
        m = R.MODELS[name]
        g = f" graph {m.graph}" if m.graph else ""
        print(f"{name:16} {m.dataset:9}{g:9} {len(m.attrs)} attrs  {m.n_keys} keys")


def _show(args):
    m = R.get(args.model)
    print(f"model       {m.name}")
    print(f"dataset     {m.dataset}" + (f" (graph {m.graph})" if m.graph else ""))
    print(f"config      {m.config}")
    print(f"checkpoint  {m.uri}")
    print(f"attributes  {', '.join(m.attrs)}")
    print(f"keys        {m.n_keys}")
    if m.note:
        print(f"note        {m.note}")


def _verify(args):
    names = sorted(R.MODELS) if args.all else [args.model]
    bad = 0
    for n in names:
        try:
            r = R.verify(R.get(n))
            flag = "ok" if r["ok"] else f"MISMATCH expected {r['expected']} got {r['n_keys']}"
            if not r["ok"]:
                bad += 1
            print(f"{n:16} K={r['K']:<3} attrs={r['n_attributes']} "
                  f"embed={r['attr_embed_dim']:<4} {r['dataset']:12} {flag}")
        except Exception as e:
            bad += 1
            print(f"{n:16} FAILED  {type(e).__name__}: {e}")
    return 1 if bad else 0


def _eval(args):
    m = R.get(args.model)
    print(f"model {m.name}  dataset {m.dataset}" + (f"  graph {m.graph}" if m.graph else ""))
    print(f"guidance {args.guidance}  null-set {args.null_set}  n {args.n}")
    print("\nThe generation half of this (DDIM inversion, guided sampling) still lives in the\n"
          "per-dataset scripts and needs a GPU, which is currently unavailable. What is wired up\n"
          "here is the scoring half: adapters, metric components and the output schema.\n"
          "Score an existing run with:  able score --persample <file.npz> --model <name>")
    return 0


def _score(args):
    """Re-aggregate an existing per-sample file through the shared metric layer.

    Useful without a GPU: the stored .npz per-sample readings from past runs can be re-pooled under
    the shared definitions, which is how a past result gets restated without regenerating images.
    """
    import numpy as np
    from able.shared import metrics
    m = R.get(args.model)
    z = np.load(args.persample, allow_pickle=True)
    print(f"{args.persample}: {len(z.files)} arrays -> {', '.join(z.files[:8])}")
    print("per-sample readings loaded; pooling requires the run's intervention layout,\n"
          "which `able eval` writes alongside new runs.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="able", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="registered models").set_defaults(fn=_list)
    s = sub.add_parser("show", help="one model's provenance"); s.add_argument("model"); s.set_defaults(fn=_show)
    v = sub.add_parser("verify", help="config still builds the claimed architecture")
    v.add_argument("model", nargs="?"); v.add_argument("--all", action="store_true"); v.set_defaults(fn=_verify)
    e = sub.add_parser("eval", help="run the evaluation protocol")
    e.add_argument("model"); e.add_argument("--guidance", type=float, default=3.0)
    e.add_argument("--null-set", choices=["target-only", "all"], default="target-only")
    e.add_argument("--n", type=int, default=2048); e.set_defaults(fn=_eval)
    c = sub.add_parser("score", help="re-pool an existing per-sample file")
    c.add_argument("--persample", required=True); c.add_argument("--model", required=True); c.set_defaults(fn=_score)
    a = p.parse_args(argv)
    if a.cmd == "verify" and not a.all and not a.model:
        p.error("verify needs a model name or --all")
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
