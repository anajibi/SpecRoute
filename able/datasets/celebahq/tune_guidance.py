"""Per-attribute guidance tuning on a small source subset.

k11 already has a per-attribute optimum measured on the final checkpoint (Male 5, Young 3,
Beard 2, Bald 6). k1's only record is from an earlier checkpoint with the grid capped at g=3
while CC was still rising, so its "best" is the grid edge rather than an optimum; k5 has none.
This measures both properly on their own final weights.

Cheap by construction: a subset of sources, T=50 rather than 100, no images retained -- only
the reader's verdicts, which is all CF1 needs. T=50 matches the historical sweeps, and guidance
optima are not sensitive to step count the way absolute scores are.

Selection is on CF1, per attribute, exactly as the protocol requires: the whole curve is kept
so the choice is auditable rather than asserted.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from able.datasets.celebahq.reader_bank import ReaderBank               # noqa: E402
from able.datasets.celebahq.run_eval import load_model, render, small   # noqa: E402
from experiments.hdae.hdae.attr_utils import to_cond_values             # noqa: E402
from experiments.hdae.hdae.config_io import load_hdae_config            # noqa: E402
from experiments.hdae.data.celeba_hq import CelebAHQPacked              # noqa: E402

ORDER = ["Male", "Young", "Beard", "Bald"]


def hm(a, b):
    a = min(max(a, 0.0), 1.0); b = min(max(b, 0.0), 1.0)
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cohort", default="outputs/cohorts/celebahq")
    ap.add_argument("--grid", default="2,3,4,5,6")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    grid = [float(x) for x in a.grid.split(",")]
    lit, cfg, specs, step = load_model(a.config, a.ckpt, dev)
    sampler = cfg.train_conf._make_diffusion_conf(a.T).make_sampler()
    rb = ReaderBank(device=dev)
    ds = CelebAHQPacked(os.path.join(REPO, "experiments/hdae/data/packed/celebahq_256.lmdb"),
                        os.path.join(REPO, "experiments/hdae/data/packed/celebahq_256_attrs.npz"))
    UNOBS = [x for x in rb.attrs if x not in ORDER and rb.admitted[x]]
    print(f"[{a.tag}] step {step} T={a.T} grid={grid} n={a.n}  fc_unobs pool={len(UNOBS)}", flush=True)

    out = {}
    rng = np.random.default_rng(a.seed)
    for attr in ORDER:
        items = json.load(open(os.path.join(REPO, a.cohort, f"graphA_{attr}.json")))["items"]
        sel = rng.choice(len(items), min(a.n, len(items)), replace=False)
        items = [items[i] for i in sorted(sel)]
        n = len(items)
        t0 = time.time()
        rd = {"recon": {x: np.zeros(n, np.int8) for x in rb.attrs},
              **{g: {x: np.zeros(n, np.int8) for x in rb.attrs} for g in grid}}
        for s in range(0, n, a.batch):
            bb = items[s:s + a.batch]; sl = slice(s, s + len(bb))
            x = torch.stack([ds[int(it["row"])]["img"] for it in bb]).to(dev).to(memory_format=torch.channels_last)
            yf = to_cond_values(torch.tensor([it["factual_model"] for it in bb], dtype=torch.float32, device=dev), specs)
            yc = to_cond_values(torch.tensor([it["target_model"] for it in bb], dtype=torch.float32, device=dev), specs)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16):
                    zs = [z.clone() for z in lit.ema_model.encode(x)]
                    cf_cond = lit.ema_model.make_cond(zs, yc)
                    base = lit.ema_model.make_cond(zs, yf)
                    x_T = sampler.ddim_reverse_sample_loop(lit.ema_model, x, model_kwargs={"cond": base})["sample"].float()
                r = rb.read(small(render(lit, sampler, x_T, base, 1.0)))
                for k in rb.attrs: rd["recon"][k][sl] = r[k]
                for g in grid:
                    c = rb.read(small(render(lit, sampler, x_T, cf_cond, g)))
                    for k in rb.attrs: rd[g][k][sl] = c[k]
            if (s // a.batch) % 5 == 0:
                el = time.time() - t0
                print(f"  {attr} {s+len(bb)}/{n} {el:.0f}s eta {el/max(s+len(bb),1)*(n-s-len(bb)):.0f}s", flush=True)
        fact = np.array([it["factual01"] for it in items])
        tgt = np.array([it["target01"] for it in items])
        ai = ORDER.index(attr)
        curve = {}
        for g in grid:
            hit = float((rd[g][attr] == tgt[:, ai]).mean())
            free = float(((rd["recon"][attr] == tgt[:, ai])).mean())
            ceil = float((rd["recon"][attr] == fact[:, ai]).mean())
            cc = (hit - free) / (ceil - free) if ceil - free > 1e-9 else None
            keep_obs = [o for o in ORDER if o != attr and (tgt[:, ORDER.index(o)] == fact[:, ORDER.index(o)]).all()]
            # preservation is measured against the RECONSTRUCTION's reading, not the
            # factual label: inversion and decoding already cost something before any edit
            def pres(names):
                if not names: return None
                return float(np.mean([float((rd[g][o] == rd["recon"][o]).mean()) for o in names]))
            fo, fu = pres(keep_obs), pres(UNOBS)
            curve[str(g)] = {"CC": cc, "hit": hit, "e_floor_acc": ceil, "inaction": free,
                             "fc_obs": fo, "fc_unobs": fu,
                             "cf1_unobs": hm(cc or 0.0, fu or 0.0),
                             "cf1_obs": hm(cc or 0.0, fo or 0.0) if fo is not None else None}
        best = max(grid, key=lambda g: curve[str(g)]["cf1_unobs"])
        out[attr] = {"best_g": best, "curve": curve, "n": n}
        print(f"  {attr}: best g={best}  CF1={curve[str(best)]['cf1_unobs']:.4f}  "
              f"CC={curve[str(best)]['CC']}  ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"tag": a.tag, "step": step, "T": a.T, "grid": grid, "n": a.n,
               "selected": {k: v["best_g"] for k, v in out.items()}, "per_attribute": out},
              open(a.out, "w"), indent=1)
    print(f"[{a.tag}] selected {json.dumps({k: v['best_g'] for k, v in out.items()})}", flush=True)


if __name__ == "__main__":
    main()
