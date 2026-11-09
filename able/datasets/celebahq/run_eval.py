"""Generate CelebA graph-A counterfactuals and read them back at 64px.

Per source, batched:
    1. DDIM-invert the 256px source to x_T  -- the only place full resolution is used
    2. reconstruct at g=1                   -- gives e_floor, the reader's error on an
                                               unedited output
    3. render each counterfactual at THAT ATTRIBUTE's guidance
    4. downsample to 64 and read with the 64px reader bank

Everything scored is scored at 64, so the numbers sit beside the other 64px baselines. The
256px output is retained but never scored.

SOURCE-CENTRIC. Inversion and reconstruction depend only on the factual conditioning, so a
source appearing in several attribute cohorts (Male and Young use all 3,000, Beard and Bald
1,121 of them) has them computed once. Attribute-centric looping redid that work 2.7x over.

BLOCK-RESUMABLE. Sources are processed in blocks and each block is written on completion, so a
crash or an OOM costs one block, not the run. Re-invoking skips finished blocks.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from able.datasets.celebahq.reader_bank import ReaderBank            # noqa: E402
from experiments.hdae.hdae.config_io import load_hdae_config          # noqa: E402
from experiments.hdae.hdae.lit_module import HDAELitModule            # noqa: E402
from experiments.hdae.hdae.attr_utils import to_cond_values           # noqa: E402
from experiments.hdae.data.celeba_hq import CelebAHQPacked            # noqa: E402

ORDER = ["Male", "Young", "Beard", "Bald"]
AMP = dict(device_type="cuda", dtype=torch.float16)


def load_model(config, ckpt, dev):
    cfg = load_hdae_config(os.path.join(REPO, config), require_data=False)
    lit = HDAELitModule(cfg.train_conf)
    blob = torch.load(ckpt, map_location="cpu")
    sd = blob.get("state_dict", blob)
    ema = {k[len("ema_model."):]: v for k, v in sd.items() if k.startswith("ema_model.")}
    if not ema:
        raise SystemExit(f"{ckpt}: no ema_model.* keys")
    lit.ema_model.load_state_dict(ema, strict=True)
    lit.ema_model = lit.ema_model.to(dev).to(memory_format=torch.channels_last).eval()
    lit.eval()
    step = int(blob.get("global_step", -1))
    specs = lit.model.hdae_conf.encoder.cond_specs
    del blob, sd, ema
    return lit, cfg, specs, step


@torch.no_grad()
def render(lit, sampler, x_T, cond, g):
    with torch.autocast(**AMP):
        if g == 1.0:
            out = sampler.sample(model=lit.ema_model, noise=x_T, model_kwargs={"cond": cond})
        else:
            from experiments.hdae.counterfactuals.hdae_adapter import AttributeCFGWrapper
            m = AttributeCFGWrapper(lit.ema_model, g).to(x_T.device).eval()
            out = sampler.sample(model=m, noise=x_T, model_kwargs={"cond": cond})
    return ((out.float() + 1) / 2).clamp(0, 1)


def small(t):
    return F.interpolate(t, size=(64, 64), mode="bicubic", align_corners=False).clamp(0, 1)


def to8(t):
    return (t.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cohort", default="outputs/cohorts/celebahq")
    ap.add_argument("--guidance", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lmdb", default="experiments/hdae/data/packed/celebahq_256.lmdb")
    ap.add_argument("--npz", default="experiments/hdae/data/packed/celebahq_256_attrs.npz")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block", type=int, default=512, help="sources per resumable block")
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attrs", nargs="*", default=ORDER)
    ap.add_argument("--save-256", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    blkdir = os.path.join(a.out, f"{a.tag}_blocks")
    os.makedirs(blkdir, exist_ok=True)
    G = json.loads(a.guidance) if a.guidance.strip().startswith("{") else json.load(open(a.guidance))

    cohorts, need = {}, {}
    for attr in a.attrs:
        rec = json.load(open(os.path.join(REPO, a.cohort, f"graphA_{attr}.json")))
        items = rec["items"][:a.limit] if a.limit else rec["items"]
        cohorts[attr] = items
        for i, it in enumerate(items):
            need.setdefault(it["row"], {})[attr] = i
    rows_all = sorted(need)
    n_cf = sum(len(v) for v in need.values())
    blocks = [rows_all[i:i + a.block] for i in range(0, len(rows_all), a.block)]
    todo = [bi for bi in range(len(blocks))
            if not os.path.exists(os.path.join(blkdir, f"b{bi:04d}.npz"))]
    print(f"[{a.tag}] {len(rows_all)} sources, {n_cf} CFs ({n_cf/len(rows_all):.2f}/source), "
          f"{len(blocks)} blocks, {len(todo)} to do  g={G}", flush=True)
    if not todo:
        print(f"[{a.tag}] all blocks present"); return

    lit, cfg, specs, step = load_model(a.config, a.ckpt, dev)
    sampler = cfg.train_conf._make_diffusion_conf(a.T).make_sampler()
    rb = ReaderBank(device=dev)
    ds = CelebAHQPacked(os.path.join(REPO, a.lmdb), os.path.join(REPO, a.npz))
    print(f"[{a.tag}] step {step}  T={a.T}  batch {a.batch}  fp16  channels_last", flush=True)

    t0, done_src = time.time(), 0
    for bi in todo:
        rows_b = blocks[bi]
        acc = {"row": [], "src64": [], "rec64": [], "rec256": []}
        cfacc = {attr: {"ci": [], "cf64": [], "cf256": []} for attr in a.attrs}
        rdacc = {"src": {att: [] for att in rb.attrs}, "recon": {att: [] for att in rb.attrs}}
        cfrd = {attr: {att: [] for att in rb.attrs} for attr in a.attrs}
        for s0 in range(0, len(rows_b), a.batch):
            rows = rows_b[s0:s0 + a.batch]
            x = torch.stack([ds[int(r)]["img"] for r in rows]).to(dev).to(memory_format=torch.channels_last)
            fact = [cohorts[next(iter(need[r]))][need[r][next(iter(need[r]))]]["factual_model"] for r in rows]
            yf = to_cond_values(torch.tensor(fact, dtype=torch.float32, device=dev), specs)
            with torch.no_grad():
                with torch.autocast(**AMP):
                    zs = [z.clone() for z in lit.ema_model.encode(x)]
                    cond_f = lit.ema_model.make_cond(zs, yf)
                    x_T = sampler.ddim_reverse_sample_loop(
                        lit.ema_model, x, model_kwargs={"cond": cond_f})["sample"].float()
                r_img = render(lit, sampler, x_T, cond_f, 1.0)
            s64, r64 = small(((x.float() + 1) / 2).clamp(0, 1)), small(r_img)
            acc["row"] += [int(r) for r in rows]
            acc["src64"].append(to8(s64)); acc["rec64"].append(to8(r64))
            if a.save_256: acc["rec256"].append(to8(r_img))
            for key, rd in (("src", rb.read(s64)), ("recon", rb.read(r64))):
                for att in rb.attrs: rdacc[key][att].append(rd[att])
            for attr in a.attrs:
                sel = [(k, need[r][attr]) for k, r in enumerate(rows) if attr in need[r]]
                if not sel: continue
                k_idx = [p[0] for p in sel]; c_idx = [p[1] for p in sel]
                yc = to_cond_values(torch.tensor([cohorts[attr][j]["target_model"] for j in c_idx],
                                                 dtype=torch.float32, device=dev), specs)
                with torch.no_grad():
                    cond_c = lit.ema_model.make_cond([z[k_idx] for z in zs], yc)
                    c_img = render(lit, sampler, x_T[k_idx], cond_c, float(G[attr]))
                c64 = small(c_img)
                cfacc[attr]["ci"] += c_idx
                cfacc[attr]["cf64"].append(to8(c64))
                if a.save_256: cfacc[attr]["cf256"].append(to8(c_img))
                rd = rb.read(c64)
                for att in rb.attrs: cfrd[attr][att].append(rd[att])
            done_src += len(rows)
            if (s0 // a.batch) % 8 == 0:
                el = time.time() - t0
                rem = (len(rows_all) - (len(blocks) - len(todo)) * a.block - done_src)
                print(f"  block {bi} {s0+len(rows):4d}/{len(rows_b)}  {el:.0f}s  "
                      f"eta {el/max(done_src,1)*max(rem,0)/3600:.2f} h", flush=True)
        out = {"row": np.array(acc["row"]),
               "src64": np.concatenate(acc["src64"]), "rec64": np.concatenate(acc["rec64"])}
        if a.save_256: out["rec256"] = np.concatenate(acc["rec256"])
        for k in ("src", "recon"):
            for att in rb.attrs: out[f"read_{k}_{att}"] = np.concatenate(rdacc[k][att])
        for attr in a.attrs:
            if not cfacc[attr]["ci"]: continue
            out[f"cfidx_{attr}"] = np.array(cfacc[attr]["ci"])
            out[f"cf64_{attr}"] = np.concatenate(cfacc[attr]["cf64"])
            if a.save_256: out[f"cf256_{attr}"] = np.concatenate(cfacc[attr]["cf256"])
            for att in rb.attrs:
                out[f"read_cf_{attr}_{att}"] = np.concatenate(cfrd[attr][att])
        np.savez_compressed(os.path.join(blkdir, f"b{bi:04d}.npz"), **out)
        print(f"  block {bi} saved ({len(rows_b)} sources)", flush=True)
    print(f"[{a.tag}] DONE", flush=True)


if __name__ == "__main__":
    main()
