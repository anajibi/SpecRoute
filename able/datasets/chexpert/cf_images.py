"""Reconstructions and counterfactuals from a CheXpert k=1 checkpoint, mid-training.

FIVE ROWS, one column per source, so the figure is exactly 20 panels:

    source            the real held-out radiograph, the factual input
    recon             DDIM-encode -> decode, no guidance. Isolates autoencoding quality from
                      conditioning: anything broken here is not an intervention failure.
    do(edema) prop    the causally valid counterfactual. Abduct the exogenous noise, act, then
                      predict: the model is conditioned on every POST-intervention value.
                      edema is a child of age_bin and has no children, so nothing propagates
                      from it -- this row is the clean single-attribute edit.
    do(edema) hard    the ablation. The intervened value is forced and every other attribute is
                      PINNED at its factual value, i.e. the graph is ignored. For a sink like
                      edema this must be identical to the row above, and the figure says so --
                      it is a null control that catches a conditioning path that leaks.
    do(age_bin) prop  the interesting one. age_bin is a parent of BOTH findings, so a hard
                      intervention on it legitimately moves effusion and edema downstream
                      through the fitted mechanisms. A model that holds the findings fixed here
                      is not respecting the graph.

Unlike 3DShapes there is NO ground-truth counterfactual image: these are real radiographs of
real patients, so no "truth" row can exist. The quantitative claim therefore rests entirely on
reading the generated images back with the FROZEN v2 predictor, which is what CC and FC below
do -- and that predictor only clears the radiologist gate on four attributes, which is why only
those four are modeled.

HONEST ABOUT n. The figure uses 4 sources and the scores use --n-score (default 16). That is
enough to see whether the conditioning does anything at all. It is NOT enough for a CC/FC
number anyone should quote: the frozen protocol cohort is 512 sources and 2,048 counterfactuals,
and these are drawn from the same held-out split but are a tiny subsample of it.

CC and FC follow the protocol's normalisation, cc_from_error / fc_from_error: an error is
divided by the error of the best constant predictor, so 1.0 is a perfect edit and 0.0 is no
better than guessing the majority class. T6 ceilings are printed beside them because a CC
cannot exceed what the instrument can read off a REAL image.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
# NOT diffae_upstream too: it ships its own choices.py without ModelName.hier_autoenc,
# and putting it ahead of the repo root shadows the patched one. config_io already
# arranges whatever the templates need.
import able.datasets.chexpert.scm_exact as S                      # noqa: E402
from experiments.hdae.hdae.attr_utils import to_cond_values       # noqa: E402
from experiments.hdae.hdae.config_io import load_hdae_config      # noqa: E402
from experiments.hdae.hdae.lit_module import HDAELitModule        # noqa: E402

COND = ["sex", "age_bin", "effusion", "edema"]
PRED_BINARY = ["sex", "effusion", "cardiomegaly", "support_devices", "ap_pa",
               "edema", "lung_opacity", "consolidation", "atelectasis", "pneumothorax"]
IMNET_MEAN, IMNET_STD = 0.449, 0.226


class Pred(nn.Module):
    """The frozen v2 attribute predictor. Measurement instrument, never trained here."""
    def __init__(self):
        super().__init__()
        from torchvision.models import densenet121
        self.trunk = densenet121(weights=None).features
        self.bin = nn.Linear(1024, len(PRED_BINARY))
        self.age = nn.Linear(1024, 3)

    def forward(self, x):
        h = Fn.adaptive_avg_pool2d(Fn.relu(self.trunk(x)), 1).flatten(1)
        return self.bin(h), self.age(h)


@torch.no_grad()
def read_back(imgs01, pred, dev):
    """Score generated images with the frozen predictor. imgs01 is (n,3,256,256) in [0,1].

    The predictor was trained on 224 centre crops of grayscale radiographs repeated to three
    channels, so the generated 256 RGB is collapsed to one channel and resized to 224 -- the
    reverse of the 224 -> 256 the datamodule does. Scoring at 256, or on three unequal
    channels, would be measuring a distribution shift rather than the image.
    """
    x = imgs01.mean(1, keepdim=True)                      # back to one channel
    x = Fn.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
    x = ((x - IMNET_MEAN) / IMNET_STD).expand(-1, 3, -1, -1).to(dev)
    with torch.autocast("cuda", dtype=torch.float16):
        lb, la = pred(x)
    return torch.sigmoid(lb.float()).cpu().numpy(), torch.softmax(la.float(), 1).cpu().numpy()


def y_of(vals, specs, dev):
    y = np.stack([vals[k] for k in COND], axis=1).astype(np.float32)
    return to_cond_values(torch.from_numpy(y).to(dev), specs)


@torch.no_grad()
def render(lit, sampler, x_T, cond, guidance):
    model = lit.ema_model
    if guidance == 1.0:
        out = sampler.sample(model=model, noise=x_T, model_kwargs={"cond": cond})
    else:
        from experiments.hdae.counterfactuals.hdae_adapter import AttributeCFGWrapper
        m = AttributeCFGWrapper(model, guidance).to(x_T.device).eval()
        out = sampler.sample(model=m, noise=x_T, model_kwargs={"cond": cond})
    return ((out + 1) / 2).clamp(0, 1)


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def compose(rows_px, row_names, col_caps, footer, out, cell=224, pad=6, lw=150):
    f, fs = _font(13), _font(11)
    ncol = len(rows_px[0])
    W = lw + ncol * (cell + pad) + pad
    H = 26 + len(rows_px) * (cell + pad + 16) + 34
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    for j, c in enumerate(col_caps):
        d.text((lw + j * (cell + pad), 6), c, fill="black", font=fs)
    y = 26
    for r, (name, row) in enumerate(zip(row_names, rows_px)):
        d.text((6, y + cell // 2 - 16), name, fill="black", font=f)
        for j, a in enumerate(row):
            t = Image.fromarray(a).convert("RGB").resize((cell, cell), Image.LANCZOS)
            im.paste(t, (lw + j * (cell + pad), y))
        y += cell + pad + 16
    d.text((6, H - 26), footer, fill="black", font=fs)
    im.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/hdae/configs/chexpert256_k1.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/home/exouser/able_data/chexpert")
    ap.add_argument("--predictor", default="/home/exouser/able_data/chexpert/predictors/best.ckpt")
    ap.add_argument("--outdir", default="outputs/chexpert_cf")
    ap.add_argument("--n-fig", type=int, default=4)
    ap.add_argument("--n-score", type=int, default=16)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    dev = "cuda"

    cfg = load_hdae_config(a.config, require_data=False)
    lit = HDAELitModule(cfg.train_conf)
    blob = torch.load(a.ckpt, map_location="cpu")
    sd = blob.get("state_dict", blob)
    ema = {k[len("ema_model."):]: v for k, v in sd.items() if k.startswith("ema_model.")}
    if not ema:
        raise SystemExit("checkpoint has no ema_model.* keys")
    lit.ema_model.load_state_dict(ema, strict=True)
    # Only the EMA copy goes to the GPU. HDAELitModule holds TWO full networks -- model and
    # ema_model, ~180M params each -- and sampling uses only the EMA one. Moving both would
    # cost an extra 0.72 GB on a card where the trainer already holds 14.8 GB of 20, which is
    # exactly how the first attempt here OOMed.
    lit.ema_model = lit.ema_model.to(dev).eval()
    lit.eval()
    step = int(blob.get("global_step", -1))
    specs = lit.model.hdae_conf.encoder.cond_specs
    sampler = cfg.train_conf._make_diffusion_conf(a.T).make_sampler()
    print(f"EMA from {os.path.basename(a.ckpt)}  global_step={step}", flush=True)

    pb = torch.load(a.predictor, map_location="cpu")
    pred = Pred()
    pred.load_state_dict(pb["state_dict"], strict=True)
    pred = pred.to(dev).eval()

    # sources: held-out only, and only rows whose edema label is certain -- the same rule the
    # cohort builder applies, so these are drawn from the pool the protocol would draw from.
    import pandas as pd
    meta = pd.read_csv(f"{a.data}/chexpert_meta.csv")
    split = np.load(f"{a.data}/split.npy")
    unc = pd.read_csv(f"{a.data}/label_uncertainty.csv")
    pool = np.flatnonzero((split == 1) & (unc["uncertain_edema"].to_numpy() == 0))
    rng = np.random.default_rng(a.seed)
    rows = np.sort(rng.choice(pool, a.n_score, replace=False))
    arr = np.load(f"{a.data}/images_224.npy", mmap_mode="r")
    vals = {k: meta.loc[rows, k].to_numpy(np.int64) for k in COND}
    params = S.load(f"{a.data}/scm_params.json")

    # ---- the three counterfactual attribute sets
    cf_sets = {}
    for var, label in (("edema", "do_edema"), ("age_bin", "do_age")):
        if var == "edema":
            new = 1 - vals["edema"]
        else:
            new = (vals["age_bin"] + 1 + rng.integers(0, 2, len(rows))) % 3
        cf, _ = S.counterfactual(vals, params, var, new, seed=a.seed)
        cf_sets[label] = {k: np.asarray(cf[k]) for k in COND}
    # the hard / no-propagation ablation on edema: force the target, pin everything else
    hard = {k: vals[k].copy() for k in COND}
    hard["edema"] = 1 - vals["edema"]
    cf_sets["do_edema_hard"] = hard

    moved = {lab: [k for k in COND if (cf_sets[lab][k] != vals[k]).any()] for lab in cf_sets}
    print("attributes that move per intervention:", moved, flush=True)

    # ---- generate, in small batches so the trainer on the same card is not starved of memory
    def gen(idx):
        x = torch.from_numpy(np.asarray(arr[rows[idx]], dtype=np.float32) / 127.5 - 1.0)[:, None]
        x = Fn.interpolate(x, size=(256, 256), mode="bicubic", align_corners=False).clamp_(-1, 1)
        x = x.expand(-1, 3, -1, -1).contiguous().to(dev)
        yf = y_of({k: vals[k][idx] for k in COND}, specs, dev)
        with torch.no_grad():
            zs = [z.clone() for z in lit.ema_model.encode(x)]
            cond = lit.ema_model.make_cond(zs, yf)
            x_T = sampler.ddim_reverse_sample_loop(lit.ema_model, x,
                                                   model_kwargs={"cond": cond})["sample"]
            out = {"source": ((x + 1) / 2).clamp(0, 1).cpu(),
                   "recon": render(lit, sampler, x_T, cond, 1.0).cpu()}
            for lab, cfv in cf_sets.items():
                ycf = y_of({k: cfv[k][idx] for k in COND}, specs, dev)
                out[lab] = render(lit, sampler, x_T,
                                  lit.ema_model.make_cond(zs, ycf), a.guidance).cpu()
        return out

    keys = ["source", "recon", "do_edema", "do_edema_hard", "do_age"]
    acc = {k: [] for k in keys}
    for s in range(0, len(rows), a.batch):
        idx = np.arange(s, min(s + a.batch, len(rows)))
        o = gen(idx)
        for k in keys:
            acc[k].append(o[k])
        print(f"  generated {idx[-1]+1}/{len(rows)}", flush=True)
    G = {k: torch.cat(v) for k, v in acc.items()}

    # ---- read every panel back with the frozen predictor
    probs = {k: read_back(G[k], pred, dev) for k in keys}
    thr = {}
    pr = json.load(open(f"{a.data}/predictors/predictor_report.json"))
    thr_src = pr.get("thresholds_youden_on_select_v2", pr["thresholds_youden_on_select"])
    for k in ("sex", "effusion", "edema"):
        thr[k] = float(thr_src[k]) if not isinstance(thr_src[k], dict) else float(thr_src[k]["threshold"])
    t6 = pr.get("t6_calibration_v2", pr["t6_calibration"])
    r_k = pr.get("fc_correlation_correction_r_k_a_v2", {})

    def hard_read(lab):
        pb_, pa_ = probs[lab]
        out = {"sex": (pb_[:, PRED_BINARY.index("sex")] >= thr["sex"]).astype(int),
               "effusion": (pb_[:, PRED_BINARY.index("effusion")] >= thr["effusion"]).astype(int),
               "edema": (pb_[:, PRED_BINARY.index("edema")] >= thr["edema"]).astype(int),
               "age_bin": pa_.argmax(1)}
        return out

    from able.adapters.base import cc_from_error, fc_from_error
    report = {"checkpoint": os.path.basename(a.ckpt), "global_step": step,
              "guidance": a.guidance, "T": a.T, "n_score": int(len(rows)),
              "n_fig": int(a.n_fig), "rows": rows.tolist(),
              "caveat": ("n is a tiny subsample of the held-out split; the frozen protocol "
                         "cohort is 512 sources / 2,048 counterfactuals. These numbers show "
                         "whether conditioning works, not how well."),
              "recon": {}, "interventions": {}}

    r = hard_read("recon")
    report["recon"]["l1_vs_source"] = float((G["source"] - G["recon"]).abs().mean())
    report["recon"]["attr_agreement_with_factual"] = {
        k: float((r[k] == vals[k]).mean()) for k in COND}
    print(f"\nrecon  L1 {report['recon']['l1_vs_source']:.4f}  "
          f"agreement {report['recon']['attr_agreement_with_factual']}")

    for lab in ("do_edema", "do_edema_hard", "do_age"):
        tgtv = cf_sets[lab]
        got = hard_read(lab)
        var = "edema" if "edema" in lab else "age_bin"
        # CC: the intervened attribute reaching its post-intervention target
        err = float((got[var] != tgtv[var]).mean())
        _, cnt = np.unique(tgtv[var], return_counts=True)
        triv = 1.0 - cnt.max() / len(rows)
        cc = cc_from_error(err, triv) if triv > 0 else None
        # FC: everything that should NOT move, staying put
        keep = [k for k in COND if (tgtv[k] == vals[k]).all() and k != var]
        fcs = {}
        for k in keep:
            e = float((got[k] != vals[k]).mean())
            _, c2 = np.unique(vals[k], return_counts=True)
            t2 = 1.0 - c2.max() / len(rows)
            fcs[k] = {"error": e, "trivial": t2,
                      "fc": (fc_from_error(e, t2) if t2 > 0 else None),
                      "r_k_a": r_k.get(k, {}).get(var) if isinstance(r_k.get(k), dict) else None}
        rec = {"moves": moved[lab], "target_hit_rate": 1.0 - err, "cc": cc,
               "cc_trivial_error": triv, "fc_per_attribute": fcs,
               "t6_ceiling_for_target": t6.get(var)}
        # descendants that are SUPPOSED to move, scored separately
        desc = [k for k in COND if k != var and (tgtv[k] != vals[k]).any()]
        rec["descendants_scored"] = {k: float((got[k] == tgtv[k]).mean()) for k in desc}
        report["interventions"][lab] = rec
        print(f"\n{lab}  moves {moved[lab]}")
        print(f"  target hit {rec['target_hit_rate']:.3f}   CC {cc if cc is None else round(cc,4)}")
        print("  FC " + "  ".join(
            f"{k}={('n/a' if v['fc'] is None else format(v['fc'], '.4f'))}" for k, v in fcs.items()))
        if desc:
            print(f"  descendants {rec['descendants_scored']}")

    # ---- the 20-panel figure, first n_fig sources
    def panels(t, n):
        aa = (t[:n].clamp(0, 1).mean(1).numpy() * 255).astype(np.uint8)
        return [aa[i] for i in range(aa.shape[0])]

    nf = min(a.n_fig, len(rows))
    AGE = {0: "<50", 1: "50-65", 2: ">65"}
    caps = []
    for i in range(nf):
        caps.append(f"{'M' if vals['sex'][i] else 'F'} {AGE[vals['age_bin'][i]]} "
                    f"eff={vals['effusion'][i]} ede={vals['edema'][i]}")
    names = ["source", "recon", f"do(edema)\npropagated", f"do(edema)\nHARD, no prop",
             "do(age_bin)\npropagated"]
    rows_px = [panels(G[k], nf) for k in keys]
    fig = compose(rows_px, names, caps,
                  f"k=1  step {step}  g={a.guidance:g}  T={a.T}  "
                  f"CC/FC on n={len(rows)} held-out sources  (subsample, not the 512 cohort)",
                  os.path.join(a.outdir, f"cf_step{step}.png"))
    json.dump(report, open(os.path.join(a.outdir, f"cf_report_step{step}.json"), "w"), indent=2)
    print(f"\nwrote {fig}")


if __name__ == "__main__":
    main()
