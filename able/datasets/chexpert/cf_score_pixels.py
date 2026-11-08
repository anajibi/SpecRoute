"""Score saved counterfactual pixels with several readers, and report the flip taxonomy.

WHY A TAXONOMY AND NOT A HIT RATE. "4 of 16 hit the target" hides three different events:

    correct flip   the reading MOVED off the reconstruction's value and landed on the target
    freebie        the reading already equalled the target on the RECONSTRUCTION, so the
                   sample scores as a hit while the intervention did nothing. This happens
                   wherever the reader misreads the reconstruction in the direction the
                   intervention was going anyway, and it inflates every pooled number.
    wrong move     the reading moved AWAY from the target

Only the first is evidence the intervention worked. The direction matters too: flipping a
positive to negative and a negative to positive are different edits, and on a dataset that is
~74% edema-negative a diffusion prior favours one of them for free.

READERS. Each needs its own operating threshold; comparing probabilities across models would
be meaningless. Ours uses the threshold the protocol already tuned on the select split. XRV
has no published threshold here, so one is derived by Youden on radiologist_val from the
probabilities the noise audit already computed -- no new XRV inference over the frozen set.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as Fn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from able.datasets.chexpert.cf_images import Pred, PRED_BINARY     # noqa: E402

XRV_PATH = ["Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
            "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
            "Cardiomegaly", "Nodule", "Mass", "Hernia", "Lung Lesion", "Fracture",
            "Lung Opacity", "Enlarged Cardiomediastinum"]
IMNET_MEAN, IMNET_STD = 0.449, 0.226


def youden(y, p):
    o = np.argsort(-p)
    y = np.asarray(y)[o]; p = np.asarray(p)[o]
    P, N = y.sum(), len(y) - y.sum()
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    j = tp / max(P, 1) - fp / max(N, 1)
    return float(p[int(j.argmax())])


@torch.no_grad()
def score_ours(px, ckpt, dev):
    """px: (n,256,256) uint8. Returns dict of per-attribute probabilities."""
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    # The 11-head measurement checkpoint has one extra binary output. Size the head from the
    # checkpoint rather than assuming 10, and load strict so a real mismatch still raises.
    n_bin = int(sd["bin.weight"].shape[0])
    m = Pred()
    if n_bin != m.bin.out_features:
        m.bin = torch.nn.Linear(m.bin.in_features, n_bin)
    m.load_state_dict(sd, strict=True)
    m = m.to(dev).eval()
    x = torch.from_numpy(px.astype(np.float32) / 255.0)[:, None]
    x = Fn.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
    x = ((x - IMNET_MEAN) / IMNET_STD).expand(-1, 3, -1, -1).to(dev)
    with torch.autocast("cuda", dtype=torch.float16):
        lb, la = m(x)
    del m; torch.cuda.empty_cache()
    return torch.sigmoid(lb.float()).cpu().numpy(), torch.softmax(la.float(), 1).cpu().numpy()


@torch.no_grad()
def score_xrv(px, dev):
    import torchxrayvision as xrv
    m = xrv.models.DenseNet(weights="densenet121-res224-all").to(dev).eval()
    x = px.astype(np.float32)
    x = xrv.datasets.normalize(x, 255)                      # -> [-1024, 1024], XRV's own scale
    t = torch.from_numpy(x)[:, None]
    t = Fn.interpolate(t, size=(224, 224), mode="bicubic", align_corners=False).to(dev)
    out = m(t).cpu().numpy()
    del m; torch.cuda.empty_cache()
    return out


def taxonomy(fac, rec, cfv, tgt):
    fac, rec, cfv, tgt = map(np.asarray, (fac, rec, cfv, tgt))
    changed = cfv != rec
    pos, neg = fac == 1, fac == 0
    return {
        "recon_correct": int((rec == fac).sum()),
        "pos_to_neg_ok": int((pos & changed & (cfv == 0)).sum()), "n_pos": int(pos.sum()),
        "neg_to_pos_ok": int((neg & changed & (cfv == 1)).sum()), "n_neg": int(neg.sum()),
        "freebie": int(((~changed) & (cfv == tgt)).sum()),
        "wrong_move": int((changed & (cfv != tgt)).sum()),
        "hit": int((cfv == tgt).sum()),
        "real_flips": int((changed & (cfv == tgt)).sum()),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", required=True, help="directory written by cf_render_save.py")
    ap.add_argument("--data", default="/home/exouser/able_data/chexpert")
    ap.add_argument("--ours10", default="/home/exouser/able_data/chexpert/predictors/best.ckpt")
    ap.add_argument("--ours11", default="/home/exouser/SpecRoute/experiments/hdae/outputs/"
                                        "attr_predictors_chexpert_head11/best.ckpt")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    man = json.load(open(f"{a.px}/manifest.json"))
    fac = np.array(man["factual"]["edema"]); tgt = np.array(man["target"]["edema"])
    gkeys = [k for k in man["keys"] if k.startswith("g")]
    px = {k: np.load(f"{a.px}/{k}.npy") for k in man["keys"]}

    pr = json.load(open(f"{a.data}/predictors/predictor_report.json"))
    tsrc = pr.get("thresholds_youden_on_select_v2", pr["thresholds_youden_on_select"])
    thr_ours = float(tsrc["edema"]) if not isinstance(tsrc["edema"], dict) else float(tsrc["edema"]["threshold"])

    # XRV threshold: Youden on radiologist_val, from probabilities the noise audit already has
    import pandas as pd
    valid = pd.read_csv(f"{a.data}/source_csv/valid.csv").set_index("Path")
    rmeta = pd.read_csv(f"{a.data}/radiologist_val_meta.csv")
    vxp = np.load(f"{a.data}/noise_audit/valid_xrv_probs.npy")
    vpos = pd.Series(np.arange(len(valid)), index=valid.index.values)
    sel = vpos.loc[rmeta["path"].values].to_numpy()
    y_rad = pd.to_numeric(valid.loc[rmeta["path"].values, "Edema"], errors="coerce").fillna(0).to_numpy()
    thr_xrv = youden(y_rad, vxp[sel][:, XRV_PATH.index("Edema")])

    readers = {}
    for name, ck in (("ours10", a.ours10), ("ours11", a.ours11)):
        if not os.path.exists(ck):
            print(f"  (skipping {name}: {ck} missing)"); continue
        readers[name] = {k: (score_ours(px[k], ck, dev)[0][:, PRED_BINARY.index("edema")] >= thr_ours
                             ).astype(int) for k in man["keys"]}
    try:
        readers["xrv"] = {k: (1 / (1 + np.exp(-0)) * 0 + score_xrv(px[k], dev)[:, XRV_PATH.index("Edema")]
                              >= thr_xrv).astype(int) for k in man["keys"]}
    except Exception as e:                                   # XRV optional, never fatal
        print(f"  (XRV unavailable: {type(e).__name__}: {e})")

    print(f"\nstep {man['global_step']}   n={len(fac)}   "
          f"factual {int((fac==1).sum())} pos / {int((fac==0).sum())} neg")
    print(f"thresholds: ours {thr_ours:.4f}   xrv {thr_xrv:.4f} (Youden on radiologist_val)\n")
    rep = {"global_step": man["global_step"], "n": len(fac),
           "thresholds": {"ours": thr_ours, "xrv": thr_xrv}, "readers": {}}
    hdr = (f"{'reader':8s} {'g':>4s} | {'recon ok':>8s} | {'pos>neg':>9s} {'neg>pos':>9s} | "
           f"{'real':>5s} {'free':>5s} {'wrong':>6s} {'hit':>6s}")
    print(hdr); print("-" * len(hdr))
    for rname, rd in readers.items():
        rep["readers"][rname] = {}
        for g in gkeys:
            t = taxonomy(fac, rd["recon"], rd[g], tgt)
            rep["readers"][rname][g] = t
            print(f"{rname:8s} {g:>4s} | {t['recon_correct']:5d}/{len(fac):<2d} | "
                  f"{t['pos_to_neg_ok']:4d}/{t['n_pos']:<4d} {t['neg_to_pos_ok']:4d}/{t['n_neg']:<4d} | "
                  f"{t['real_flips']:5d} {t['freebie']:5d} {t['wrong_move']:6d} "
                  f"{t['hit']:3d}/{len(fac):<2d}")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")
