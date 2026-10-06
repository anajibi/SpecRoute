"""Best guidance per attribute, using the new per-attribute readers.

For each of the four conditioned attributes, flip it in BOTH directions over the 2048-male
cohort and sweep g. Target-only nulling throughout: the 9-config measurement showed it beats
all-four nulling on CC at every g (+0.055 at g=5, paired CI [+0.039,+0.070]).

THRESHOLDS -- two different criteria, deliberately:
  the attribute BEING INTERVENED ON is read at its F1-optimal threshold, because CC counts
    positive readings and a permissive reader inflates it. At the balanced-accuracy threshold
    the Bald reader ran precision 0.44 against a base rate of 0.06.
  every FC reader is read UNTUNED (logit>0), because tuning on photographs transfers badly to
    generated images -- it helped only 15 of 37 attributes and lowered the mean.
"""
import sys, os, glob, json, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/home/exouser/SpecRoute/diffae_upstream"); sys.path.insert(0, "/home/exouser/SpecRoute")
from torchvision import models
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.attr_utils import to_cond_values

SP = "/tmp/claude-1001/-home-exouser/ee943fc6-c5ef-4405-b557-1b557434dfe9/scratchpad"
P = "experiments/hdae/outputs/attr_predictors_celebahq"
OBS = ["Male", "Young", "Beard", "Bald"]
ONLY = "Beard"
# target-only nulling was still improving at g=5 in the 9-config run, so the grid
# extends to 6 rather than stopping at 5.
GS = [1.0, 1.5]
N, T, BS = 2048, 50, 64
OUT = "experiments/hdae/outputs/nullctrl/gsweep_attrs.json"
MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda(); STD = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda()

fe = json.load(open(f"{P}/full_eval.json"))
okk = {a: v for a, v in fe.items() if v.get("recon_tuned")}
THIN = [a for a, v in okk.items() if v["n_pos_recon"] < 70]
NOISY = [a for a, v in okk.items() if v["recon_tuned"]["bal"] < 0.75 and v["n_pos_recon"] >= 70]
KEEP = sorted([a for a in okk if a not in THIN and a not in NOISY])
print(f"FC readers kept: {len(KEEP)}", flush=True)

class R:
    def __init__(s, a, thr):
        b = torch.load(f"{P}/single/{a}.pt", map_location="cpu")
        m = getattr(models, b["arch"])(); m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
        m.load_state_dict({k: v.float() for k, v in b["state_dict"].items()})
        s.net = m.cuda().eval(); s.size = b["img_size"]; s.thr = thr
    @torch.no_grad()
    def __call__(s, img):
        x = (img.cuda().float() + 1) / 2
        if x.shape[-1] != s.size:
            x = torch.nn.functional.interpolate(x, size=(s.size,)*2, mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.float16):
            return (s.net((x - MEAN) / STD).squeeze(1).float() > s.thr).float().cpu()

class CFG(torch.nn.Module):
    def __init__(s, base, g, idx):
        super().__init__(); s.b = base; s.g = float(g); s.i = list(idx)
    def forward(s, x, t, cond, **kw):
        co = s.b.forward(x=x, t=t, cond=cond, **kw)
        nm = torch.zeros_like(cond["y_idx"], dtype=torch.bool); nm[:, s.i] = True
        uo = s.b.forward(x=x, t=t, **kw, cond={"zs": cond["zs"], "y_idx": cond["y_idx"], "null_mask": nm})
        return co.__class__(pred=uo.pred + s.g * (co.pred - uo.pred), cond=cond)

cfg = load_hdae_config("experiments/hdae/configs/celebahq256_k11_cd015r.yaml")
ck = sorted(glob.glob("experiments/hdae/outputs/celebahq256_k11_cd015r/checkpoints/*.ckpt"))
net = HDAELitModule.load_from_checkpoint(ck[-1], conf=cfg.train_conf, map_location="cpu").ema_model.cuda().eval()
specs = cfg.hdae_conf.encoder.cond_specs
samp = cfg.train_conf._make_diffusion_conf(T).make_sampler()

ds = CelebAHQPacked("experiments/hdae/data/packed/celebahq_256.lmdb", "experiments/hdae/data/packed/celebahq_256_attrs.npz")
names = list(ds.attribute_names); cols = [names.index(a) for a in OBS]
pool = np.where((ds.partitions > 0) & (ds.attrs[:, names.index("Male")] == 1))[0]
idx = np.sort(np.random.default_rng(7).choice(pool, N, replace=False))
blob = torch.load(f"{SP}/xt_cache_2048.pt"); XT, REC = blob["xt"], blob["rec"]
y_all = to_cond_values(torch.from_numpy(ds.attrs[idx][:, cols].astype("float32")), specs)

def f1_thr(a):
    """F1-optimal threshold on the real test split, for the attribute being intervened on."""
    b = torch.load(f"{P}/single/{a}.pt", map_location="cpu")
    m = getattr(models, b["arch"])(); m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
    m.load_state_dict({k: v.float() for k, v in b["state_dict"].items()}); m = m.cuda().eval()
    te = np.where(ds.partitions == 2)[0]
    L = []
    with torch.no_grad():
        for lo in range(0, len(te), 64):
            x = torch.stack([ds[int(i)]["img"] for i in te[lo:lo+64]]).cuda().float(); x = (x + 1) / 2
            if x.shape[-1] != b["img_size"]:
                x = torch.nn.functional.interpolate(x, size=(b["img_size"],)*2, mode="bilinear", align_corners=False)
            with torch.autocast("cuda", dtype=torch.float16):
                L.append(m((x - MEAN) / STD).squeeze(1).float().cpu())
    L = torch.cat(L); y = torch.from_numpy((ds.attrs[te, names.index(a)] == 1).astype("float32"))
    best, bf = 0.0, -1
    for t in torch.quantile(L, torch.linspace(0.5, 0.999, 250)).tolist():
        p = (L > t).float()
        tp = float(((p == 1) & (y == 1)).sum()); fp = float(((p == 1) & (y == 0)).sum()); fn = float(((p == 0) & (y == 1)).sum())
        pr = tp/(tp+fp) if tp+fp else 0.; rc = tp/(tp+fn) if tp+fn else 0.
        f = 2*pr*rc/(pr+rc) if pr+rc else 0.
        if f > bf: bf, best = f, t
    del m; torch.cuda.empty_cache()
    return float(best), bf

FCR = {a: R(a, 0.0) for a in KEEP}
base = {a: torch.cat([FCR[a](REC[lo:lo+BS]) for lo in range(0, N, BS)]) for a in KEEP}
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
hm = lambda x, z: 0.0 if x + z == 0 else 2*x*z/(x+z)
print(f"\n{'attr':8} {'g':>3} {'CC':>7} {'FC_obs':>7} {'FC_unobs':>9} {'CF1':>7}", flush=True)
for ai, A in enumerate(OBS):
    if A != ONLY: continue
    t, f1v = f1_thr(A)
    tgt = R(A, t)
    tgt_base = torch.cat([tgt(REC[lo:lo+BS]) for lo in range(0, N, BS)])
    y2 = y_all.clone(); y2[:, ai] = -y2[:, ai]
    want = (y2[:, ai] > 0).float()
    for g in GS:
        key = f"{A}|g{g:g}"
        if key in res: continue
        t0 = time.time(); rd = {a: [] for a in KEEP}; rt = []
        for lo in range(0, N, BS):
            hi = min(lo+BS, N)
            xb = torch.stack([ds[int(i)]["img"] for i in idx[lo:hi]]).cuda()
            with torch.no_grad():
                zs = net.encode(xb); c2 = net.make_cond(zs, y2[lo:hi].cuda())
                o = samp.sample(model=CFG(net, g, [ai]).cuda().eval(),
                                noise=XT[lo:hi].float().cuda(), model_kwargs={"cond": c2}).cpu()
            rt.append(tgt(o))
            for a in KEEP: rd[a].append(FCR[a](o))
        rt = torch.cat(rt); rd = {a: torch.cat(v) for a, v in rd.items()}
        CC = float((rt == want).float().mean())
        fo = {a: float((rd[a] == base[a]).float().mean()) for a in OBS if a != A and a in KEEP}
        fu = {a: float((rd[a] == base[a]).float().mean()) for a in KEEP if a not in OBS}
        FCo = float(np.mean(list(fo.values()))) if fo else float("nan")
        FCu = float(np.mean(list(fu.values())))
        res[key] = dict(attr=A, g=g, CC=CC, FC_obs=FCo, FC_unobs=FCu,
                        CF1=hm(CC, FCu), CF1_obs=hm(CC, FCo), thr_f1=t, thr_f1_score=f1v,
                        fc_obs_per=fo, fc_unobs_per=fu, minutes=(time.time()-t0)/60)
        json.dump(res, open(OUT, "w"), indent=2)
        print(f"{A:8} {g:>3.0f} {CC:7.4f} {FCo:7.4f} {FCu:9.4f} {hm(CC,FCu):7.4f}  ({(time.time()-t0)/60:.0f}m)", flush=True)
    del tgt; torch.cuda.empty_cache()
print("\nBEARD_LOWG DONE", flush=True)
