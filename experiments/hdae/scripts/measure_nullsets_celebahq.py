"""do(Bald) flipped in BOTH directions, 2048 males, conditional x_T, 9 guidance configs.

  g in {3,4,5}  x  null set in { {Bald}, {Bald,Beard}, {all 4} }

METRICS (binary attributes, so no baseline normalisation is needed -- a predictor reading is
either the requested class or it is not):
  CC        fraction where the Bald reader returns the REQUESTED (flipped) value.
  FC_obs    mean over {Male, Young, Beard} of the fraction whose reading is UNCHANGED from the
            same subject's reconstruction. Measured against the reconstruction, not the source
            photo, so inversion error is not charged to the edit.
  FC_unobs  same, over ALL 36 attributes the model was never conditioned on (41 in the pack,
            minus the 4 observed, minus No_Beard which is just -Beard). Reported four ways,
            because one mean would hide who is carrying it:
              all36      every head
              reliable   the 33 with test balanced accuracy >=0.75 and >=60 positives.
                         Big_Lips (0.686) and Pointy_Nose (0.695) are NEAR-BALANCED and still
                         weak -- subjective CelebA labels, not an imbalance failure, so no
                         amount of retraining fixes them. Blurry has 5 positives in 3000 and
                         cannot be estimated at all. A reader that disagrees with itself 30% of
                         the time manufactures drift that looks real.
              hair       the 13 plausibly CAUSED by Bald (Receding_Hairline, Gray_Hair, Bangs,
                         Goatee, ...). Movement here may be a legitimate downstream effect.
              nonhair    the rest -- face geometry, expression, accessories. Bald should not
                         touch these, so this is the cleanest read of true collateral damage.
  CF1_obs / CF1_unobs   harmonic mean of CC with the corresponding FC.

x_T is computed ONCE with conditional inversion and cached to disk; all nine configs reuse it,
so a difference between configs is the null set or g and nothing else.
"""
import sys, os, glob, json, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/home/exouser/SpecRoute/diffae_upstream"); sys.path.insert(0, "/home/exouser/SpecRoute")
from torchvision.models import convnext_tiny
from experiments.hdae.hdae.config_io import load_hdae_config
from experiments.hdae.hdae.lit_module import HDAELitModule
from experiments.hdae.data.celeba_hq import CelebAHQPacked
from experiments.hdae.hdae.attr_utils import to_cond_values

SP = "/tmp/claude-1001/-home-exouser/ee943fc6-c5ef-4405-b557-1b557434dfe9/scratchpad"
OBS = ["Male", "Young", "Beard", "Bald"]; TARGET = 3
N, T, BS = 2048, 50, 64
GS = [3.0, 4.0, 5.0]
SETS = {"null1_target": [3], "null2_target_beard": [2, 3], "null4_all": [0, 1, 2, 3]}
MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda()
STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda()
XT_CACHE = f"{SP}/xt_cache_2048.pt"
OUT = f"{SP}/bigmeasure.json"


THR = json.load(open("experiments/hdae/outputs/attr_predictors_celebahq/thresholds.json"))
# Readers are trained with pos_weight, which shifts the logits; reading them at a fixed >0
# ignores that shift. Each threshold below was picked on partition 1 to maximise balanced
# accuracy and is applied unchanged here. Mean balanced accuracy 0.8816 -> 0.8920, with the
# gain concentrated on rare attributes (Chubby +0.032, Double_Chin +0.032, Pale_Skin +0.030)
# and the target Bald 0.966 -> 0.970.


class Reader:
    """Single-attribute predictor (the four observed ones)."""
    def __init__(self, a):
        b = torch.load(f"experiments/hdae/outputs/attr_predictors_celebahq/{a}.pt", map_location="cpu")
        m = convnext_tiny(); m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
        m.load_state_dict(b["state_dict"]); self.net = m.cuda().eval(); self.size = b["img_size"]
        # The target reader uses an F1-optimal threshold, not a balanced-accuracy one. CC
        # counts POSITIVE readings, so a permissive reader inflates it: at the balanced-tuned
        # threshold Bald ran prec 0.420 / rec 0.969, i.e. wrong 58% of the times it said
        # "bald". F1-optimal gives prec 0.823 / rec 0.785.
        # CONSEQUENCE: CC on the haired->bald direction is capped near the reader's recall of
        # 0.785. A perfect edit cannot score 1.0. Read CC against that ceiling.
        self.thr = THR[a].get("thr_f1", THR[a]["thr"])
    @torch.no_grad()
    def logit(self, img):
        x = (img.cuda() + 1) / 2
        if x.shape[-1] != self.size:
            x = torch.nn.functional.interpolate(x, size=(self.size,)*2, mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.float16):
            return self.net((x - MEAN) / STD).squeeze(1).float().cpu()
    def __call__(self, img):
        return (self.logit(img) > self.thr).float()


class MultiReader:
    """Shared-backbone reader for the unobserved attributes."""
    def __init__(self, p):
        b = torch.load(p, map_location="cpu")
        m = convnext_tiny(); m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(b["attrs"]))
        m.load_state_dict(b["state_dict"]); self.net = m.cuda().eval()
        self.size = b["img_size"]; self.attrs = b["attrs"]
        self.thr = torch.tensor([THR[a]["thr"] for a in self.attrs]).view(1, -1)
    @torch.no_grad()
    def logit(self, img):
        x = (img.cuda() + 1) / 2
        if x.shape[-1] != self.size:
            x = torch.nn.functional.interpolate(x, size=(self.size,)*2, mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.float16):
            return self.net((x - MEAN) / STD).float().cpu()
    def __call__(self, img):
        return (self.logit(img) > self.thr).float()


class MultiNullCFG(torch.nn.Module):
    def __init__(self, base, g, idx):
        super().__init__(); self.base = base; self.g = float(g); self.idx = list(idx)
    def forward(self, x, t, cond, **kw):
        co = self.base.forward(x=x, t=t, cond=cond, **kw)
        nm = torch.zeros_like(cond["y_idx"], dtype=torch.bool); nm[:, self.idx] = True
        uo = self.base.forward(x=x, t=t, **kw, cond={
            "zs": cond["zs"], "y_idx": cond["y_idx"], "null_mask": nm})
        return co.__class__(pred=uo.pred + self.g * (co.pred - uo.pred), cond=cond)


cfg = load_hdae_config("experiments/hdae/configs/celebahq256_k11_cd015r.yaml")
ck = sorted(glob.glob("experiments/hdae/outputs/celebahq256_k11_cd015r/checkpoints/*.ckpt"))
net = HDAELitModule.load_from_checkpoint(ck[-1], conf=cfg.train_conf, map_location="cpu").ema_model.cuda().eval()
specs = cfg.hdae_conf.encoder.cond_specs
samp = cfg.train_conf._make_diffusion_conf(T).make_sampler()

D = "experiments/hdae/data/packed"
ds = CelebAHQPacked(f"{D}/celebahq_256.lmdb", f"{D}/celebahq_256_attrs.npz")
names = list(ds.attribute_names); cols = [names.index(a) for a in OBS]
# The model trained on partition 0 only; partitions 1 and 2 are both held out from it.
pool = np.where((ds.partitions > 0) & (ds.attrs[:, cols[0]] == 1))[0]
assert len(pool) >= N, f"only {len(pool)} held-out males"
idx = np.sort(np.random.default_rng(7).choice(pool, N, replace=False))
bald0 = (ds.attrs[idx, cols[3]] == 1)
print(f"cohort {len(idx)} males from val+test ({len(pool)} available); "
      f"{int(bald0.sum())} already bald -> haired, {int((~bald0).sum())} haired -> bald", flush=True)

R = {a: Reader(a) for a in OBS}
MR = MultiReader("experiments/hdae/outputs/attr_predictors_celebahq/_unobs.pt")
_ev = json.load(open("experiments/hdae/outputs/attr_predictors_celebahq/test_eval.json"))
# Omitted from FC_unobs, 6 of 36. Two failure modes, opposite directions:
#   saturated  -- Blurry reads positive on nearly everything (prec 0.024, rec 1.000), so it
#                 never changes between reconstruction and counterfactual and would report
#                 FC ~ 1.0, HIDING drift.
#   noisy      -- Big_Lips / Pointy_Nose are subjective CelebA labels stuck near BAL 0.69 that
#                 no retraining fixes; Chubby / Double_Chin / Narrow_Eyes read at precision
#                 ~0.32, flipping on noise and MANUFACTURING drift.
# All hair attributes are retained, so the hair vs non-hair split stays intact.
OMIT = {"Big_Lips", "Pointy_Nose", "Blurry", "Chubby", "Double_Chin", "Narrow_Eyes"}
RELIABLE = [a for a in MR.attrs if a not in OMIT]
HAIRSET = [a for a in MR.attrs if _ev[a]["group"] == "unobs-hair"]
NONHAIR = [a for a in MR.attrs if a in RELIABLE and a not in HAIRSET]
print(f"unobs heads {len(MR.attrs)}; reliable {len(RELIABLE)} "
      f"(dropped {[a for a in MR.attrs if a not in RELIABLE]}); "
      f"hair {len(HAIRSET)}; reliable-nonhair {len(NONHAIR)}", flush=True)

y_all = to_cond_values(torch.from_numpy(ds.attrs[idx][:, cols].astype("float32")), specs)
y2_all = y_all.clone(); y2_all[:, TARGET] = -y2_all[:, TARGET]
requested_bald = (y2_all[:, TARGET] > 0).float()      # what CC must match


def batches():
    for lo in range(0, N, BS):
        hi = min(lo + BS, N)
        x = torch.stack([ds[int(i)]["img"] for i in idx[lo:hi]])
        yield lo, hi, x


# ---- pass 1: conditional inversion, cached -------------------------------------------------
if os.path.exists(XT_CACHE):
    blob = torch.load(XT_CACHE); XT = blob["xt"]; REC = blob["rec"]
    print(f"loaded cached x_T and reconstruction from {XT_CACHE}", flush=True)
else:
    XT = torch.empty(N, 3, 256, 256, dtype=torch.float16)
    REC = torch.empty(N, 3, 256, 256, dtype=torch.float16)
    t0 = time.time()
    for lo, hi, x in batches():
        xb = x.cuda(); yb = y_all[lo:hi].cuda()
        with torch.no_grad():
            zs = net.encode(xb); c = net.make_cond(zs, yb)
            xt = samp.ddim_reverse_sample_loop(net, xb, model_kwargs={"cond": c})["sample"]
            rec = samp.sample(model=net, noise=xt, model_kwargs={"cond": c})
        XT[lo:hi] = xt.half().cpu(); REC[lo:hi] = rec.half().cpu()
        if lo % (BS*8) == 0:
            el = time.time()-t0
            print(f"  invert+recon {hi}/{N}  {el/60:.1f} min elapsed, "
                  f"{(el/max(hi,1))*(N-hi)/60:.1f} min left", flush=True)
    torch.save({"xt": XT, "rec": REC, "idx": idx}, XT_CACHE)
    print(f"cached -> {XT_CACHE}", flush=True)

# reference readings, taken on the reconstruction
_lb_obs = {a: torch.cat([R[a].logit(REC[lo:hi].float()) for lo, hi, _ in batches()]) for a in OBS}
_lb_un = torch.cat([MR.logit(REC[lo:hi].float()) for lo, hi, _ in batches()])
torch.save({"obs": _lb_obs, "unobs": _lb_un, "attrs": MR.attrs}, f"{SP}/logits_recon.pt")
base_obs = {a: (_lb_obs[a] > R[a].thr).float() for a in OBS}
base_un = (_lb_un > MR.thr).float()
print("reconstruction readings: " + ", ".join(f"{a} {base_obs[a].mean():.3f}" for a in OBS), flush=True)

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
hm = lambda a, b: 0.0 if (a + b) == 0 else 2*a*b/(a + b)
print(f"\n{'config':22} {'g':>3} {'CC':>7} {'FC_obs':>7} {'FCu_all':>8} {'FCu_rel':>8} "
      f"{'FCu_hair':>9} {'FCu_non':>8} {'CF1_obs':>8}", flush=True)
for sname, sidx in SETS.items():
    for g in GS:
        key = f"{sname}|g{g:g}"
        if key in res:
            v = res[key]
            print(f"{sname:22} {g:>3.0f} {v['CC']:7.4f} {v['FC_obs']:7.4f} {v['FC_unobs']:8.4f} "
                  f"{v.get('FC_unobs_reliable',float('nan')):8.4f} {v.get('FC_unobs_hair',float('nan')):9.4f} "
                  f"{v.get('FC_unobs_nonhair',float('nan')):8.4f} {v['CF1_obs']:8.4f}  (cached)", flush=True)
            continue
        t0 = time.time()
        rd = {a: [] for a in OBS}; ru = []
        for lo, hi, x in batches():
            xb = x.cuda(); yb2 = y2_all[lo:hi].cuda()
            with torch.no_grad():
                zs = net.encode(xb)
                c2 = net.make_cond(zs, yb2)
                m = MultiNullCFG(net, g, sidx).cuda().eval()
                o = samp.sample(model=m, noise=XT[lo:hi].float().cuda(), model_kwargs={"cond": c2}).cpu()
            for a in OBS: rd[a].append(R[a].logit(o.float()))
            ru.append(MR.logit(o.float()))
        lg_obs = {a: torch.cat(v) for a, v in rd.items()}; lg_un = torch.cat(ru)
        torch.save({"obs": lg_obs, "unobs": lg_un, "attrs": MR.attrs},
                   f"{SP}/logits_{sname}_g{g:g}.pt")
        rd = {a: (lg_obs[a] > R[a].thr).float() for a in OBS}; ru = (lg_un > MR.thr).float()
        CC = float((rd["Bald"] == requested_bald).float().mean())
        fo = {a: float((rd[a] == base_obs[a]).float().mean()) for a in OBS if a != "Bald"}
        FCo = float(np.mean(list(fo.values())))
        fu = {MR.attrs[k]: float((ru[:, k] == base_un[:, k]).float().mean()) for k in range(len(MR.attrs))}
        FCu = float(np.mean(list(fu.values())))
        FCu_rel = float(np.mean([fu[a] for a in RELIABLE])) if RELIABLE else float("nan")
        FCu_hair = float(np.mean([fu[a] for a in HAIRSET])) if HAIRSET else float("nan")
        FCu_non = float(np.mean([fu[a] for a in NONHAIR])) if NONHAIR else float("nan")
        res[key] = dict(set=sname, null_idx=sidx, g=g, CC=CC, FC_obs=FCo, FC_unobs=FCu,
                        FC_unobs_reliable=FCu_rel, FC_unobs_hair=FCu_hair,
                        FC_unobs_nonhair=FCu_non, CF1_unobs_nonhair=hm(CC, FCu_non),
                        CF1_obs=hm(CC, FCo), CF1_unobs=hm(CC, FCu),
                        fc_obs_per_attr=fo, fc_unobs_per_attr=fu,
                        cc_haired_to_bald=float((rd["Bald"] == requested_bald).float()[~bald0].mean()),
                        cc_bald_to_haired=float((rd["Bald"] == requested_bald).float()[bald0].mean()),
                        minutes=(time.time()-t0)/60)
        json.dump(res, open(OUT, "w"), indent=2)
        print(f"{sname:22} {g:>3.0f} {CC:7.4f} {FCo:7.4f} {FCu:8.4f} {FCu_rel:8.4f} "
              f"{FCu_hair:9.4f} {FCu_non:8.4f} {hm(CC,FCo):8.4f}  ({(time.time()-t0)/60:.0f} min)", flush=True)
print("\nBIGMEASURE DONE", flush=True)
