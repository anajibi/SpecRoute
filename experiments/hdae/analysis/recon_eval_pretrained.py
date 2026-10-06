"""Quantify the pretrained ffhq256_autoenc reconstruction on CelebA-HQ before fine-tuning.

This is the transfer baseline: FFHQ-trained weights applied to CelebA-HQ with NO adaptation.
Whatever the hierarchical fine-tune produces has to be read against this floor, not against
zero. It also fixes the DDIM step count the rest of the study will use.
"""
import sys, time, json, torch
sys.path.insert(0, "/home/exouser/SpecRoute")
sys.path.insert(0, "/home/exouser/SpecRoute/diffae_upstream")
from templates import ffhq256_autoenc
from experiments.hdae.data.celeba_hq import CelebAHQPacked
import torchvision.utils as vu

dev = "cuda"
conf = ffhq256_autoenc()
net = conf.make_model_conf().make_model().to(dev).eval()
raw = torch.load("experiments/hdae/pretrained/ffhq256_autoenc_model.pt", map_location="cpu")
net.load_state_dict({k[len("ema_model."):]: v for k, v in raw.items() if k.startswith("ema_model.")}, strict=True)

D = "experiments/hdae/data/packed"
ds = CelebAHQPacked(f"{D}/celebahq_256.lmdb", f"{D}/celebahq_256_attrs.npz")
names = ds.attribute_names
import numpy as np
test = np.where(ds.partitions == 2)[0]            # held-out split only
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(test, 64, replace=False))
x = torch.stack([ds[int(i)]["img"] for i in idx]).to(dev)
A = ds.attrs[idx]

out = {}
for T in [20, 50, 100]:
    samp = conf._make_diffusion_conf(T).make_sampler()
    t0 = time.time()
    recs = []
    with torch.no_grad():
        for lo in range(0, len(x), 8):
            xb = x[lo:lo+8]
            c = net.encode(xb)
            c = c if torch.is_tensor(c) else c["cond"]
            xT = samp.ddim_reverse_sample_loop(net, xb, model_kwargs={"cond": c})["sample"]
            recs.append(samp.sample(model=net, noise=xT, model_kwargs={"cond": c}))
    rec = torch.cat(recs)
    dt = time.time() - t0
    mae = (rec - x).abs().mean(dim=(1, 2, 3))
    mse = ((rec - x) ** 2).mean(dim=(1, 2, 3))
    psnr = (10 * torch.log10(4.0 / mse))           # data range is 2.0 -> peak^2 = 4
    out[T] = dict(mae=float(mae.mean()), mae_std=float(mae.std()),
                  psnr=float(psnr.mean()), psnr_min=float(psnr.min()), psnr_max=float(psnr.max()),
                  sec_per_image=dt / len(x))
    print(f"T={T:3d}  MAE {mae.mean():.4f}+/-{mae.std():.4f}  PSNR {psnr.mean():.2f} dB "
          f"[{psnr.min():.1f},{psnr.max():.1f}]  {dt/len(x):.2f} s/img")
    if T == 50:
        torch.save({"mae": mae.cpu(), "psnr": psnr.cpu(), "idx": idx}, f"{'/tmp/claude-1001/-home-exouser/ee943fc6-c5ef-4405-b557-1b557434dfe9/scratchpad'}/recon_T50.pt")
        # worst and best four, so the grid is honest about the range rather than cherry-picked
        order = torch.argsort(mae)
        pick = torch.cat([order[:4], order[-4:]])
        vu.save_image(torch.cat([x[pick], rec[pick]]).add(1).div(2).clamp(0, 1),
                      "/tmp/claude-1001/-home-exouser/ee943fc6-c5ef-4405-b557-1b557434dfe9/scratchpad/recon_grid.png", nrow=8)
        # per-attribute reconstruction difficulty
        per = {}
        for a in ["Male", "Young", "Beard", "Bald", "Eyeglasses", "Wearing_Hat"]:
            j = names.index(a); m = A[:, j] == 1
            if m.sum() >= 3:
                per[a] = dict(n=int(m.sum()), mae=float(mae[torch.from_numpy(m)].mean()),
                              psnr=float(psnr[torch.from_numpy(m)].mean()))
        out["per_attribute_T50"] = per
json.dump(out, open("/home/exouser/SpecRoute/experiments/hdae/outputs/recon_baseline.json", "w"), indent=2)
print("\nwrote experiments/hdae/outputs/recon_baseline.json")
