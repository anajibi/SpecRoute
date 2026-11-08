"""Score this project's own classifiers over the WHOLE frozen row set.

predictor_report.json only ever recorded held-out and radiologist_val. A
reader comparison on train as well as test needs probabilities everywhere, so
both frozen checkpoints are run over all 171,996 rows plus radiologist_val.

Inference only. Neither checkpoint is opened for writing.

The 11-head checkpoint is a MEASUREMENT artifact, not part of the protocol: it
exists so enlarged_cardiomediastinum has one of our own numbers to be judged
by. Its ten shared heads moved by mean |delta| 0.0097 against best.ckpt, so it
is reported beside the frozen predictor rather than instead of it.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torchvision.models import densenet121

IMNET_MEAN, IMNET_STD = 0.449, 0.226
BINARY10 = ["sex", "effusion", "cardiomegaly", "support_devices", "ap_pa",
            "edema", "lung_opacity", "consolidation", "atelectasis", "pneumothorax"]
BINARY11 = BINARY10 + ["enlarged_cardiomediastinum"]


class Net(nn.Module):
    def __init__(self, n_bin):
        super().__init__()
        self.trunk = densenet121(weights=None).features
        self.bin = nn.Linear(1024, n_bin)
        self.age = nn.Linear(1024, 3)

    def forward(self, x):
        h = Fn.adaptive_avg_pool2d(Fn.relu(self.trunk(x)), 1).flatten(1)
        return self.bin(h), self.age(h)


@torch.no_grad()
def run(ckpt, n_bin, arr, dev, bs=192):
    net = Net(n_bin).to(dev).eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
    missing = net.load_state_dict(sd, strict=True)
    print(f"  loaded {ckpt} {missing}")
    P, A = [], []
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(np.asarray(arr[i:i + bs], dtype=np.float32) / 255.0)[:, None]
        x = ((x - IMNET_MEAN) / IMNET_STD).expand(-1, 3, -1, -1).to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            lb, la = net(x)
        P.append(torch.sigmoid(lb.float()).cpu().numpy())
        A.append(torch.softmax(la.float(), 1).cpu().numpy())
        if (i // bs) % 100 == 0:
            print(f"    {i}/{len(arr)}", flush=True)
    del net
    torch.cuda.empty_cache()
    return np.concatenate(P), np.concatenate(A)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/exouser/able_data/chexpert")
    ap.add_argument("--ckpt10", default="/home/exouser/able_data/chexpert/predictors/best.ckpt")
    ap.add_argument("--ckpt11", default="/home/exouser/SpecRoute/experiments/hdae/"
                                       "outputs/attr_predictors_chexpert_head11/best.ckpt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    for tag, npy, n_bin, ck in [
        ("frozen_10", f"{a.data}/images_224.npy", 10, a.ckpt10),
        ("radval_10", f"{a.data}/radiologist_val_224.npy", 10, a.ckpt10),
        ("frozen_11", f"{a.data}/images_224.npy", 11, a.ckpt11),
        ("radval_11", f"{a.data}/radiologist_val_224.npy", 11, a.ckpt11),
    ]:
        arr = np.load(npy, mmap_mode="r")
        print(f"{tag}: {arr.shape} from {os.path.basename(ck)}", flush=True)
        P, A = run(ck, n_bin, arr, dev)
        np.save(f"{a.out}/ours{n_bin}_{tag.split('_')[0]}_bin.npy", P.astype(np.float32))
        np.save(f"{a.out}/ours{n_bin}_{tag.split('_')[0]}_age.npy", A.astype(np.float32))
        print(f"  wrote {P.shape} {A.shape}", flush=True)
