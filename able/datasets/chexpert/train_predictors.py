"""CheXpert protocol v0, section 5: the attribute predictors and the gate.

One DenseNet-121, ImageNet-pretrained, with two heads sharing the trunk: a
10-way multi-label head (sex, the two modeled findings, the seven eval-only
findings) and a 3-way softmax head for the age bin. Eleven attributes in all.
Sharing one trunk is deliberate -- eleven separate DenseNets would cost eleven
times the GPU for predictors that are measurement instruments, not results.

Three disjoint patient sets, and the distinction matters:

  * fit     -- 90% of the section-4 train split. Gradients only.
  * select  -- the other 10% of the TRAIN split. Picks the epoch.
  * heldout -- the section-4 held-out split, untouched during training.
                Reported, and never selected on, because the evaluation cohort
                is drawn from it; choosing a checkpoint on it would tune the
                instrument on the pool it later has to measure.

So "selection on train-split held-out AUC" is read as a sub-split inside the
train split, which is the reading that keeps the cohort pool clean.

NO HORIZONTAL FLIP. A chest radiograph is not left-right symmetric -- the
heart sits left of midline, and cardiomegaly is judged from the cardiothoracic
ratio measured on that silhouette. Mirroring would teach the predictor that
situs inversus is ordinary and would corrupt the one finding the SCM most
depends on. Augmentation is a small affine only.

The gate: an eval-only label is kept iff its held-out AUC >= 0.85; effusion
and cardiomegaly must also clear 0.85, and if either fails the run stops and
says the graph has to be reconsidered rather than quietly carrying on.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision.models import densenet121, DenseNet121_Weights

BINARY = ["sex", "effusion", "cardiomegaly", "support_devices", "ap_pa",
          "edema", "lung_opacity", "consolidation", "atelectasis", "pneumothorax"]
MODELED_BINARY = ["effusion", "cardiomegaly"]
GATE = 0.85
IMNET_MEAN, IMNET_STD = 0.449, 0.226   # grayscale-collapsed ImageNet statistics


class Chex(Dataset):
    def __init__(self, npy, meta, rows, train):
        self.npy, self.rows, self.train = npy, np.asarray(rows), train
        self.y = meta[BINARY].to_numpy(np.float32)
        self.a = meta["age_bin"].to_numpy(np.int64)
        self.arr = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.arr is None:                      # open per worker, after fork
            self.arr = np.load(self.npy, mmap_mode="r")
        r = int(self.rows[i])
        x = torch.from_numpy(np.asarray(self.arr[r], dtype=np.float32) / 255.0)[None]
        if self.train:
            ang = float(torch.empty(1).uniform_(-10, 10))
            tr = [float(torch.empty(1).uniform_(-0.05, 0.05)) * 224 for _ in range(2)]
            sc = float(torch.empty(1).uniform_(0.95, 1.05))
            x = TF.affine(x, angle=ang, translate=tr, scale=sc, shear=[0.0])
        x = (x - IMNET_MEAN) / IMNET_STD
        return x.expand(3, -1, -1), torch.from_numpy(self.y[r]), int(self.a[r])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.trunk = m.features
        self.bin = nn.Linear(1024, len(BINARY))
        self.age = nn.Linear(1024, 3)

    def forward(self, x):
        h = F.adaptive_avg_pool2d(F.relu(self.trunk(x)), 1).flatten(1)
        return self.bin(h), self.age(h)


@torch.no_grad()
def predict(net, loader, dev):
    net.eval()
    P, A, Y, Ab = [], [], [], []
    for x, y, a in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            lb, la = net(x.to(dev, non_blocking=True))
        P.append(torch.sigmoid(lb.float()).cpu().numpy())
        A.append(torch.softmax(la.float(), 1).cpu().numpy())
        Y.append(y.numpy())
        Ab.append(np.asarray(a))
    return np.concatenate(P), np.concatenate(A), np.concatenate(Y), np.concatenate(Ab)


def scores(p, a_prob, y, a_true):
    out = {}
    for j, k in enumerate(BINARY):
        if len(np.unique(y[:, j])) < 2:
            out[k] = {"auc": None, "balanced_acc": None, "positives": int(y[:, j].sum()),
                      "note": "one class only in this set"}
            continue
        auc = float(roc_auc_score(y[:, j], p[:, j]))
        out[k] = {"auc": round(auc, 4),
                  "balanced_acc": round(float(balanced_accuracy_score(y[:, j], p[:, j] >= 0.5)), 4),
                  "positives": int(y[:, j].sum()), "n": int(len(y))}
    try:
        auc_age = float(roc_auc_score(a_true, a_prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc_age = None
    out["age_bin"] = {"auc_ovr_macro": None if auc_age is None else round(auc_age, 4),
                      "balanced_acc": round(float(balanced_accuracy_score(a_true, a_prob.argmax(1))), 4),
                      "n": int(len(a_true))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/exouser/able_data/chexpert")
    ap.add_argument("--out", default="/home/exouser/SpecRoute/experiments/hdae/outputs/attr_predictors_chexpert")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--select-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: cap each row set at this many images")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = "cuda"

    meta = pd.read_csv(f"{args.data}/chexpert_meta.csv")
    split = np.load(f"{args.data}/split.npy")
    assert len(split) == len(meta)
    train_rows = np.flatnonzero(split == 0)
    held_rows = np.flatnonzero(split == 1)

    # patient-level sub-split of the TRAIN split into fit / select
    tp = np.sort(meta.loc[train_rows, "patient_id"].unique())
    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(tp)
    sel_pat = set(perm[:int(round(len(tp) * args.select_frac))].tolist())
    is_sel = meta["patient_id"].isin(sel_pat).to_numpy()
    fit_rows = train_rows[~is_sel[train_rows]]
    sel_rows = train_rows[is_sel[train_rows]]
    assert set(meta.loc[fit_rows, "patient_id"]).isdisjoint(set(meta.loc[sel_rows, "patient_id"]))
    assert set(meta.loc[train_rows, "patient_id"]).isdisjoint(set(meta.loc[held_rows, "patient_id"]))
    if args.limit:
        fit_rows, sel_rows, held_rows = (r[:args.limit] for r in (fit_rows, sel_rows, held_rows))
        print(f"SMOKE TEST: row sets capped at {args.limit}")
    print(f"fit {len(fit_rows)}  select {len(sel_rows)}  heldout {len(held_rows)} images")

    npy = f"{args.data}/images_224.npy"
    mk = lambda rows, tr, bs: DataLoader(
        Chex(npy, meta, rows, tr), batch_size=bs, shuffle=tr, num_workers=args.workers,
        pin_memory=True, drop_last=tr, persistent_workers=args.workers > 0)
    dl_fit, dl_sel, dl_held = mk(fit_rows, True, args.batch), mk(sel_rows, False, 256), mk(held_rows, False, 256)

    rmeta = pd.read_csv(f"{args.data}/radiologist_val_meta.csv")
    dl_rad = DataLoader(Chex(f"{args.data}/radiologist_val_224.npy", rmeta, np.arange(len(rmeta)), False),
                        batch_size=128, num_workers=2)

    net = Net().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(dl_fit), pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()

    # positive weights from the FIT set only
    pw = torch.tensor([(1 - meta.loc[fit_rows, k].mean()) / max(meta.loc[fit_rows, k].mean(), 1e-6)
                       for k in BINARY], dtype=torch.float32, device=dev).clamp(max=10.0)
    print("pos_weight:", {k: round(float(v), 2) for k, v in zip(BINARY, pw)})

    hist, best, best_state = [], -1.0, None
    for ep in range(args.epochs):
        net.train()
        t0, nb, run = time.time(), 0, 0.0
        for x, y, a in dl_fit:
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            y, a = y.to(dev, non_blocking=True), a.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                lb, la = net(x)
                loss = F.binary_cross_entropy_with_logits(lb, y, pos_weight=pw) + F.cross_entropy(la, a)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss)
            nb += 1
            if nb % 200 == 0:
                print(f"  ep{ep} {nb}/{len(dl_fit)} loss {run/nb:.4f} "
                      f"{nb*args.batch/(time.time()-t0):.0f} img/s", flush=True)
        s = scores(*predict(net, dl_sel, dev))
        mean_auc = float(np.mean([s[k]["auc"] for k in BINARY if s[k]["auc"] is not None]))
        hist.append({"epoch": ep, "train_loss": round(run / nb, 5), "select_mean_auc": round(mean_auc, 4),
                     "minutes": round((time.time() - t0) / 60, 1), "select": s})
        print(f"epoch {ep}: loss {run/nb:.4f}  select mean AUC {mean_auc:.4f}  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        if mean_auc > best:
            best, best_state = mean_auc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            torch.save({"state_dict": best_state, "epoch": ep, "select_mean_auc": mean_auc,
                        "binary": BINARY}, f"{args.out}/best.ckpt")
            print(f"  -> new best, saved epoch {ep}")

    net.load_state_dict(best_state)
    held = scores(*predict(net, dl_held, dev))
    rad = scores(*predict(net, dl_rad, dev))

    kept = [k for k in BINARY if held[k]["auc"] is not None and held[k]["auc"] >= GATE]
    failed_modeled = [k for k in MODELED_BINARY if k not in kept]
    rep = {
        "protocol": "chexpert v0 section 5",
        "arch": "densenet121 imagenet1k_v1, shared trunk, 10-way multilabel + 3-way age head",
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "seed": args.seed,
        "augmentation": "random affine +-10 deg, +-5% translate, 0.95-1.05 scale; NO horizontal flip",
        "n_fit": int(len(fit_rows)), "n_select": int(len(sel_rows)), "n_heldout": int(len(held_rows)),
        "n_radiologist_val": int(len(rmeta)),
        "selected_epoch": int(max(hist, key=lambda h: h["select_mean_auc"])["epoch"]),
        "gate": GATE, "heldout": held, "radiologist_val": rad,
        "kept_labels": kept,
        "dropped_by_gate": [k for k in BINARY if k not in kept],
        "modeled_failing_gate": failed_modeled,
        "history": [{k: v for k, v in h.items() if k != "select"} for h in hist],
        "history_full": hist,
    }
    with open(f"{args.out}/predictor_report.json", "w") as f:
        json.dump(rep, f, indent=2)

    print("\n%-17s %8s %8s %8s" % ("label", "held AUC", "bal.acc", "rad AUC"))
    for k in BINARY:
        ra = rad[k]["auc"]
        print("%-17s %8s %8s %8s %s" % (k, held[k]["auc"], held[k]["balanced_acc"],
                                        "n/a" if ra is None else ra,
                                        "KEPT" if k in kept else "dropped (AUC < %.2f)" % GATE))
    print("%-17s %8s %8s" % ("age_bin", held["age_bin"]["auc_ovr_macro"], held["age_bin"]["balanced_acc"]))
    print(f"\nreport -> {args.out}/predictor_report.json")
    if failed_modeled:
        raise SystemExit(
            f"GATE FAILURE on modeled attribute(s) {failed_modeled}: held-out AUC below {GATE}. "
            "The protocol says to stop here and reconsider the graph, not to proceed.")


if __name__ == "__main__":
    main()
