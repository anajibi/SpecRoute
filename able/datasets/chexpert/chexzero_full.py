"""CheXzero zero-shot over the WHOLE frozen row set, not just radiologist_val.

Same pipeline as chexzero_eval.py -- CheXzero's own Normalize, the ("{}",
"no {}") prompt pair, cosine similarity, softmax over the pair, the
checkpoint ensemble averaged -- extended to all 171,996 frozen rows so the
reader can be compared against ours and XRV on train and test as well.

Two changes forced by scale:

 1. Streaming. chexzero_eval.py materialised the whole set as one float32
    tensor; at 171,996 x 3 x 224 x 224 that is 103 GB. Images are read from
    the memmap in batches instead.
 2. One pass over disk, not one per checkpoint. All checkpoints are resident
    on the GPU at once (ViT-B/32 is ~350 MB each) and every batch is scored by
    all of them, so the 8.6 GB memmap is read once rather than ten times.

The same input deviation as before still applies and is still recorded: these
are the 224 centre crops, not CheXzero's letterbox-to-320-then-224, because
the source JPEGs went with the deleted archive. Identical pixels to our own
classifier, so the comparison is controlled on input.

Checkpoints load with weights_only=True: a .pt is a pickle and these came off
a Google Drive link.
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

MEAN, STD = 101.48761, 83.43944
PAIR = ("{}", "no {}")
# CheXpert's own label vocabulary, in CheXzero's phrasing. The first five are
# CheXzero's published competition pathologies and double as the sanity gate.
LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion",
          "Enlarged Cardiomediastinum", "Lung Opacity", "Lung Lesion", "Pneumonia",
          "Pneumothorax", "Pleural Other", "Fracture", "Support Devices"]
GATE_LABELS = LABELS[:5]


def text_weights(model, labels, template, dev):
    import clip
    with torch.no_grad():
        W = []
        for c in labels:
            t = clip.tokenize([template.format(c)], context_length=77).to(dev)
            e = model.encode_text(t)
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(dim=0)
            W.append(e / e.norm())
        return torch.stack(W, dim=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    import clip

    dev = a.device if torch.cuda.is_available() else "cpu"
    paths = sorted(glob.glob(os.path.join(a.weights, "*.pt")))
    if not paths:
        raise SystemExit(f"no checkpoints in {a.weights}")
    print(f"{len(paths)} checkpoints, device {dev}", flush=True)

    models, Wpos, Wneg = [], [], []
    for p in paths:
        m, _ = clip.load("ViT-B/32", device="cpu", jit=False)
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        m = m.to(dev).eval()
        models.append(m)
        Wpos.append(text_weights(m, LABELS, PAIR[0], dev))
        Wneg.append(text_weights(m, LABELS, PAIR[1], dev))
        print(f"  resident: {os.path.basename(p)}", flush=True)

    arr = np.load(a.npy, mmap_mode="r")
    N, L = len(arr), len(LABELS)
    acc = np.zeros((N, L), np.float64)
    with torch.no_grad():
        for i in range(0, N, a.bs):
            x = torch.from_numpy(np.asarray(arr[i:i + a.bs], dtype=np.float32))
            x = ((x - MEAN) / STD)[:, None].repeat(1, 3, 1, 1).to(dev, non_blocking=True)
            for m, wp, wn in zip(models, Wpos, Wneg):
                f = m.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                pos = (f @ wp).float()
                neg = (f @ wn).float()
                # softmax over the pos/neg pair, per label
                acc[i:i + len(x)] += torch.sigmoid(pos - neg).double().cpu().numpy()
            if (i // a.bs) % 50 == 0:
                print(f"  {i}/{N}", flush=True)
    P = (acc / len(models)).astype(np.float32)

    os.makedirs(a.out, exist_ok=True)
    base = os.path.basename(a.npy).replace(".npy", "")
    np.save(f"{a.out}/chexzero_{base}_probs.npy", P)
    json.dump({"labels": LABELS, "gate_labels": GATE_LABELS, "prompt_pair": list(PAIR),
               "n": int(N), "npy": a.npy,
               "checkpoints": [os.path.basename(p) for p in paths],
               "note": ("softmax over the ('{}','no {}') pair is sigmoid(pos-neg); "
                        "ensemble is the mean of per-checkpoint probabilities"),
               "input_deviation": ("224 centre crops, not CheXzero's letterbox-to-320-"
                                   "then-224; source JPEGs gone with the deleted archive")},
              open(f"{a.out}/chexzero_{base}_meta.json", "w"), indent=2)
    print(f"wrote {a.out}/chexzero_{base}_probs.npy {P.shape}", flush=True)
