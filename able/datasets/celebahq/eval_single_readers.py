"""Evaluate the 40 per-attribute CelebA readers on the test partition and compare.

The per-attribute form is the canonical one at 256px (readers40_per_attribute.tar.gz). This
scores the 64px equivalents on partition 2 and puts them beside the multi-head runs, so the
choice between them is made on held-out numbers rather than on which was trained last.
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/hdae/outputs/attr_predictors_celebahq64/single")
    ap.add_argument("--compare", nargs="*", default=[
        "experiments/hdae/outputs/attr_predictors_celebahq64/test_eval64.json",
        "experiments/hdae/outputs/attr_predictors_celebahq64/test_eval64_convnext_tiny.json",
        "experiments/hdae/outputs/attr_predictors_celebahq64/test_eval64_convnext_base.json"])
    ap.add_argument("--out", default="experiments/hdae/outputs/attr_predictors_celebahq64/single_eval.json")
    a = ap.parse_args()

    per = {}
    cks = sorted(glob.glob(os.path.join(REPO, a.dir, "*.pt")))
    print(f"{len(cks)} per-attribute checkpoints")
    for i, ck in enumerate(cks, 1):
        name = os.path.basename(ck)[:-3]
        tmp = f"/tmp/_se_{name}.json"
        r = subprocess.run([os.path.join(REPO, ".venv/bin/python"),
                            os.path.join(REPO, "able/datasets/celebahq/eval_readers.py"),
                            "--ckpt", os.path.relpath(ck, REPO), "--out", tmp],
                           capture_output=True, text=True, cwd=REPO)
        if not os.path.exists(tmp):
            print(f"  [{i}/{len(cks)}] {name}: FAILED {r.stderr.strip()[-120:]}"); continue
        d = json.load(open(tmp))
        per[name] = d["attributes"][name]
        os.remove(tmp)
        print(f"  [{i}/{len(cks)}] {name:22s} bal {per[name]['bal']:.4f}  F1 {per[name]['f1']:.4f}"
              f"  {'PASS' if per[name]['admitted'] else 'fail'}", flush=True)

    mb = float(np.mean([v["bal"] for v in per.values()]))
    mf = float(np.mean([v["f1"] for v in per.values()]))
    na = sum(1 for v in per.values() if v["admitted"])
    print(f"\nper-attribute: mean bal {mb:.4f}  mean F1 {mf:.4f}  admitted {na}/{len(per)}")

    others = {}
    for p in a.compare:
        fp = os.path.join(REPO, p)
        if os.path.exists(fp):
            d = json.load(open(fp))
            others[f"{d['arch']}"] = d
            print(f"  {d['arch']:18s} mean bal {d['mean_bal']:.4f}  mean F1 {d['mean_f1']:.4f}"
                  f"  admitted {d['n_admitted']}/40")

    # best of everything, per attribute, selected on F1
    best = {}
    for name, v in per.items():
        cand = [("per_attribute", v)]
        for k, d in others.items():
            if name in d["attributes"]:
                cand.append((k, d["attributes"][name]))
        k, b = max(cand, key=lambda kv: kv[1]["f1"])
        best[name] = {"source": k, **b}
    wins = {}
    for v in best.values():
        wins[v["source"]] = wins.get(v["source"], 0) + 1
    bb = float(np.mean([v["bal"] for v in best.values()]))
    bf = float(np.mean([v["f1"] for v in best.values()]))
    ba = sum(1 for v in best.values() if v["admitted"])
    print(f"\nbest-of-all per attribute: mean bal {bb:.4f}  mean F1 {bf:.4f}  admitted {ba}/{len(best)}")
    print(f"  wins: {wins}")
    json.dump({"per_attribute": per, "mean_bal": round(mb, 4), "mean_f1": round(mf, 4),
               "n_admitted": na, "best_of_all": best, "wins": wins,
               "best_mean_bal": round(bb, 4), "best_mean_f1": round(bf, 4),
               "best_n_admitted": ba},
              open(os.path.join(REPO, a.out), "w"), indent=1)
    print(f"wrote {a.out}")
