#!/usr/bin/env python
"""Plot and summarize per-level CelebA linear-probe results."""
import argparse, csv, json, logging, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def read_rows(metrics_csv):
    with open(metrics_csv, newline="") as f:
        return list(csv.DictReader(f))


def metric_matrix(rows, metric="test_balanced_accuracy"):
    levels = sorted({int(r["level"]) for r in rows})
    attrs = sorted({r["attribute_name"] for r in rows})
    matrix = np.full((len(levels), len(attrs)), np.nan, dtype=float)
    level_to_i = {level: i for i, level in enumerate(levels)}
    attr_to_j = {attr: j for j, attr in enumerate(attrs)}
    for row in rows:
        matrix[level_to_i[int(row["level"])]][attr_to_j[row["attribute_name"]]] = float(row[metric])
    return levels, attrs, matrix


def best_level_summary(rows, metric="test_balanced_accuracy"):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["attribute_name"]].append(row)
    summary = []
    for attr, attr_rows in sorted(grouped.items()):
        best = max(attr_rows, key=lambda r: float(r[metric]))
        summary.append({"attribute_name": attr, "best_level": int(best["level"]),
                        "best_metric": float(best[metric]),
                        "best_latent_key": best.get("latent_key", f"z_level_{best['level']}")})
    return summary


def write_best_summary(path, summary):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attribute_name", "best_level", "best_metric", "best_latent_key"])
        writer.writeheader(); writer.writerows(summary)


def plot_heatmap(path, levels, attrs, matrix, metric):
    import matplotlib.pyplot as plt
    fig_w = max(10, len(attrs) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, 3 + len(levels) * 0.5))
    im = ax.imshow(matrix, aspect="auto", vmin=0.5, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(attrs))); ax.set_xticklabels(attrs, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(levels))); ax.set_yticklabels([f"Z{level + 1}" for level in levels])
    ax.set_title(f"Linear probe {metric} by latent level")
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_best_counts(path, summary):
    import matplotlib.pyplot as plt
    counts = Counter(row["best_level"] for row in summary)
    levels = sorted(counts)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"Z{level + 1}" for level in levels], [counts[level] for level in levels])
    ax.set_ylabel("# attributes best predicted")
    ax.set_title("Best latent level per attribute")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe-metrics", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--metric", default="test_balanced_accuracy")
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.probe_metrics)
    levels, attrs, matrix = metric_matrix(rows, args.metric)
    summary = best_level_summary(rows, args.metric)
    write_best_summary(out / "best_level_by_attribute.csv", summary)
    json_summary = {"metric": args.metric, "levels": levels, "num_attributes": len(attrs),
                    "best_level_counts": dict(Counter(row["best_level"] for row in summary)),
                    "best_level_csv": str(out / "best_level_by_attribute.csv")}
    (out / "analysis_summary.json").write_text(json.dumps(json_summary, indent=2))
    np.save(out / "metric_matrix.npy", matrix)
    logging.info("wrote summaries to %s", out)
    plot_heatmap(out / "probe_heatmap.png", levels, attrs, matrix, args.metric)
    plot_best_counts(out / "best_level_counts.png", summary)
    logging.info("wrote plots: probe_heatmap.png, best_level_counts.png")


if __name__ == "__main__":
    main()
