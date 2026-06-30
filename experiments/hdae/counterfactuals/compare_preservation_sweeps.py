#!/usr/bin/env python
"""Compare preservation-sweep CSVs across HDAE configs with summary plots."""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

import numpy as np

DEFAULT_METRICS = [
    "non_target_abs_delta_mean",
    "non_target_flip_fraction",
    "non_target_severe_fraction",
    "target_delta_abs_mean",
    "target_intended_flip_rate",
]


def parse_named_paths(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"expected NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        if not name:
            raise ValueError(f"empty NAME in {item!r}")
        out[name] = Path(path)
    return out


def parse_csv_list(value):
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def _maybe_float(value):
    if value in (None, ""):
        return value
    try:
        return float(value)
    except ValueError:
        return value


def _load_summary(csv_path):
    summary_path = csv_path.parent / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {}


def _load_config_meta(config_path):
    if not config_path:
        return {}, {}
    import yaml
    raw = yaml.safe_load(Path(config_path).read_text())
    dims = {i: int(dim) for i, dim in enumerate(raw["encoder"]["level_dims"])}
    resolutions = {i: int(res) for i, res in enumerate(raw["encoder"].get("tap_resolutions", []))}
    return dims, resolutions


def read_sweep(name, csv_path, config_path=None):
    """Read one preservation_sweep.csv and annotate config, dims, resolutions."""
    summary = _load_summary(csv_path)
    dims = {int(k): int(v) for k, v in summary.get("level_dims", {}).items()}
    config_dims, resolutions = _load_config_meta(config_path)
    dims.update(config_dims)
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            level = int(row["level"])
            rec = {k: _maybe_float(v) for k, v in row.items()}
            rec["config"] = name
            rec["level"] = level
            rec["strength"] = float(row["strength"])
            rec["level_dim"] = int(float(row.get("level_dim") or dims.get(level, 0)))
            rec["resolution"] = resolutions.get(level, level)
            rows.append(rec)
    return rows


def filter_rows(rows, attributes=None, directions=None, strengths=None):
    attr_set = set(attributes) if attributes else None
    dir_set = set(directions) if directions else None
    strength_set = {float(x) for x in strengths} if strengths else None
    out = []
    for row in rows:
        if attr_set and row.get("attribute") not in attr_set:
            continue
        if dir_set and row.get("direction") not in dir_set:
            continue
        if strength_set and float(row.get("strength")) not in strength_set:
            continue
        out.append(row)
    return out


def mean(rows, metric):
    vals = [float(row[metric]) for row in rows if row.get(metric) not in (None, "")]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(rows, group_keys, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_keys)].append(row)
    out = []
    for key, group in sorted(grouped.items(), key=lambda kv: kv[0]):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec["num_rows"] = len(group)
        for metric in metrics:
            rec[metric] = mean(group, metric)
        out.append(rec)
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    preferred = ["config", "attribute", "direction", "level", "resolution", "level_dim", "strength"]
    fieldnames = preferred + sorted({k for row in rows for k in row.keys()} - set(preferred))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def pivot_matrix(rows, y_key, x_values, y_values, metric):
    by = {(row[y_key], float(row["strength"])): row for row in rows}
    mat = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            row = by.get((y, float(x)))
            if row and row.get(metric) not in (None, ""):
                mat[i, j] = float(row[metric])
    return mat


def plot_metric_vs_strength(rows, configs, strengths, metric, out_path, dpi=200):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for config in configs:
        ys = []
        for strength in strengths:
            group = [r for r in rows if r["config"] == config and float(r["strength"]) == float(strength)]
            ys.append(mean(group, metric))
        ax.plot(strengths, ys, marker="o", label=config)
    ax.set_xlabel("CF strength")
    ax.set_ylabel(metric)
    ax.set_title(f"Preservation sweep: {metric}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)


def plot_metric_by_level(rows, configs, metric, strength, out_path, y_key="resolution", dpi=200):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(configs), figsize=(max(4 * len(configs), 5), 4), squeeze=False, sharey=False)
    for ax, config in zip(axes[0], configs):
        subset = [r for r in rows if r["config"] == config and abs(float(r["strength"]) - strength) < 1e-9]
        grouped = aggregate(subset, [y_key], [metric])
        xs = [row[y_key] for row in grouped]
        ys = [row[metric] for row in grouped]
        ax.bar([str(x) for x in xs], ys)
        ax.set_title(config)
        ax.set_xlabel(y_key)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0][0].set_ylabel(metric)
    fig.suptitle(f"{metric} by {y_key} at strength={strength:g}")
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)


def plot_attribute_direction_heatmaps(rows, configs, strengths, metric, out_path, attribute, direction, dpi=200):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(configs), figsize=(max(4 * len(configs), 5), 4), squeeze=False)
    mats = []
    labels = []
    for config in configs:
        subset = [r for r in rows if r["config"] == config and r["attribute"] == attribute and r["direction"] == direction]
        y_values = sorted({r["resolution"] for r in subset})
        mat = pivot_matrix(subset, "resolution", strengths, y_values, metric)
        mats.append(mat); labels.append(y_values)
    finite = np.concatenate([m[np.isfinite(m)] for m in mats if np.isfinite(m).any()]) if any(np.isfinite(m).any() for m in mats) else np.array([0.0])
    vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    for ax, config, mat, y_values in zip(axes[0], configs, mats, labels):
        im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(config)
        ax.set_xticks(np.arange(len(strengths))); ax.set_xticklabels([f"{s:g}" for s in strengths])
        ax.set_yticks(np.arange(len(y_values))); ax.set_yticklabels([str(y) for y in y_values])
        ax.set_xlabel("strength")
    axes[0][0].set_ylabel("resolution")
    fig.suptitle(f"{attribute} / {direction}: {metric}")
    fig.colorbar(im, ax=axes.ravel().tolist(), label=metric, shrink=0.8)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def plot_tradeoff(rows, configs, strength, out_path, dpi=200):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    for config in configs:
        subset = [r for r in rows if r["config"] == config and abs(float(r["strength"]) - strength) < 1e-9]
        x = mean(subset, "non_target_abs_delta_mean")
        y = mean(subset, "target_intended_flip_rate")
        ax.scatter([x], [y], s=80, label=config)
        ax.annotate(config, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("non-target abs delta mean (lower is better)")
    ax.set_ylabel("target intended flip rate (higher is better)")
    ax.set_title(f"Edit/preservation tradeoff at strength={strength:g}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweeps", nargs="+", required=True, help="NAME=path/to/preservation_sweep.csv entries")
    p.add_argument("--configs", nargs="*", default=[], help="Optional NAME=path/to/config.yaml entries for resolution labels")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    p.add_argument("--attributes", default=None, help="Optional comma-separated attribute subset")
    p.add_argument("--directions", default=None, help="Optional comma-separated directions, e.g. positive,negative")
    p.add_argument("--strengths", default=None, help="Optional comma-separated strength subset")
    p.add_argument("--tradeoff-strength", type=float, default=1.0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--max-heatmaps", type=int, default=80, help="Safety cap for per-attribute/direction heatmaps")
    args = p.parse_args()

    sweeps = parse_named_paths(args.sweeps)
    configs = parse_named_paths(args.configs)
    metrics = parse_csv_list(args.metrics) or DEFAULT_METRICS
    attributes = parse_csv_list(args.attributes)
    directions = parse_csv_list(args.directions)
    strengths_filter = [float(x) for x in parse_csv_list(args.strengths)] if args.strengths else None

    rows = []
    for name, path in sweeps.items():
        rows.extend(read_sweep(name, path, configs.get(name)))
    rows = filter_rows(rows, attributes=attributes, directions=directions, strengths=strengths_filter)
    if not rows:
        raise ValueError("no preservation rows after filtering")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    configs_order = list(sweeps.keys())
    strengths = sorted({float(r["strength"]) for r in rows})
    attrs = sorted({r["attribute"] for r in rows})
    dirs = sorted({r["direction"] for r in rows})

    combined_path = out / "combined_preservation_sweep.csv"
    write_csv(combined_path, rows)
    agg_strength = aggregate(rows, ["config", "strength"], metrics)
    agg_strength_path = out / "aggregate_by_config_strength.csv"
    write_csv(agg_strength_path, agg_strength)
    agg_level = aggregate(rows, ["config", "resolution", "level", "strength"], metrics)
    agg_level_path = out / "aggregate_by_config_level_strength.csv"
    write_csv(agg_level_path, agg_level)

    plots = []
    for metric in metrics:
        path = out / f"metric_vs_strength_{metric}.png"
        plot_metric_vs_strength(rows, configs_order, strengths, metric, path, dpi=args.dpi)
        plots.append(str(path))
        path = out / f"metric_by_resolution_s{args.tradeoff_strength:g}_{metric}.png"
        plot_metric_by_level(rows, configs_order, metric, args.tradeoff_strength, path, y_key="resolution", dpi=args.dpi)
        plots.append(str(path))

    tradeoff_path = out / f"tradeoff_s{args.tradeoff_strength:g}.png"
    plot_tradeoff(rows, configs_order, args.tradeoff_strength, tradeoff_path, dpi=args.dpi)
    plots.append(str(tradeoff_path))

    heatmap_dir = out / "attribute_direction_heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    made = 0
    for metric in metrics:
        for attr in attrs:
            for direction in dirs:
                if made >= args.max_heatmaps:
                    break
                path = heatmap_dir / f"{attr}_{direction}_{metric}.png"
                plot_attribute_direction_heatmaps(rows, configs_order, strengths, metric, path, attr, direction, dpi=args.dpi)
                plots.append(str(path)); made += 1
            if made >= args.max_heatmaps:
                break
        if made >= args.max_heatmaps:
            break

    summary = {
        "configs": configs_order,
        "num_rows": len(rows),
        "attributes": attrs,
        "directions": dirs,
        "strengths": strengths,
        "metrics": metrics,
        "combined_csv": str(combined_path),
        "aggregate_by_config_strength_csv": str(agg_strength_path),
        "aggregate_by_config_level_strength_csv": str(agg_level_path),
        "plots": plots,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
