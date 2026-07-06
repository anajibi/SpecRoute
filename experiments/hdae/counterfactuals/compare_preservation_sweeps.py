#!/usr/bin/env python
"""Compare preservation sweeps across configs without per-attribute plots.

The comparison unit is a latent level, not a whole config. Rows are first
aggregated over attributes at each config/level/direction/strength, then each
latent level chooses its best counterfactual strength by an aggregate target
change metric. This avoids comparing a K=5 model averaged over five latents to a
K=3 model averaged over three latents.
"""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))

import numpy as np

DEFAULT_METRICS = [
    "target_delta_abs_mean",
    "target_intended_flip_rate",
    "non_target_abs_delta_mean",
    "non_target_flip_fraction",
    "non_target_severe_fraction",
    "non_target_abs_delta_per_target_delta",
]
PRESERVATION_METRICS = [
    "non_target_abs_delta_mean",
    "non_target_flip_fraction",
    "non_target_severe_fraction",
    "non_target_abs_delta_per_target_delta",
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
            rec["latent_label"] = f"{name}:Z{level + 1}"
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
        rec["num_attributes"] = len({r.get("attribute") for r in group})
        rec["num_rows"] = len(group)
        for metric in metrics:
            rec[metric] = mean(group, metric)
        out.append(rec)
    return out


def add_target_normalized_metrics(rows, eps=1e-8):
    """Add preservation cost normalized by target movement to each row."""
    out = []
    for row in rows:
        rec = dict(row)
        target = abs(float(rec.get("target_delta_abs_mean", 0.0)))
        nontarget = float(rec.get("non_target_abs_delta_mean", float("nan")))
        rec["non_target_abs_delta_per_target_delta"] = nontarget / max(target, eps)
        rec["target_delta_norm_eps"] = eps
        out.append(rec)
    return out


def aggregate_attributes_by_latent_strength(rows, metrics, combine_directions=False, target_norm_eps=1e-8):
    """Average metrics over attributes while keeping latent levels separate."""
    base_metrics = [m for m in metrics if m not in {"non_target_abs_delta_per_target_delta", "target_delta_norm_eps"}]
    group_keys = ["config", "level", "resolution", "level_dim", "strength"]
    if not combine_directions:
        group_keys.append("direction")
    agg = aggregate(rows, group_keys, base_metrics)
    return add_target_normalized_metrics(agg, eps=target_norm_eps)


def select_best_strength_by_latent(agg_rows, selection_metric="target_delta_abs_mean", combine_directions=False):
    """Pick one CF strength per latent by maximum aggregate target change."""
    group_keys = ["config", "level", "resolution", "level_dim"]
    if not combine_directions:
        group_keys.append("direction")
    grouped = defaultdict(list)
    for row in agg_rows:
        grouped[tuple(row[k] for k in group_keys)].append(row)
    selected = []
    for key, group in sorted(grouped.items(), key=lambda kv: kv[0]):
        best = max(group, key=lambda r: (float(r.get(selection_metric, float("nan"))), -float(r["strength"])))
        rec = dict(best)
        rec["selected_strength"] = float(best["strength"])
        rec["selection_metric"] = selection_metric
        selected.append(rec)
    return selected


def _fmt(value):
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(_fmt(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_markdown_report(path, selected_rows, metrics, selection_metric, combine_directions):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(selected_rows, key=lambda r: (r["config"], r.get("direction", ""), int(r["level"])))
    columns = ["config"]
    if not combine_directions:
        columns.append("direction")
    columns.extend([
        "level", "resolution", "level_dim", "selected_strength", selection_metric,
        "target_intended_flip_rate", "non_target_abs_delta_mean",
        "non_target_abs_delta_per_target_delta", "non_target_flip_fraction",
        "non_target_severe_fraction", "num_attributes",
    ])
    available = [c for c in columns if any(c in row for row in rows)]
    best_ratio = sorted(rows, key=lambda r: float(r.get("non_target_abs_delta_per_target_delta", float("inf"))))[:10]
    best_target = sorted(rows, key=lambda r: float(r.get(selection_metric, float("-inf"))), reverse=True)[:10]
    lines = [
        "# Preservation sweep comparison",
        "",
        "Rows aggregate over attributes while keeping each latent level separate.",
        f"CF strength is selected per latent by maximizing `{selection_metric}`.",
        "The main normalized preservation cost is `non_target_abs_delta_mean / target_delta_abs_mean`; lower is better, especially when target change is non-trivial.",
        "",
        "## Selected strength by latent",
        markdown_table(rows, available),
        "",
        "## Best normalized preservation cost (lower is better)",
        markdown_table(best_ratio, available),
        "",
        "## Largest selected target change",
        markdown_table(best_target, available),
        "",
        "## Metrics plotted",
        "\n".join(f"- `{m}`" for m in metrics),
        "",
    ]
    path.write_text("\n".join(lines))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    preferred = ["config", "direction", "level", "resolution", "level_dim", "strength", "selected_strength",
                 "selection_metric", "num_attributes", "num_rows"]
    fieldnames = preferred + sorted({k for row in rows for k in row.keys()} - set(preferred))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def _x_labels(rows):
    return [f"{r['config']}\nZ{int(r['level']) + 1}\n{int(r['resolution'])}px" for r in rows]


def plot_selected_metric_by_latent(selected_rows, metric, out_path, direction=None, dpi=200):
    import matplotlib.pyplot as plt
    rows = [r for r in selected_rows if direction is None or r.get("direction") == direction]
    rows = sorted(rows, key=lambda r: (r["config"], int(r["level"])))
    fig_w = max(8, len(rows) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    colors = {cfg: i for i, cfg in enumerate(sorted({r["config"] for r in rows}))}
    cmap = plt.get_cmap("tab10")
    ax.bar(np.arange(len(rows)), [float(r[metric]) for r in rows],
           color=[cmap(colors[r["config"]] % 10) for r in rows])
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(_x_labels(rows), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    title = f"Selected-strength latent comparison: {metric}"
    if direction:
        title += f" ({direction})"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)


def plot_strength_curves_by_latent(agg_rows, metric, out_path, direction=None, dpi=200):
    import matplotlib.pyplot as plt
    rows = [r for r in agg_rows if direction is None or r.get("direction") == direction]
    configs = sorted({r["config"] for r in rows})
    fig, axes = plt.subplots(1, len(configs), figsize=(max(4 * len(configs), 6), 4), squeeze=False, sharey=True)
    for ax, config in zip(axes[0], configs):
        cfg_rows = [r for r in rows if r["config"] == config]
        levels = sorted({int(r["level"]) for r in cfg_rows})
        for level in levels:
            lvl = sorted([r for r in cfg_rows if int(r["level"]) == level], key=lambda r: float(r["strength"]))
            label = f"Z{level + 1} ({int(lvl[0]['resolution'])}px)"
            ax.plot([float(r["strength"]) for r in lvl], [float(r[metric]) for r in lvl], marker="o", label=label)
        ax.set_title(config)
        ax.set_xlabel("CF strength")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel(metric)
    title = f"Attribute-aggregated latent strength curves: {metric}"
    if direction:
        title += f" ({direction})"
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)


def plot_tradeoff_by_latent(selected_rows, out_path, direction=None, dpi=200):
    import matplotlib.pyplot as plt
    rows = [r for r in selected_rows if direction is None or r.get("direction") == direction]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    configs = sorted({r["config"] for r in rows})
    cmap = plt.get_cmap("tab10")
    for ci, config in enumerate(configs):
        cfg_rows = sorted([r for r in rows if r["config"] == config], key=lambda r: int(r["level"]))
        x = [float(r["non_target_abs_delta_mean"]) for r in cfg_rows]
        y = [float(r["target_delta_abs_mean"]) for r in cfg_rows]
        ax.plot(x, y, marker="o", label=config, color=cmap(ci % 10))
        for r, xi, yi in zip(cfg_rows, x, y):
            ax.annotate(f"Z{int(r['level']) + 1}", (xi, yi), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("non-target abs delta mean (lower is better)")
    ax.set_ylabel("target delta abs mean at selected strength (higher is stronger)")
    title = "Per-latent edit/preservation tradeoff"
    if direction:
        title += f" ({direction})"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweeps", nargs="+", required=True, help="NAME=path/to/preservation_sweep.csv entries")
    p.add_argument("--configs", nargs="*", default=[], help="Optional NAME=path/to/config.yaml entries for resolution labels")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    p.add_argument("--selection-metric", default="target_delta_abs_mean",
                   help="Metric maximized over CF strength separately for each latent level.")
    p.add_argument("--target-norm-eps", type=float, default=1e-8,
                   help="Epsilon denominator for non_target_abs_delta_mean / target_delta_abs_mean.")
    p.add_argument("--attributes", default=None, help="Optional comma-separated attribute subset before aggregation")
    p.add_argument("--directions", default=None, help="Optional comma-separated directions, e.g. positive,negative")
    p.add_argument("--strengths", default=None, help="Optional comma-separated strength subset")
    p.add_argument("--combine-directions", action="store_true",
                   help="Aggregate directions together; by default positive/negative are kept separate.")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    sweeps = parse_named_paths(args.sweeps)
    configs = parse_named_paths(args.configs)
    metrics = parse_csv_list(args.metrics) or DEFAULT_METRICS
    if args.selection_metric not in metrics:
        metrics = [args.selection_metric, *metrics]
    attributes = parse_csv_list(args.attributes)
    directions = parse_csv_list(args.directions)
    strengths_filter = [float(x) for x in parse_csv_list(args.strengths)] if args.strengths else None

    raw_rows = []
    for name, path in sweeps.items():
        raw_rows.extend(read_sweep(name, path, configs.get(name)))
    raw_rows = filter_rows(raw_rows, attributes=attributes, directions=directions, strengths=strengths_filter)
    if not raw_rows:
        raise ValueError("no preservation rows after filtering")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    agg_rows = aggregate_attributes_by_latent_strength(
        raw_rows, metrics, combine_directions=args.combine_directions, target_norm_eps=args.target_norm_eps)
    selected_rows = select_best_strength_by_latent(
        agg_rows, selection_metric=args.selection_metric, combine_directions=args.combine_directions)

    combined_path = out / "combined_preservation_sweep.csv"
    write_csv(combined_path, raw_rows)
    agg_path = out / "attribute_aggregated_by_latent_strength.csv"
    write_csv(agg_path, agg_rows)
    selected_path = out / "selected_strength_by_latent.csv"
    write_csv(selected_path, selected_rows)
    markdown_path = out / "preservation_comparison.md"
    write_markdown_report(markdown_path, selected_rows, metrics, args.selection_metric, args.combine_directions)

    directions_for_plots = [None] if args.combine_directions else sorted({r["direction"] for r in selected_rows})
    plots = []
    for direction in directions_for_plots:
        suffix = "combined" if direction is None else direction
        for metric in metrics:
            path = out / f"selected_by_latent_{suffix}_{metric}.png"
            plot_selected_metric_by_latent(selected_rows, metric, path, direction=direction, dpi=args.dpi)
            plots.append(str(path))
            path = out / f"strength_curves_by_latent_{suffix}_{metric}.png"
            plot_strength_curves_by_latent(agg_rows, metric, path, direction=direction, dpi=args.dpi)
            plots.append(str(path))
        path = out / f"tradeoff_by_latent_{suffix}.png"
        plot_tradeoff_by_latent(selected_rows, path, direction=direction, dpi=args.dpi)
        plots.append(str(path))

    summary = {
        "configs": list(sweeps.keys()),
        "num_raw_rows": len(raw_rows),
        "num_attribute_aggregated_rows": len(agg_rows),
        "num_selected_rows": len(selected_rows),
        "attributes_aggregated": True,
        "latents_kept_separate": True,
        "directions_combined": bool(args.combine_directions),
        "selection_metric": args.selection_metric,
        "target_norm_eps": args.target_norm_eps,
        "metrics": metrics,
        "combined_csv": str(combined_path),
        "attribute_aggregated_by_latent_strength_csv": str(agg_path),
        "selected_strength_by_latent_csv": str(selected_path),
        "markdown_report": str(markdown_path),
        "plots": plots,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
