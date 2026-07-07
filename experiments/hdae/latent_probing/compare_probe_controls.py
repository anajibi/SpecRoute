#!/usr/bin/env python
"""Combine probe metrics from capacity/resolution control runs into tables."""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))


def read_metrics(config_name, path):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out = dict(row)
            out["config"] = config_name
            out["level"] = int(out["level"])
            for key in ("val_balanced_accuracy", "test_balanced_accuracy"):
                if key in out and out[key] != "":
                    out[key] = float(out[key])
            yield out


def write_long(rows, path):
    preferred = ["config", "level", "latent_key", "attribute_index", "attribute_name",
                 "probe_type", "val_balanced_accuracy", "test_balanced_accuracy"]
    fieldnames = preferred + sorted({k for r in rows for k in r.keys()} - set(preferred))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def write_wide(rows, path, metric):
    attrs = sorted({r["attribute_name"] for r in rows})
    cols = sorted({(r["config"], int(r["level"])) for r in rows})
    by_key = {(r["attribute_name"], r["config"], int(r["level"])): r.get(metric, "") for r in rows}
    headers = ["attribute_name", *[f"{cfg}_level{level}_{metric}" for cfg, level in cols]]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for attr in attrs:
            writer.writerow([attr, *[by_key.get((attr, cfg, level), "") for cfg, level in cols]])


def summarize(rows, metric):
    by_config_attr = defaultdict(list)
    by_config_level = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value == "" or value is None:
            continue
        value = float(value)
        by_config_attr[(row["config"], row["attribute_name"])].append((value, int(row["level"])))
        by_config_level[(row["config"], int(row["level"]))].append(value)
    best = []
    for (config, attr), vals in sorted(by_config_attr.items()):
        value, level = max(vals, key=lambda x: x[0])
        best.append({"config": config, "attribute_name": attr, "best_level": level, metric: value})
    level_means = [
        {"config": config, "level": level, f"mean_{metric}": sum(vals) / len(vals), "num_attributes": len(vals)}
        for (config, level), vals in sorted(by_config_level.items())
    ]
    return {"best_level_by_attribute": best, "mean_metric_by_level": level_means}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default="test_balanced_accuracy")
    p.add_argument("--output-dir", required=True)
    p.add_argument("metrics", nargs="+", help="NAME=path/to/probe_metrics.csv entries")
    args = p.parse_args()
    rows = []
    for item in args.metrics:
        if "=" not in item:
            raise ValueError(f"metric input {item!r} must be NAME=CSV")
        name, path = item.split("=", 1)
        rows.extend(read_metrics(name, path))
    if not rows:
        raise ValueError("no probe metrics rows found")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    long_path = out / "probe_control_long.csv"
    wide_path = out / "probe_control_wide.csv"
    write_long(rows, long_path)
    write_wide(rows, wide_path, args.metric)
    summary = summarize(rows, args.metric)
    summary.update({"metric": args.metric, "num_rows": len(rows), "long_csv": str(long_path), "wide_csv": str(wide_path)})
    (out / "probe_control_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {long_path} and {wide_path}")


if __name__ == "__main__":
    main()
