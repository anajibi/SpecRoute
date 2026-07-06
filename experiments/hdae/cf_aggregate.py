#!/usr/bin/env python
"""Aggregate CF consistency run CSVs into the canonical cf_table.csv."""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

NUMERIC = ["counterfactual_consistency", "factual_flip_success", "factual_flip_fail",
           "n_source", "n_success", "n_fail"]
KEYS = ["model", "attribute", "latent_used", "direction"]


def read_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in KEYS)].append(row)
    out = []
    for key, group in sorted(grouped.items()):
        rec = {k: v for k, v in zip(KEYS, key)}
        for col in NUMERIC:
            vals = [float(r[col]) for r in group if r.get(col) not in (None, "", "nan")]
            rec[col] = sum(vals) / len(vals) if vals else ""
        rec["num_runs"] = len(group)
        out.append(rec)
    return out


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = KEYS + NUMERIC + ["num_runs"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = aggregate_rows(read_rows([Path(x) for x in args.inputs]))
    write_csv(args.output, rows)
    Path(args.output).with_suffix(".json").write_text(json.dumps({"rows": len(rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
