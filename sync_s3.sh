#!/bin/bash
# Sync this machine's work to/from s3://najibi-research-7f2a/wip-from-826a88e/
# Touches ONLY that prefix. Never writes to hdae-handoff/ or any other prefix.
set -euo pipefail

AWS=/home/exouser/.local/bin/aws
BUCKET=s3://najibi-research-7f2a/wip-from-826a88e
REPO=/home/exouser/SpecRoute
OUT="$REPO/experiments/hdae/outputs"

usage() { echo "usage: $0 {push|pull|status} [--dry-run]"; exit 1; }
[ $# -ge 1 ] || usage
CMD=$1; shift
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="--dryrun"

case "$CMD" in
  push)
    mkdir -p "$OUT"
    echo ">>> outputs/ -> $BUCKET/outputs/"
    $AWS s3 sync "$OUT" "$BUCKET/outputs/" $DRY
    if compgen -G "$REPO"/*.log > /dev/null; then
      echo ">>> *.log -> $BUCKET/logs/"
      for f in "$REPO"/*.log; do $AWS s3 cp "$f" "$BUCKET/logs/$(basename "$f")" $DRY; done
    fi
    ;;
  pull)
    mkdir -p "$OUT"
    echo ">>> $BUCKET/outputs/ -> outputs/"
    $AWS s3 sync "$BUCKET/outputs/" "$OUT" $DRY
    ;;
  status)
    echo ">>> remote:"
    $AWS s3 ls "$BUCKET/" --recursive --summarize | tail -20
    ;;
  *) usage ;;
esac
