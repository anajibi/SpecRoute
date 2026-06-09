#!/usr/bin/env bash
# Generate image grids from already-trained stage1.pt and stage2.pt checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
STEPS="${STEPS:-50}"
NUM_IMAGES="${NUM_IMAGES:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"

# Pass config stems as arguments to select runs. Defaults to the trained K=3 and K=5 runs.
if [[ "$#" -eq 0 ]]; then
  CONFIG_STEMS=(k3_levels k5_levels)
else
  CONFIG_STEMS=("$@")
fi

for STEM in "${CONFIG_STEMS[@]}"; do
  CONFIG="$ROOT/config/${STEM}.yaml"
  if [[ ! -f "$CONFIG" ]]; then
    echo "Missing config: $CONFIG" >&2
    exit 1
  fi
  echo "Generating visualizations for $STEM"
  python "$ROOT/scripts/visualize_hierarchy.py" \
    --config "$CONFIG" \
    --steps "$STEPS" \
    --num-images "$NUM_IMAGES" \
    --num-samples "$NUM_SAMPLES"
done
