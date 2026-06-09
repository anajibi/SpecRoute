#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
for K in 3 5 8; do
  C="$ROOT/config/k${K}_levels.yaml"
  python "$ROOT/scripts/train_stage1_encoder_decoder.py" --config "$C"
  python "$ROOT/scripts/extract_latents.py" --config "$C"
  python "$ROOT/scripts/train_stage2_priors.py" --config "$C"
  python "$ROOT/scripts/run_preservation_probe.py" --config "$C"
  python "$ROOT/scripts/run_counterfactual_probe.py" --config "$C"
  python "$ROOT/scripts/visualize_hierarchy.py" --config "$C"
done
