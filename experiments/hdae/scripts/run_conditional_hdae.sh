#!/usr/bin/env bash
set -euo pipefail

# One-command runner for the conditional per-block HDAE configs.
# Usage:
#   bash experiments/hdae/scripts/run_conditional_hdae.sh experiments/hdae/configs/hier_k5.yaml
# Optional env vars:
#   PYTHON=python, FORCE=1, SKIP_PREPROCESS=1, SKIP_TRAIN=1, CKPT=/path/to/last.ckpt,
#   ATTRIBUTES=Smiling,Eyeglasses,Male,Young, NUM_IMAGES=256, STRENGTHS=0,0.5,1,2,4

CONFIG=${1:-experiments/hdae/configs/hier_k5.yaml}
PYTHON=${PYTHON:-python}
FORCE=${FORCE:-0}
SKIP_PREPROCESS=${SKIP_PREPROCESS:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
CKPT=${CKPT:-}
ATTRIBUTES=${ATTRIBUTES:-Smiling,Eyeglasses,Male,Young}
NUM_IMAGES=${NUM_IMAGES:-256}
STRENGTHS=${STRENGTHS:-0,0.5,1,2,4}

read_yaml() {
  "$PYTHON" - "$CONFIG" "$1" <<'PY'
import sys, yaml
path, dotted = sys.argv[1], sys.argv[2]
obj = yaml.safe_load(open(path))
for part in dotted.split('.'):
    obj = obj[part]
print(obj)
PY
}

OUT_DIR=$(read_yaml output_dir)
if [[ -z "$CKPT" ]]; then
  CKPT="$OUT_DIR/checkpoints/last.ckpt"
fi
ATTR_CKPT="$OUT_DIR/counterfactuals/attr_classifier.pt"
LATENTS="$OUT_DIR/latent_probing/latents.npz"
PROBES="$OUT_DIR/latent_probing/probes"
PROBE_METRICS="$PROBES/probe_metrics.csv"

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  "$PYTHON" experiments/hdae/scripts/preprocess_data.py --config "$CONFIG"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  "$PYTHON" experiments/hdae/scripts/train.py --config "$CONFIG"
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  echo "Set CKPT=/path/to/checkpoint or run without SKIP_TRAIN=1." >&2
  exit 1
fi

"$PYTHON" experiments/hdae/scripts/reconstruct.py --config "$CONFIG" --ckpt "$CKPT"
"$PYTHON" experiments/hdae/latent_probing/extract_latents.py --config "$CONFIG" --ckpt "$CKPT" --output "$LATENTS"
"$PYTHON" experiments/hdae/latent_probing/train_linear_probes.py --latents "$LATENTS" --output-dir "$PROBES"
"$PYTHON" experiments/hdae/counterfactuals/train_attr_classifier.py --config "$CONFIG" --output "$ATTR_CKPT"
"$PYTHON" experiments/hdae/counterfactuals/run_preservation_sweep.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --probe-metrics "$PROBE_METRICS" \
  --probe-weights-dir "$PROBES/weights" \
  --attr-classifier "$ATTR_CKPT" \
  --attributes "$ATTRIBUTES" \
  --strengths "$STRENGTHS" \
  --num-images "$NUM_IMAGES" \
  --output-dir "$OUT_DIR/counterfactuals/preservation_sweep"

echo "Done. Main outputs under: $OUT_DIR"
