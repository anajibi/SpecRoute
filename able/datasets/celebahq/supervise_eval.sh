#!/bin/bash
# Unattended runner for the graph-A ladder evaluation.
#
# Each model is restartable because run_eval.py writes one npz per block of sources and skips
# blocks already present. A crash or an OOM therefore costs at most one block, and re-invoking
# the same command resumes. This loop does that automatically.
#
# On an OOM specifically the batch size is halved before the retry, because a retry at the same
# batch will fail the same way. Floor of 4.
set -u
R=/home/exouser/SpecRoute
S=/tmp/claude-1001/-home-exouser/9f41f3b2-f613-45aa-80de-f57861cf1804/scratchpad
OUT=$R/outputs/eval_graphA
LOG=$S/eval_supervisor.log
BATCH=${BATCH:-24}
MAX_RETRY=${MAX_RETRY:-6}

say () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

run_model () {   # run_model <tag> <config> <guidance-json-file>
  local tag=$1 cfg=$2 gfile=$3 batch=$BATCH try=0
  while [ $try -le $MAX_RETRY ]; do
    say "$tag: attempt $((try+1)) at batch $batch"
    cd "$R"
    .venv/bin/python able/datasets/celebahq/run_eval.py \
        --config "$cfg" --ckpt "$R/experiments/hdae/outputs/eval_ckpts/$tag.ckpt" \
        --tag "$tag" --guidance "$gfile" --out "$OUT" \
        --batch $batch --block 512 --save-256 1 \
        >> "$S/eval_$tag.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && grep -q "\[$tag\] DONE" "$S/eval_$tag.log"; then
      say "$tag: COMPLETE"; return 0
    fi
    if grep -qi "out of memory" "$S/eval_$tag.log"; then
      batch=$(( batch / 2 )); [ $batch -lt 4 ] && batch=4
      say "$tag: OOM -> retrying at batch $batch"
    else
      say "$tag: exited rc=$rc, resuming from last completed block"
    fi
    try=$((try+1))
    sleep 20
  done
  say "$tag: GAVE UP after $MAX_RETRY retries"
  return 1
}

say "supervisor up: batch $BATCH, up to $MAX_RETRY retries per model"
for m in k1 k5 k11; do
  cfgvar=CFG_$m; gvar=G_$m
  run_model "$m" "${!cfgvar}" "${!gvar}"
done
say "ALL MODELS DONE"
