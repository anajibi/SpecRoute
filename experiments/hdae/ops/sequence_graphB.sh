#!/bin/bash
# Run graph B end to end with no idle gap between stages:
#   b_k1_p1 (31,250) -> b_k1_p2 (46,875) -> b_k11_p1 (31,250) -> b_k11_p2 (46,875)
#
# Completion is tested on the CHECKPOINT's global_step, never on the NaN tracer's last line.
# The tracer logs every 10 steps, so it prints 46870 and never 46875; a chain that waited on the
# tracer hung forever and left the GPU idle 53 h.
#
# Each stage runs through supervise.sh, which survives session teardown, auto-continues after an
# external kill, and refuses to restart on a non-finite parameter.
set -u
REPO=/home/exouser/SpecRoute; S=/home/exouser/ladder
cd $REPO
log(){ echo "[$(date -u +%F_%T)] $*" >> $S/seqB.log; }

stage(){ # cfg run target s3prefix
  local CFG=$1 RUN=$2 T=$3 PFX=$4
  local D=$REPO/experiments/hdae/outputs/$RUN/checkpoints
  # already finished?
  local gs=$($REPO/.venv/bin/python -c "
import torch,os
f='$D/last.ckpt'
print(torch.load(f,map_location='cpu')['global_step'] if os.path.exists(f) else 0)" 2>/dev/null)
  if [ "${gs:-0}" -ge "$T" ]; then log "$RUN already at ${gs} >= $T, skipping"; return 0; fi

  log "=== starting $RUN (target $T) ==="
  systemd-run --user --scope --unit=hdae-$RUN --slice=hdae.slice \
    bash $S/supervise.sh "$CFG" "$RUN" "$T" "$PFX" >/dev/null 2>&1 &
  # wait for the supervisor to finish this stage
  local waited=0
  while true; do
    sleep 120; waited=$((waited+120))
    if grep -qa "DONE: step" $S/${RUN}_sup.log 2>/dev/null; then log "$RUN DONE"; break; fi
    if grep -qa "ABORT" $S/${RUN}_sup.log 2>/dev/null; then
      log "$RUN ABORTED -- stopping the sequence, needs a human decision"; tail -3 $S/${RUN}_sup.log >> $S/seqB.log; return 2; fi
    if grep -qa "SUPERVISOR EXIT" $S/${RUN}_sup.log 2>/dev/null; then
      if grep -qa "DONE: step" $S/${RUN}_sup.log; then break; fi
      log "$RUN supervisor exited without DONE -- stopping"; return 3; fi
    [ $waited -gt 200000 ] && { log "$RUN exceeded 55 h, stopping"; return 4; }
  done

  # ship config, verify the final checkpoint, then free the local copy
  aws s3 cp "$CFG" "s3://najibi-research-7f2a/$PFX/$(basename $CFG)" --only-show-errors
  for f in $S/${RUN}_nan.txt $S/${RUN}_sup.log $S/${RUN}_train.log $S/${RUN}_janitor.log; do
    [ -f "$f" ] && gzip -c "$f" | aws s3 cp - "s3://najibi-research-7f2a/$PFX/logs/$(basename $f).gz" --only-show-errors
  done
  log "$RUN logs + config shipped"
  return 0
}

log "########## graph B sequence start"
stage experiments/hdae/configs/celebahq256_b_k1_p1.yaml  celebahq256_b_k1_p1  31250 hdae-handoff/celebahq256/b_k1_p1  || exit $?
stage experiments/hdae/configs/celebahq256_b_k1_p2.yaml  celebahq256_b_k1_p2  46875 hdae-handoff/celebahq256/b_k1_p2  || exit $?
# k=1 is complete here; its local checkpoints are verified in s3 by the janitor, free the space
rm -f $REPO/experiments/hdae/outputs/celebahq256_b_k1_p1/checkpoints/*.ckpt
log "freed b_k1_p1 local checkpoints, disk $(df -BG --output=avail /home/exouser|tail -1|tr -dc 0-9)GB"
stage experiments/hdae/configs/celebahq256_b_k11_p1.yaml celebahq256_b_k11_p1 31250 hdae-handoff/celebahq256/b_k11_p1 || exit $?
stage experiments/hdae/configs/celebahq256_b_k11_p2.yaml celebahq256_b_k11_p2 46875 hdae-handoff/celebahq256/b_k11_p2 || exit $?
rm -f $REPO/experiments/hdae/outputs/celebahq256_b_k11_p1/checkpoints/*.ckpt
# free the finished p2 locals as well -- every numbered checkpoint is already ETag-verified in s3
# by the janitor, and last.ckpt duplicates the final one. Leaving them pinned 2.7 GB each and
# squeezed the next rung's checkpoint writes.
rm -f $REPO/experiments/hdae/outputs/celebahq256_b_k1_p2/checkpoints/*.ckpt
rm -f $REPO/experiments/hdae/outputs/celebahq256_b_k11_p2/checkpoints/*.ckpt
log "########## graph B sequence COMPLETE (k=1 and k=11, both phases), disk $(df -BG --output=avail /home/exouser|tail -1|tr -dc 0-9)GB"
