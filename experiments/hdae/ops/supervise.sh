#!/bin/bash
# Generic ladder-run supervisor:  supervise.sh <config> <run> <target_step> <s3prefix>
#
# Continues from the newest checkpoint; never silently restarts from scratch. Runs in its own
# systemd scope (lingering on) so Claude-session teardown cannot kill it -- that killed k=1 p2 at
# step 36990 and left the GPU idle 27 h.
#
# Liveness via PID FILE, not `pgrep -f`: a -f pattern also matches the agent's own shell wrapper
# when its command line contains the pattern, which deadlocked an earlier supervisor into an
# endless sleep without ever launching the trainer.
#
# Refuses to auto-restart on a non-finite PARAMETER (real model failure) or 3 failures under
# 5 min. Non-finite GRADIENTS are ignored: the AMP scaler skips those and the run is fine.
set -u
CFG=$1; RUN=$2; TARGET=$3; S3=$4
REPO=/home/exouser/SpecRoute; S=/home/exouser/ladder
D=$REPO/experiments/hdae/outputs/$RUN/checkpoints
PIDF=$S/$RUN.pid; NAN=$S/${RUN}_nan.txt; TLOG=$S/${RUN}_train.log; LOG=$S/${RUN}_sup.log
cd $REPO; mkdir -p "$D"
log(){ echo "[$(date -u +%F_%T)] $*" >> $LOG; }
running(){ [ -f "$PIDF" ] && kill -0 "$(cat $PIDF 2>/dev/null)" 2>/dev/null; }
log "supervisor start pid $$ run=$RUN target=$TARGET cfg=$CFG"
fails=0
while true; do
  if running; then sleep 120; continue; fi
  # progress: prefer the checkpoint (authoritative), fall back to the tracer
  ck=$($REPO/.venv/bin/python -c "
import torch,glob,os
f='$D/last.ckpt'
print(torch.load(f,map_location='cpu')['global_step'] if os.path.exists(f) else 0)" 2>/dev/null)
  ck=${ck:-0}
  tr=$(grep -oE 'step=[0-9]+' $NAN 2>/dev/null|tail -1|cut -d= -f2); tr=${tr:-0}
  cur=$ck; [ "$tr" -gt "$cur" ] && cur=$tr
  if [ "$cur" -ge "$TARGET" ]; then log "DONE: step $cur >= $TARGET"; break; fi
  if [ "$(grep -c 'bad_param=[^N]' $NAN 2>/dev/null)" -gt 0 ]; then
    log "ABORT: non-finite PARAMETER -- needs a human decision, not a restart"; break; fi
  pgrep -f "ckpt_janitor[.]sh $D" >/dev/null || {
    setsid nohup bash $REPO/experiments/hdae/scripts/ckpt_janitor.sh "$D" "$S3" 0 \
      >> $S/${RUN}_janitor.log 2>&1 </dev/null & }
  log "launching from step $ck (attempt $((fails+1)))"
  t0=$(date +%s)
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HDAE_NAN_TRACE=$NAN HDAE_NAN_TRACE_ABORT=1 \
    $REPO/.venv/bin/python $REPO/experiments/hdae/scripts/train.py --config $CFG >> $TLOG 2>&1 &
  tp=$!; echo $tp > $PIDF; wait $tp; rc=$?; rm -f $PIDF
  dur=$(( $(date +%s)-t0 ))
  log "trainer exited rc=$rc after ${dur}s at step $(grep -oE 'step=[0-9]+' $NAN 2>/dev/null|tail -1|cut -d= -f2)"
  [ "$dur" -lt 300 ] && fails=$((fails+1)) || fails=0
  if [ "$fails" -ge 3 ]; then
    log "ABORT: 3 consecutive failures under 5 min each"
    grep -aE "Traceback|Error|out of memory|No space left" $TLOG|tail -5 >> $LOG; break; fi
  sleep 15
done
log "########## SUPERVISOR EXIT run=$RUN"
