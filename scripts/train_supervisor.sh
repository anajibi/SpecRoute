#!/bin/bash
# Keep one training run alive, restarting it if it dies. Runs ON the training node.
#
#   ./scripts/train_supervisor.sh <config-path> [max_restarts]
#
# train.py already auto-resumes from checkpoints/last.ckpt, so a restart continues rather
# than starting over.
#
# The duplicate-launch bug this is written to avoid: an earlier supervisor verified its
# launch with `PPid == 1`, but a process started with `setsid` from a STILL-RUNNING parent
# keeps that parent's pid until the parent exits. The check read "not running", so it
# launched a second copy -- two processes then trained the same config into the same output
# directory, clobbering each other's checkpoints. Here the liveness test never looks at
# PPid: it counts real `python` processes whose command line contains this exact config,
# which is true regardless of who started them.

set -u
CFG="${1:?usage: train_supervisor.sh <config-path> [max_restarts]}"
MAX_RESTARTS="${2:-20}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

LOG="$REPO/supervisor.log"
RUN_LOG="$REPO/train_run.log"
MIN_UPTIME=300          # a run that dies faster than this counts as a crash-loop
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"; }

# Count real python processes training THIS config -- MAIN processes only.
#
# Two traps here, both hit for real:
#  1. A shell whose command line merely CONTAINS the pattern matches `pgrep -f`. Requiring
#     comm == python excludes it.
#  2. DataLoader workers and torch.compile workers are forked children that inherit the
#     parent's exact command line, so a single training run shows up as ~37 matching python
#     processes. Counting those made this function report 37 and trip the "refusing to add
#     another" guard, which would have silently disabled auto-restart for the whole run.
#     A worker's parent is itself a matching python; the main process's parent is this
#     supervisor's shell. So: skip any process whose parent is also python.
running(){
  local n=0 p ppid
  for p in $(pgrep -f "train.py --config $CFG" 2>/dev/null); do
    [ -r "/proc/$p/comm" ] || continue
    [ "$(cat "/proc/$p/comm" 2>/dev/null)" = "python" ] || continue
    ppid=$(awk '/^PPid:/{print $2}' "/proc/$p/status" 2>/dev/null)
    if [ -r "/proc/$ppid/comm" ] && [ "$(cat "/proc/$ppid/comm" 2>/dev/null)" = "python" ]; then
      continue    # forked worker, not a separate run
    fi
    n=$((n+1))
  done
  echo $n
}

# Finished when global_step in last.ckpt has reached max_steps from the config.
finished(){
  "$REPO/.venv/bin/python" - "$CFG" <<'PY' 2>/dev/null
import os, sys, yaml, torch
cfg = yaml.safe_load(open(sys.argv[1]))
ck = os.path.join(cfg["output_dir"], "checkpoints", "last.ckpt")
if not os.path.exists(ck):
    sys.exit(1)
step = torch.load(ck, map_location="cpu").get("global_step", 0)
sys.exit(0 if step >= int(cfg["train"]["max_steps"]) else 1)
PY
}

log "supervisor up (pid $$) config=$CFG max_restarts=$MAX_RESTARTS"
restarts=0
while : ; do
  if finished; then log "max_steps reached -- training complete, supervisor exiting"; exit 0; fi

  n=$(running)
  if [ "$n" -gt 1 ]; then
    log "FATAL: $n processes already training this config -- refusing to add another"
    exit 1
  fi
  if [ "$n" -eq 1 ]; then sleep 120; continue; fi

  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    log "give up: $restarts restarts reached without completing"; exit 1
  fi

  log "no run alive -> starting (restart #$restarts)"
  start=$(date +%s)
  "$REPO/.venv/bin/python" experiments/hdae/scripts/train.py --config "$CFG" >> "$RUN_LOG" 2>&1
  rc=$?
  up=$(( $(date +%s) - start ))
  log "exited rc=$rc after ${up}s"

  if finished; then log "max_steps reached -- complete"; exit 0; fi
  if [ "$rc" -eq 0 ] && [ "$up" -gt "$MIN_UPTIME" ]; then
    log "clean exit before max_steps (rc=0, ${up}s) -- not a crash, stopping"; exit 0
  fi
  if [ "$up" -lt "$MIN_UPTIME" ]; then
    restarts=$((restarts+1))
    log "died after only ${up}s -- crash-loop guard, backing off 120s"
    sleep 120
  else
    restarts=$((restarts+1))
    sleep 20
  fi
done
