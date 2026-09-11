#!/bin/bash
# One-shot archive of the FINISHED morpho tuning runs to s3, then prune the local copies
# that nothing downstream still needs.
#
# Unlike ckpt_janitor.sh this is not a daemon -- these runs are done, so there is no race
# with a trainer writing into the directory. It reuses the janitor's verification rule:
# nothing is deleted until its multipart ETag has been recomputed locally and matched
# against the object in s3. Size equality is not accepted as proof.
#
# Each run also gets its config, sweep json and per-sample npz uploaded alongside the
# weights, so a restored checkpoint arrives with the numbers it produced and the exact
# config that produced it -- a checkpoint on its own is not reproducible.
set -u
BUCKET=najibi-research-7f2a
PREFIX=hdae-handoff/morpho_tuning
REPO=/home/exouser/SpecRoute
PY=$REPO/.venv/bin/python
O=$REPO/experiments/hdae/outputs
C=$REPO/experiments/hdae/configs
SW=$O/morpho_sweep
COMMIT=$(cd $REPO && git rev-parse --short HEAD)
BRANCH=$(cd $REPO && git branch --show-current)

etag () {
  "$PY" -c "
import hashlib,sys
m=[]
fh=open(sys.argv[1],'rb')
while True:
    b=fh.read(8*1024*1024)
    if not b: break
    m.append(hashlib.md5(b).digest())
print(hashlib.md5(b''.join(m)).hexdigest()+'-'+str(len(m)) if len(m)>1 else hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())
" "$1"
}

put () {  # put <localfile> <key> ; upload if absent, then verify. rc=0 only on ETag match.
  local f=$1 key=$2 remote
  [ -e "$f" ] || { echo "  MISSING $f"; return 1; }
  remote=$(aws s3api head-object --bucket "$BUCKET" --key "$key" --query ETag --output text 2>/dev/null | tr -d '"')
  if [ -z "$remote" ]; then
    aws s3 cp "$f" "s3://$BUCKET/$key" --only-show-errors || { echo "  UPLOAD FAILED $key"; return 1; }
    remote=$(aws s3api head-object --bucket "$BUCKET" --key "$key" --query ETag --output text 2>/dev/null | tr -d '"')
  fi
  if [ "$(etag "$f")" = "$remote" ]; then echo "  verified  $key"; return 0
  else echo "  ETAG MISMATCH $key -- keeping local"; return 1; fi
}

# name:config:outputdir -- `base` is morpho_scale_100, which was verified identical to
# tune_base on every tuned knob and so serves as the comparison's reference run.
ROWS="
base:morpho_scale_100:morpho_scale_100
ed256:tune_ed256:tune_ed256
f0:tune_f0:tune_f0
f8:tune_f8:tune_f8
f32:tune_f32:tune_f32
mf100:tune_mf100:tune_mf100
"
# Losers that are fully evaluated and are not inputs to any remaining step. ed256 (current
# winner) and base (reference) stay local.
PRUNABLE=" f0 f8 f32 mf100 "

echo "commit $COMMIT on $BRANCH -> s3://$BUCKET/$PREFIX/"
for row in $ROWS; do
  name=${row%%:*}; rest=${row#*:}; cfg=${rest%%:*}; out=${rest#*:}
  echo "[$name]"
  ok=1
  for f in $O/$out/checkpoints/*.ckpt; do
    [ -e "$f" ] || continue
    put "$f" "$PREFIX/$name/checkpoints/$(basename "$f")" || ok=0
  done
  put "$C/$cfg.yaml" "$PREFIX/$name/config.yaml" || ok=0
  for f in $SW/sweep_tune_$name.json $SW/persample_tune_$name.npz; do
    [ -e "$f" ] && { put "$f" "$PREFIX/$name/$(basename "$f")" || ok=0; }
  done
  if [ "$ok" = 1 ] && [[ "$PRUNABLE" == *" $name "* ]]; then
    rm -f $O/$out/checkpoints/*.ckpt
    echo "  pruned local checkpoints (verified in s3)"
  fi
done

# The comparison's own outputs, so the conclusion travels with the weights.
echo "[analysis]"
for f in $O/tune_metrics.json $O/tune_bootstrap.json; do
  [ -e "$f" ] && put "$f" "$PREFIX/analysis/$(basename "$f")"
done
for f in $REPO/experiments/hdae/scripts/morpho_metrics.py \
         $REPO/experiments/hdae/scripts/tune_bootstrap.py \
         $REPO/experiments/hdae/scripts/cfg_sweep_morpho.py; do
  put "$f" "$PREFIX/analysis/$(basename "$f")"
done
echo "done. free: $(df -h /home/exouser | tail -1 | awk '{print $4}')"
