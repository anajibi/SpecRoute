#!/bin/bash
# Upload every new checkpoint to s3 as it lands, VERIFY it, then prune old local copies.
#
# The control node has ~8.5 GB free and each checkpoint is 2.1 GB, so a 25-epoch extension
# saving every 6 epochs cannot keep them all on disk. Nothing is deleted until its multipart
# ETag has been recomputed locally and matched against the object in s3 -- size equality is
# not accepted as proof.
#
#   ckpt_janitor.sh <checkpoint-dir> <s3-prefix> [keep]
set -u
DIR="${1:?usage: ckpt_janitor.sh <dir> <s3-prefix> [keep]}"
PREFIX="${2:?}"
KEEP="${3:-2}"
BUCKET=najibi-research-7f2a
PY=/home/exouser/SpecRoute/.venv/bin/python

etag () {  # etag <file> ; prints multipart etag using the 8MB chunk the cli uses
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

while true; do
  for f in "$DIR"/epoch=*.ckpt; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    # skip anything still being written
    s1=$(stat -c%s "$f"); sleep 3; s2=$(stat -c%s "$f")
    [ "$s1" = "$s2" ] || continue
    key="$PREFIX/$b"
    remote=$(aws s3api head-object --bucket "$BUCKET" --key "$key" --query ETag --output text 2>/dev/null | tr -d '"')
    if [ -z "$remote" ]; then
      aws s3 cp "$f" "s3://$BUCKET/$key" --only-show-errors || continue
      remote=$(aws s3api head-object --bucket "$BUCKET" --key "$key" --query ETag --output text 2>/dev/null | tr -d '"')
    fi
    local_tag=$(etag "$f")
    if [ "$local_tag" = "$remote" ]; then
      echo "[$(date -u +%FT%TZ)] verified in s3: $b"
    else
      echo "[$(date -u +%FT%TZ)] ETAG MISMATCH, keeping local: $b (local $local_tag remote $remote)"
    fi
  done
  # prune: keep the newest $KEEP epoch checkpoints that are verified in s3
  mapfile -t all < <(ls -1t "$DIR"/epoch=*.ckpt 2>/dev/null)
  i=0
  for f in "${all[@]:-}"; do
    [ -e "$f" ] || continue
    i=$((i+1)); [ "$i" -le "$KEEP" ] && continue
    b=$(basename "$f")
    remote=$(aws s3api head-object --bucket "$BUCKET" --key "$PREFIX/$b" --query ETag --output text 2>/dev/null | tr -d '"')
    [ -n "$remote" ] && [ "$(etag "$f")" = "$remote" ] && { rm -f "$f"; echo "[$(date -u +%FT%TZ)] pruned (safe in s3): $b"; }
  done
  sleep 120
done
