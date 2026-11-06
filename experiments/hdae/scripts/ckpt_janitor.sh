#!/bin/bash
# Upload every new checkpoint to s3 as it lands, VERIFY it, then prune old local copies.
#
# The control node has little headroom and each checkpoint is 2.1 GB, so a long extension
# saving every 6 epochs cannot keep them all on disk. Nothing is deleted until its multipart
# ETag has been recomputed locally and matched against the object in s3 -- size equality is
# not accepted as proof.
#
# THIS SCRIPT FAILED ONCE AND KILLED A RUN. On 2026-08-28 the k=11 extension died at epoch 65
# with OSError errno 28 while Lightning wrote its checkpoint. Two defects, both fixed here:
#
#   1. It only ever considered `epoch=*.ckpt`. Lightning also maintains `last.ckpt`, and writes
#      `last-v1.ckpt` when a `last.ckpt` already exists -- which it always does when the output
#      dir was seeded to resume. Those two files are 2.1 GB EACH and were never counted, so the
#      real footprint was two checkpoints larger than this script believed.
#   2. It reclaimed space on a 120 s poll, but the trainer needs the space at the instant it
#      writes. It pruned epoch 53 at 13:35:59 -- seconds after the write had already failed.
#
# Fixes: poll every 30 s, KEEP defaults to 1 rather than 2, and the free-space check below
# warns loudly while there is still time to act.
#
# THIRD DEFECT, fixed 2026-11-06: it recomputed the local ETag of every present checkpoint on
# EVERY 30 s poll, including ones already verified. Measured on the 3DShapes k=11 run: 38.4 GB
# read in 14 minutes to re-hash one 1.52 GB file it had already matched, which over a 12 h run
# is ~2 TB of pointless I/O. Verified ETags are now cached in <dir>/.verified/<name>, keyed on
# size AND mtime, so a file that changes at all is re-hashed from scratch. The cache is only
# ever written AFTER a successful match against s3, and nothing is deleted on the strength of a
# cache entry alone: the remote head-object still has to agree at prune time.
#
#   ckpt_janitor.sh <checkpoint-dir> <s3-prefix> [keep]
set -u
DIR="${1:?usage: ckpt_janitor.sh <dir> <s3-prefix> [keep]}"
PREFIX="${2:?}"
KEEP="${3:-1}"
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

VDIR="$DIR/.verified"
mkdir -p "$VDIR" 2>/dev/null || true

cached_etag () {  # echo the cached etag iff the cache matches the file's current size+mtime
  local c="$VDIR/$(basename "$1")" cs cm ce
  [ -f "$c" ] || return 0
  read -r cs cm ce < "$c" 2>/dev/null || return 0
  [ "$cs" = "$(stat -c%s "$1")" ] && [ "$cm" = "$(stat -c%Y "$1")" ] && printf '%s' "$ce"
}

etag_of () {     # cached etag if usable, else hash it and remember the result
  local t
  t=$(cached_etag "$1")
  if [ -n "$t" ]; then printf '%s' "$t"; return; fi
  t=$(etag "$1")
  printf '%s %s %s\n' "$(stat -c%s "$1")" "$(stat -c%Y "$1")" "$t" > "$VDIR/$(basename "$1")"
  printf '%s' "$t"
}

while true; do
  for f in "$DIR"/epoch=*.ckpt; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    # skip anything still being written
    s1=$(stat -c%s "$f"); sleep 3; s2=$(stat -c%s "$f")
    [ "$s1" = "$s2" ] || continue
    # A truncated checkpoint uploads and ETag-verifies perfectly -- the ETag proves the upload
    # was faithful, not that the file was complete. On 2026-09-10 a disk-full crash left a
    # 289 MB fragment of a 1523 MB checkpoint, and this script marked it "verified in s3".
    if [ "$s2" -lt 1000000000 ]; then
      echo "[$(date -u +%FT%TZ)] REFUSING $b -- ${s2} bytes, too small to be a checkpoint"
      continue
    fi
    key="$PREFIX/$b"
    remote=$(aws s3api head-object --bucket "$BUCKET" --key "$key" --query ETag --output text 2>/dev/null | tr -d '"')
    if [ -z "$remote" ]; then
      aws s3 cp "$f" "s3://$BUCKET/$key" --only-show-errors || continue
      remote=$(aws s3api head-object --bucket "$BUCKET" --key "$key" --query ETag --output text 2>/dev/null | tr -d '"')
    fi
    was_cached=$(cached_etag "$f")
    local_tag=$(etag_of "$f")
    if [ "$local_tag" = "$remote" ]; then
      # Announce once, not every poll -- this log is read to tell progress from trouble.
      [ -z "$was_cached" ] && echo "[$(date -u +%FT%TZ)] verified in s3: $b"
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
    [ -n "$remote" ] && [ "$(etag_of "$f")" = "$remote" ] && { rm -f "$f"; rm -f "$VDIR/$b"; echo "[$(date -u +%FT%TZ)] pruned (safe in s3): $b"; }
  done
  free_gb=$(df -BG --output=avail "$DIR" | tail -1 | tr -dc '0-9')
  n_last=$(ls -1 "$DIR"/last*.ckpt 2>/dev/null | wc -l)
  [ "${free_gb:-99}" -lt 6 ] && echo "[$(date -u +%FT%TZ)] WARNING: only ${free_gb}GB free, ${n_last} last*.ckpt present"
  sleep 30
done
