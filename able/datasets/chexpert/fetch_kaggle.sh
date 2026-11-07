#!/bin/bash
# CheXpert protocol v0, step 0: get the source archive.
#
# Source of record: https://www.kaggle.com/datasets/ashery/chexpert
# Needs a Kaggle API token at ~/.kaggle/kaggle.json (chmod 600), or
# KAGGLE_USERNAME / KAGGLE_KEY in the environment.
#
# Disk is the binding constraint on this box. The archive is ~11 GB and the
# uint8 image memmap is ~9.6 GB, so both cannot sit beside a full JPEG
# extraction. The plan this script sets up:
#   1. download the archive into its own empty directory (untrusted data),
#   2. back it up to S3 so the local copy is expendable,
#   3. build the memmap by streaming out of the archive (no extraction),
#   4. delete the local archive.
set -euo pipefail

DST=${1:-/home/exouser/able_data/chexpert_src}
KAG=/home/exouser/SpecRoute/.venv/bin/kaggle
LOG=/home/exouser/ladder/chexpert_fetch.log

need=23000000   # KiB, ~22 GB: archive + headroom for the memmap that follows
free=$(df --output=avail -k / | tail -1)
if [ "$free" -lt "$need" ]; then
  echo "refusing to start: ${free} KiB free on /, need ${need} KiB" >&2
  exit 1
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_KEY:-}" ]; then
  echo "no Kaggle credentials: put the token at ~/.kaggle/kaggle.json or export KAGGLE_USERNAME/KAGGLE_KEY" >&2
  exit 2
fi

mkdir -p "$DST" "$(dirname "$LOG")"
: > "$LOG"
echo "[$(date -Is)] free before: ${free} KiB" >>"$LOG"
echo "[$(date -Is)] download start -> $DST" >>"$LOG"
"$KAG" datasets download -d ashery/chexpert -p "$DST" >>"$LOG" 2>&1
echo "[$(date -Is)] download exit=$? " >>"$LOG"

cd "$DST"
for z in *.zip; do
  echo "[$(date -Is)] $z bytes=$(stat -c%s "$z")" >>"$LOG"
  sha256sum "$z" >>"$LOG"
  # manifest only -- the archive is NOT extracted here
  unzip -Z1 "$z" > "$DST/archive_filelist.txt" 2>>"$LOG" || true
  echo "[$(date -Is)] entries=$(wc -l < "$DST/archive_filelist.txt")" >>"$LOG"
done
echo "[$(date -Is)] free after: $(df --output=avail -k / | tail -1) KiB" >>"$LOG"
echo "[$(date -Is)] done" >>"$LOG"
