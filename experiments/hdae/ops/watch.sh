#!/bin/bash
# watch.sh <run> <target> -- silent unless actionable. Exits with a diagnosis.
set -u
R=$1; T=$2; S=/home/exouser/ladder; cd /home/exouser/SpecRoute
NAN=$S/${R}_nan.txt; SUP=$S/${R}_sup.log; PIDF=$S/$R.pid
prev=""; tick=0
while true; do
  n=$(grep -oE 'step=[0-9]+' $NAN 2>/dev/null|tail -1|cut -d= -f2); n=${n:-0}
  if [ "$(grep -c 'bad_param=[^N]' $NAN 2>/dev/null)" -gt 0 ]; then
    echo "ACTIONABLE: non-finite PARAMETER at step $n in $R -- unrecoverable"; grep -a 'FIRST NON-FINITE' $NAN|tail -1; exit 1; fi
  if grep -qa 'ABORT' $SUP 2>/dev/null; then echo "ACTIONABLE: supervisor aborted $R"; tail -3 $SUP; exit 2; fi
  if grep -qa 'DONE: step' $SUP 2>/dev/null; then
    echo "DONE: $R reached $T. $(aws s3 ls s3://najibi-research-7f2a/hdae-handoff/celebahq256/${R#celebahq256_}/ 2>/dev/null|wc -l) ckpts in s3."; exit 0; fi
  free=$(df -BG --output=avail /home/exouser|tail -1|tr -dc 0-9)
  if [ "${free:-9}" -lt 3 ]; then echo "ACTIONABLE: disk ${free}GB during $R -- checkpoint write will fail"; exit 3; fi
  if [ -n "$prev" ] && [ "$n" = "$prev" ]; then tick=$((tick+1))
    if [ "$tick" -ge 4 ]; then echo "ACTIONABLE: $R stalled at step $n ~20 min, gpu $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"; exit 4; fi
  else tick=0; fi
  prev=$n; sleep 300
done
