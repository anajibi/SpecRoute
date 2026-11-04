#!/bin/bash
# Watches the whole graph-B sequence. Silent unless actionable; reports each stage transition.
set -u
S=/home/exouser/ladder; cd /home/exouser/SpecRoute
seen=""
while true; do
  for r in celebahq256_b_k1_p1 celebahq256_b_k1_p2 celebahq256_b_k11_p1 celebahq256_b_k11_p2; do
    NAN=$S/${r}_nan.txt
    [ -f "$NAN" ] || continue
    if [ "$(grep -c 'bad_param=[^N]' $NAN 2>/dev/null)" -gt 0 ]; then
      echo "ACTIONABLE: non-finite PARAMETER in $r -- sequence will stop"; grep -a 'FIRST NON-FINITE' $NAN|tail -1; exit 1; fi
  done
  if grep -qa "ABORTED\|exceeded 55 h\|without DONE" $S/seqB.log 2>/dev/null; then
    echo "ACTIONABLE: graph B sequence stopped"; tail -4 $S/seqB.log; exit 2; fi
  free=$(df -BG --output=avail /home/exouser|tail -1|tr -dc 0-9)
  if [ "${free:-9}" -lt 3 ]; then echo "ACTIONABLE: disk ${free}GB during graph B"; exit 3; fi
  cur=$(grep -a "=== starting\|DONE\|COMPLETE" $S/seqB.log 2>/dev/null | tail -1)
  if [ "$cur" != "$seen" ]; then echo "stage: ${cur#*] }"; seen="$cur"; fi
  grep -qa "sequence COMPLETE" $S/seqB.log 2>/dev/null && { echo "DONE: graph B complete -- k=1 and k=11, both phases"; exit 0; }
  sleep 300
done
