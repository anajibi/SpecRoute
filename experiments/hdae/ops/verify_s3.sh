#!/bin/bash
# verify_s3.sh <bucket> <key>  -> COMPLETE / TRUNCATED / MISSING
# Size equality is NOT proof of a good upload: a truncated multipart file size-matches perfectly.
# A torch .ckpt is a zip, so a complete archive must still carry its end-of-central-directory
# record in the final bytes. This reads the real tail back from S3 and looks for it.
set -u
B=$1; K=$2; T=$(mktemp)
info=$(aws s3api head-object --bucket "$B" --key "$K" --query '[ContentLength,ETag]' --output text 2>/dev/null) || { echo "MISSING $K"; rm -f $T; exit 1; }
aws s3api get-object --bucket "$B" --key "$K" --range "bytes=-32768" "$T" >/dev/null 2>&1
r=$(python3 -c "
import sys
d=open('$T','rb').read()
sys.stdout.write('COMPLETE' if d.rfind(b'PK\x05\x06')>=0 else 'TRUNCATED')")
echo "$r  $K  [$info]"
rm -f $T
[ "$r" = COMPLETE ]
