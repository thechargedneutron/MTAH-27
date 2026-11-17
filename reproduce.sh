#!/bin/bash
# Rebuild the submitted file from the stored segment labels in labels/ and check it is identical to submission/.
# Needs Python 3 only (standard library).  usage: ./reproduce.sh [OUT.csv]
set -e; cd "$(dirname "$0")"; OUT=${1:-work/Pikash07.csv}; mkdir -p "$(dirname "$OUT")"
python3 src/assemble.py "$OUT" labels
cmp "$OUT" submission/Pikash07.csv && echo "identical to submission/Pikash07.csv"
sha256sum "$OUT"
