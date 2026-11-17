#!/bin/bash
# Rebuild the submitted file from the stored segment labels in labels/ and check it is identical to submission/.
# Needs Python 3 only (standard library).  usage: ./reproduce.sh [OUT.csv]
set -e; cd "$(dirname "$0")"; OUT=${1:-work/BAR-ATH_possession.csv}; mkdir -p "$(dirname "$OUT")"
python3 src/assemble.py "$OUT" labels
cmp "$OUT" submission/BAR-ATH_possession.csv && echo "identical to submission/BAR-ATH_possession.csv"
sha256sum "$OUT"
