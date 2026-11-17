#!/bin/bash
# Step 5. Join every window's segment labels into one submission file.  usage: pipeline/assemble.sh OUT.csv
set -e; source "$(dirname "$0")/env.sh"
L=$MTAH_WORK/labels_all; mkdir -p $L; cp $MTAH_WORK/w*/labels/*_*.json $L/
$PYTHON $ROOT/src/assemble.py "$1" $L
