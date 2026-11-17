#!/bin/bash
# Step 1 (GPU). RF-DETR detections for both halves in 7500-frame batches. We ran each half as one job on an NVIDIA GH200
# (about 3 h per half).  usage: pipeline/detect.sh [1|2]   (default: both halves)
set -e; source "$(dirname "$0")/env.sh"
batches() { $PYTHON -c "a0, b0 = $1, $2; print(' '.join(f'{a}:{min(a + 7499, b0)}' for a in range(a0, b0 + 1, 7500)))"; }
for h in ${1:-1 2}; do
  [ $h = 1 ] && R=$(batches 11185 90651) || R=$(batches 110808 190197)
  $PYTHON $ROOT/src/detect.py $MTAH_DATA/tactical/BAR-ATH_tactical.mp4 $MTAH_WORK/detections $R
done
