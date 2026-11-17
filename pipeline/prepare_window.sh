#!/bin/bash
# Step 3. Per-window features (teams, ball, broadcast time map) and the window's review folder.
# Needs steps 1 and 2.  usage: pipeline/prepare_window.sh A B LAG   (or: pipeline/prepare_window.sh all)
set -e; source "$(dirname "$0")/env.sh"
if [ "$1" = all ]; then window_list | while read A B LAG; do "$0" $A $B $LAG; done; exit; fi
A=$1; B=$2; LAG=$3; W=$MTAH_WORK/w$A
mkdir -p $MTAH_WORK/det $W/labels $W/img
C=$(ls $MTAH_WORK/detections | awk -F_ -v a=$A -v b=$B '$1 <= a && $2 >= b')
$PYTHON -c "
import numpy as np
d = np.load('$MTAH_WORK/detections/$C/dets_$C.npz')['d']; m = (d[:, 0] >= $A) & (d[:, 0] <= $B)
np.savez_compressed('$MTAH_WORK/det/dets_${A}_${B}.npz', d=d[m])"
export MTAH_WIN=$A,$B MTAH_FRAMES=$W/frames MTAH_FULL=$W/tac MTAH_BCF=$W/bc MTAH_BLAG=$LAG MTAH_OUT=$W/img
cd $ROOT/src
for s in "kit.py $A $B" teamfix.py "ballcheck.py $A $B" balltrack.py ballfuse.py bmap.py; do $PYTHON $s > /dev/null; done
cat > $W/evidence.py <<X
"""evidence.py fixed to window $A-$B."""
import os, sys, runpy
os.environ.update(MTAH_WIN="$A,$B", MTAH_FRAMES="$W/frames", MTAH_FULL="$W/tac", MTAH_BCF="$W/bc", MTAH_BLAG="$LAG", MTAH_OUT="$W/img")
sys.path.insert(0, "$ROOT/src"); runpy.run_path("$ROOT/src/evidence.py", run_name="__main__")
X
sed -e "s#{WINDOW_DIR}#$W#g" -e "s#{PYTHON}#$PYTHON#g" -e "s#{WA}#$A#g" -e "s#{WB}#$B#g" $ROOT/pipeline/review_rules.md > $W/rules.md
for f in $MTAH_WORK/det/bmap_${A}_${B}.npz $MTAH_WORK/det/fused_${A}_${B}.npz $MTAH_WORK/teamfix_${A}_${B}.npy $MTAH_WORK/ballcheck_${A}_${B}.npy; do
  [ -f $f ] || { echo "w$A: missing $f"; exit 1; }; done
echo "w$A prepared"
