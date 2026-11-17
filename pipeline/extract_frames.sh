#!/bin/bash
# Step 2. Frame extracts for every window: tactical frames (960x540, all frames), broadcast frames at the same moments
# (fixed lag of the half), and every 2nd tactical frame at full resolution.  usage: pipeline/extract_frames.sh
set -e; source "$(dirname "$0")/env.sh"
TV=$MTAH_DATA/tactical/BAR-ATH_tactical.mp4; BV=$MTAH_DATA/broadcast/BAR-ATH_broadcast.mp4
window_list | while read A B LAG; do
  W=$MTAH_WORK/w$A; n=$((B - A + 1)); mkdir -p $W/frames $W/bc $W/tac
  ts=$($PYTHON -c "print(f'{$A / 25:.2f}')"); bs=$($PYTHON -c "print(f'{($A - $LAG) / 25:.2f}')")
  $FFMPEG -nostdin -v error -ss $ts -i $TV -frames:v $n -vf scale=960:540 -q:v 4 -start_number $A $W/frames/f%06d.jpg
  $FFMPEG -nostdin -v error -ss $bs -i $BV -frames:v $n -vf scale=960:540 -q:v 5 -start_number $A $W/bc/f%06d.jpg
  $FFMPEG -nostdin -v error -ss $ts -i $TV -frames:v $(((n + 1) / 2)) -vf "select=not(mod(n\,2))" -vsync 0 -frame_pts 1 -q:v 4 $W/tac/p%06d.jpg
  echo "w$A $(ls $W/frames | wc -l) $(ls $W/bc | wc -l) $(ls $W/tac | wc -l)"
done
