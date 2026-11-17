#!/bin/bash
# Step 4. Label one window: each 750-frame segment is reviewed by Claude Opus 5.5 (model id claude-opus-5-5) through
# the Claude Code CLI, which follows rules.md and may only run evidence.py and view the images it writes.
# Two segments run at a time. Needs step 3 and a logged-in `claude` CLI.
# usage: pipeline/review_window.sh A   (or: pipeline/review_window.sh all)
set -e; source "$(dirname "$0")/env.sh"
if [ "$1" = all ]; then window_list | while read A B LAG; do "$0" $A; done; exit; fi
A=$1; W=$MTAH_WORK/w$A
B=$($PYTHON -c "import json; print([w['b'] for w in json.load(open('$WINDOWS')) if w['a'] == $A][0])")
segment() {
  local a=$1 b=$2
  (cd $W && claude -p "You are a football possession labeller. Read $W/rules.md and follow it exactly. Your piece: frames A=$a to B=$b. Use only the commands it names (run from $W with $PYTHON evidence.py ...) and look at the images with the Read tool. Do not open any other file or folder. Write your output file, then reply with \"done\" and the run list." \
     --model claude-opus-5-5 --allowedTools Read Write Bash > $W/labels/${a}_${b}.log)
  [ -s $W/labels/${a}_${b}.json ] || echo "segment $a-$b wrote no labels"
}
n=0
for ((a = A; a <= B; a += 750)); do
  b=$((a + 749)); ((b > B)) && b=$B
  segment $a $b & n=$((n + 1)); ((n % 2 == 0)) && wait
done; wait; echo "w$A labelled"
