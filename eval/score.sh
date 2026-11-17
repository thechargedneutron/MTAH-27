#!/bin/bash
# Score submission files with the organisers' scorer against a ground-truth file (the public file, or one part of it).
# MTAH_EVAL = the evaluation/ folder of the competition repository.  usage: eval/score.sh GT.csv FILE.csv [FILE.csv ...]
set -e; GT=$(realpath "$1"); shift; D=$(mktemp -d); cp "$@" $D/
cd "${MTAH_EVAL:?set MTAH_EVAL to the competition repository's evaluation/ folder}"
${PYTHON:-python3} run_evaluation.py --results $D --gt "$GT" > $D/report.txt 2>&1
cut -d, -f2,5,6 scoreboard.csv | column -t -s,; rm -rf $D
