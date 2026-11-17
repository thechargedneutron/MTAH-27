# Paths used by every pipeline step. Override any of them in the environment before running a step.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MTAH_DATA=${MTAH_DATA:-$ROOT/data}      # competition data bundle: tactical/, broadcast/
export MTAH_WORK=${MTAH_WORK:-$ROOT/work}      # intermediate files
PYTHON=${PYTHON:-python3}
FFMPEG=${FFMPEG:-ffmpeg}
WINDOWS=$ROOT/pipeline/windows.json
window_list() { $PYTHON -c "import json; [print(w['a'], w['b'], w['lag']) for w in json.load(open('$WINDOWS'))]"; }
