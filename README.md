# Team Possession Labelling: FC Barcelona vs Athletic Club

This is a submission to **More than a Hack 2027**, the Metrica Sports × FC Barcelona possession challenge.

**Author:** Kumar Ashutosh

## Deliverables

| deliverable | location |
|---|---|
| Possession file (199 319 rows, `frame,label`) | [`submission/BAR-ATH_possession.csv`](submission/BAR-ATH_possession.csv) |
| Technical report | [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) |
| Code that reproduces the file | this repository; see [Reproduction](#reproduction) |

On the public 5 minutes, the official scorer gives the file **spellF1 0.560** and **macroF1 0.910**. The best provided
baseline scores 0.424 and 0.581.

## Repository layout

```
submission/          the submitted possession file
labels/              212 segment label files (frames A..B as runs of H/A/D), the stored output of step 4
src/                 Python code
  paths.py             data and working directories (MTAH_DATA, MTAH_WORK)
  detect.py            RF-DETR ball and person detection (GPU)
  bsync.py bmap.py     broadcast ↔ tactical time alignment (bcast.py, align_trk.py: loaders and matching)
  kit.py teamfix.py    team of each track from shirt colours
  ballcheck.py balltrack.py ballfuse.py   ball position from detections, tracking and the image
  evidence.py          per-window signal tables and frame views used in segment labelling
  make_windows.py      cuts the halves into 3750-frame windows (writes pipeline/windows.json)
  assemble.py          joins segment labels into the submission file
pipeline/            shell scripts for steps 1–5, windows.json, review_rules.md (possession rules for step 4)
eval/                splits the public ground truth into development / held-out parts and runs the official scorer
reproduce.sh         exact rebuild of the submission file
```

The repository contains **no competition data**: no video, tracking, tagging, homography or ground truth.

## Reproduction

### Exact rebuild (Python 3, standard library only)

```bash
./reproduce.sh
```

This joins the 212 stored segment files in `labels/` and checks that the result is byte-identical to
`submission/BAR-ATH_possession.csv` (sha256 `6336564c58508a0031d73524aa54edb686f4d388ba5e6d03267b359d6be19bab`).

### Full pipeline from the raw data

**Setup.**
- Put the competition data bundle in `data/` (with `tactical/` and `broadcast/`), or set `MTAH_DATA`.
- Intermediate files go to `work/`, or to `MTAH_WORK`.
- Environments:
  - CPU steps: Python 3.12, `pip install -r requirements.txt`, and `ffmpeg`.
  - Step 1: Python 3.11, `pip install -r requirements-gpu.txt`, and a CUDA GPU.
  - Step 4: the [Claude Code](https://www.anthropic.com/claude-code) CLI, signed in with access to Claude Opus 5.5.
- Set `PYTHON` if the interpreter is not `python3`.

| step | command | output |
|---|---|---|
| 0 | `python src/make_windows.py` | `pipeline/windows.json`: 44 windows and the broadcast lag of each half (already included) |
| 1 | `pipeline/detect.sh` | RF-DETR detections for both halves (GPU; about 3 h per half on an NVIDIA GH200) |
| 2 | `pipeline/extract_frames.sh` | tactical and broadcast frame extracts per window |
| 3 | `pipeline/prepare_window.sh all` | per-window teams, ball and time map, plus a review folder for each window |
| 4 | `pipeline/review_window.sh all` | segment labels in `work/w<A>/labels/` |
| 5 | `pipeline/assemble.sh OUT.csv` | the possession file |

Steps 0–3 and 5 are deterministic. Step 4 uses a hosted language model whose sampling cannot be repeated exactly, so a
new run gives a file of similar quality, not the same bytes. This is why the labels of the submitted run are stored in
`labels/`.

### Scoring on the public ground truth

```bash
python eval/split_public_gt.py <competition repo>/evaluation/gt/BAR-ATH_possession_gt_public.csv eval/gt
MTAH_EVAL=<competition repo>/evaluation eval/score.sh eval/gt/gt_holdout.csv submission/BAR-ATH_possession.csv
```

## Licence

The code is released under the Apache License 2.0 (see `LICENSE`). The licence does not cover the competition data.
Third-party components:
- RF-DETR and its COCO weights (Apache-2.0), downloaded at run time.
- The Python packages listed in the requirements files, each under its own licence.
