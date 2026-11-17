# Team Possession Labelling for FC Barcelona vs Athletic Club

**Technical report.** More than a Hack 2027, Metrica Sports × FC Barcelona possession challenge.
Author: Kumar Ashutosh. Code: https://github.com/thechargedneutron/MTAH-27

## 1. Summary

We label every frame of the BAR–ATH tactical video as `H` (Barcelona in possession), `A` (Athletic Club in possession),
`D` (dead ball) or `X` (outside the two halves). The method has two stages:

1. **Signal processing.** Code turns the provided tracking and both videos into clean per-frame signals: a corrected
   ball position, corrected team labels and a time map between the broadcast and the tactical video.
2. **Possession decisions.** Each 30-second segment is labelled by a vision-language model (Claude Opus 5.5 [1]). The
   model applies written possession rules to those signals and to rendered video frames.

On the public 5 minutes, the submitted file scores **spellF1 0.560** and **macroF1 0.910**. The best provided baseline
scores 0.424 and 0.581 (Table 1).

## 2. Motivation

Possession is a judgement about *control*: a deliberate touch, not a deflection or a contested ball. Distance to the
ball cannot capture that [2]. The provided baselines assign the ball to the nearest tracked player and smooth the result.
They get many frames right (macroF1 ≈ 0.6), but they split possessions into fragments, and spellF1 penalises
fragmentation. Our own rule-based labellers, built on proximity, ball speed and simple control rules, stayed at
spellF1 ≤ 0.42 on the development data (§6).

The input data has two further problems:
- The provided ball position is missing, or sits on empty grass, for a large share of the match. It is tracked on about
  57% of frames.
- Some player tracks carry the wrong team.

We therefore rebuilt the ball and team signals from the video, and left the control decision to a model that can see
the touch in the image. We had no labelled football data that we could legally use to train a control classifier
(§5). So we used a pretrained model with written rules instead of a trained one.

## 3. Method

**3.1 Time alignment.** Tactical frame *f* is at time *f*/25 s.

For each half, we find the broadcast lag that minimises the median distance between the tactical and broadcast ball
tracks (`src/bsync.py`). The lag is 4014 frames in the first half and 26 920 in the second. `src/bmap.py` then refines
the lag piecewise, matching player positions every 10 frames. This is needed because the broadcast rejoins live play
after each replay. `bmap.py` also marks which broadcast frames show live play and which show replays.

**3.2 Detection.** We run RF-DETR Large [3], a DETR-family detector [4] with COCO-pretrained weights [5], on every
1920×1080 tactical frame (`src/detect.py`). Each frame is split into 3×2 tiles with 64 pixels of overlap (sliced
inference [6]), so that the small ball is still detected. We keep balls at confidence ≥ 0.05 and persons at ≥ 0.3. The
weights were not fine-tuned.

**3.3 Per-window signals.** The match is processed in 3750-frame windows. Positions are in pitch metres, mapped through
the provided homography [7].
- *Teams* (`kit.py`, `teamfix.py`):
  - We sample the shirt colour of each track just above its projected foot point.
  - The field kits are the median colours of each team's tracks.
  - Goalkeepers are colours far from both field kits. Each is assigned to the team defending that half.
  - A track with no person under it is marked as a ghost.
- *Ball* (`ballcheck.py`, `balltrack.py`, `ballfuse.py`):
  1. A Viterbi track [8] over the detector's ball boxes (confidence ≥ 0.25), limited to 35 m/s.
  2. A check of the provided ball against the image; it is kept only where a ball is visible.
  3. A fused position, taking the first available of:
     - detector chains seeded at confidence ≥ 0.70 and grown at ≥ 0.35;
     - the checked tactical ball;
     - the broadcast ball, moved to tactical time with the lag map;
     - the Viterbi track.
- *Candidate moments* (`evidence.py events`):
  - the other team staying at the ball;
  - the ball leaving the pitch;
  - broadcast replays;
  - slow play (low mean player speed);
  - Smart Tagging set pieces.

**3.4 Segment labelling.** Each window is cut into 750-frame (30 s) segments, giving 212 in total. Each segment is
labelled independently by Claude Opus 5.5, run through the Claude Code command-line tool
(`pipeline/review_window.sh`). The model receives the possession rules (`pipeline/review_rules.md`, §5). It can run one
program, `src/evidence.py`, which prints:
- signal tables: ball position and speed, nearest player of each team, ball source, mean player speed, live/replay;
- images: tactical frames with tracks and ball drawn, full-resolution zooms, the matching broadcast frames and a
  top-down pitch view.

It returns a list of contiguous runs covering the segment. It has no access to any other file, and in particular none
to the ground truth.

**3.5 Assembly.** The 212 segment files are joined into one file (`src/assemble.py`). Frames outside the two halves are
`X`. The halves are 11185–90651 and 110808–190197, as given in the bundle's `DATA.md`.

## 4. Assets used

| asset | use |
|---|---|
| tactical video | detection, shirt colours, frames for review |
| broadcast video | close view of touches and fouls; replays as a sign of stoppage |
| tactical tracking (ATD) | player positions and teams (after correction); ball where visible |
| broadcast tracking | time alignment (§3.1) and a fallback ball position |
| homography | pixel ↔ pitch metres |
| Smart Tagging | set-piece hints only (automatic, can be wrong) |
| RF-DETR Large, COCO weights (Apache-2.0) | ball and person detection |
| Claude Opus 5.5 (Anthropic, hosted) | possession decisions per segment |

## 5. Possession rules and training data

The rules follow the organisers' definitions and the event-based view of possession in [2]:
- While the ball is in play, exactly one team has possession.
- A **loose or contested ball stays** with the last team in possession.
- Possession **changes at the new team's first controlled touch**: the ball is at a player's feet or is deliberately
  played on. This includes short spells of 1–3 s.
- These are **not** changes:
  - deflections and blocks;
  - pokes that go straight to an opponent;
  - 50/50s that are not won (less than about 0.5 s on the ball).
- If play is stopped during a duel, the label goes from the old team straight to `D`.
- **Restarts.** `D` runs from the ball crossing the line, or the whistle, until the restart kick or throw. Play restarts
  with the team taking the set piece.
- **Stoppages without the ball leaving the pitch** (fouls, offsides, injuries) are recognised from:
  - players slowing down;
  - a still ball;
  - broadcast replays;
  - the set piece that follows.

The only match-specific facts in the rules are the two kit colours, read once from the video. No rule or constant is
tied to a particular frame.

**Training data.** No model was trained or fine-tuned for this work. We used two pretrained models as released:
- **RF-DETR.** Its weights were trained on COCO [5]: about 118 000 training images with human-annotated boxes for 80
  object classes, including *person* and *sports ball*.
- **Claude Opus 5.5.** A general-purpose model; its training data is not public.

No hand labels were made on the provided match.

**Use of the public ground truth.** The public ground truth (frames 11185–18684) was used only for evaluation, split
into two parts:
- a development part (11185–14934);
- a held-out part (14935–18684).

The rules were revised four times after reviewing errors on the development part. The held-out part was scored once for
each frozen version of the rules.

## 6. Results and limitations

**Table 1.** Public 5 minutes, official scorer.

| file | spellF1 | macroF1 |
|---|---|---|
| **submitted file**, whole public window | **0.560** | **0.910** |
| development part / held-out part | 0.620 / 0.518 | 0.867 / 0.875 |
| baseline smooth125 / smooth50 / smooth25 / raw | 0.424 / 0.326 / 0.287 / 0.038 | 0.581 / 0.621 / 0.620 / 0.597 |

On the held-out part, every frozen version of the rules scored spellF1 0.45–0.54. The best baseline scores 0.32 on the
held-out part alone. The differences between versions are within run-to-run variation.

**Tried and dropped** (scored on the development and held-out parts):
- *Rule-based labelling* (proximity plus ball speed): development spellF1 ≤ 0.42; held-out 0.38.
- *Coarse-to-fine review* (one look every 24 frames, refined at each change): development 0.73, but held-out 0.32. It
  missed stoppages and short spells.
- *Majority vote over 3–9 labelling runs* [9]: no gain beyond run-to-run variation, at three times the cost.
- *Snapping changes to detected ball contacts or to dead-ball moments*: no measurable gain.
- *Audio* (referee whistle; audio cross-correlation for broadcast alignment): no usable signal.

**Limitations.**
1. **Run-to-run variation.** Repeated labelling of the development part scored spellF1 0.60–0.73. Five minutes of
   ground truth (17 possessions) is too little for fine comparisons. The official scorer recommends a block bootstrap
   for this [10].
2. **Segment boundaries.** Segments are labelled independently, so labels can disagree where two segments meet.
3. **Timing of changes.** Almost all of the lost spellF1 comes from the timing of changes:
   - duels are called when the challenge starts, rather than when it is decided;
   - headers and aerial interceptions do not show in a 2-D ball position.
4. **Stoppage onsets** during play are estimated from slowing players.
5. **Cost.** Labelling the match used about 14.3 M model tokens and relies on a hosted model.

**Future work.**
- Overlapping segments, with their boundaries reconciled.
- More precise change timing from a ball-contact detector trained on labelled data from *other* matches.

## 7. Reproduction

- **Exact.** `./reproduce.sh` rebuilds `submission/BAR-ATH_possession.csv` **byte for byte** from the 212 stored segment
  labels in `labels/`, using only the Python standard library. Its sha256 begins `6336564c…`.
- **Full pipeline.** `README.md` lists the five steps from the raw data: detection, frame extraction, per-window
  signals, segment labelling and assembly.
  - Every step except segment labelling is deterministic. We checked that the refactored code reproduces the original
    per-window signals bit for bit.
  - Model sampling cannot be repeated exactly. A new labelling run gives a file of similar quality, not the same bytes.
    The stored segment labels are therefore the record of the submitted run.

## References

1. Anthropic. *Claude Opus 5.5* (model id `claude-opus-5-5`), 2026. https://www.anthropic.com/claude
2. D. Link, M. Hoernig. Individual ball possession in soccer. *PLOS ONE* 12(7): e0179953, 2017.
3. I. Robinson, P. Robicheaux, M. Popov, D. Ramanan, N. Peri. RF-DETR: Neural architecture search for real-time
   detection transformers. arXiv:2511.09554, 2025. https://github.com/roboflow/rf-detr (Apache-2.0)
4. N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, S. Zagoruyko. End-to-end object detection with
   transformers. *ECCV* 2020.
5. T.-Y. Lin, M. Maire, S. Belongie, et al. Microsoft COCO: Common objects in context. *ECCV* 2014.
6. F. C. Akyon, S. O. Altinuc, A. Temizel. Slicing aided hyper inference and fine-tuning for small object detection.
   *ICIP* 2022.
7. R. Hartley, A. Zisserman. *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge University Press, 2004.
8. A. J. Viterbi. Error bounds for convolutional codes and an asymptotically optimum decoding algorithm. *IEEE Trans.
   Information Theory* 13(2):260–269, 1967.
9. X. Wang, J. Wei, D. Schuurmans, et al. Self-consistency improves chain of thought reasoning in language models.
   *ICLR* 2023.
10. H. R. Künsch. The jackknife and the bootstrap for general stationary observations. *Annals of Statistics*
    17(3):1217–1241, 1989.
