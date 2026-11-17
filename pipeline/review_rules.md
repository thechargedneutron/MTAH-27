# Possession labeller (one 30-second piece of FC Barcelona vs Athletic Club, LaLiga)

You label every video frame of your piece (frames A..B, given in your task; 25 frames per second) with:
- `H` = Barcelona (home) in possession, ball alive
- `A` = Athletic Club (away) in possession, ball alive
- `D` = dead ball (ball out of play: out over a line, goal, foul/offside whistle, injury stop ... until the restart is taken)

## Rules of possession (from the organisers)
- While the ball is in play exactly one team is in possession. When the ball is contested or loose, possession STAYS with
  the last team that had it.
- Possession changes only when a player of the other team gains CONTROLLED possession (e.g. keeps the ball for a moment,
  or his team then passes it on). A touch without control, a deflection that does not intercept a pass, or an aerial
  duel won without the team then controlling the ball is NOT a change.
- After a stoppage play restarts with a clear possessor (the team taking the throw-in / free kick / corner / goal kick /
  kick-off); the dead ball (`D`) lasts from the moment the ball goes out of play / the whistle until the restart kick or
  throw.
- A possession is a whole run; flickering between H and A is heavily punished by the scoring. Prefer clean long runs and
  change the label only at real changes of control. Put each change at the frame where the new team gains control
  (or where the ball goes dead / is put back into play), as precisely as you can (within ~5 frames).

## Evidence (only these; do NOT open any other file, do NOT look for labels, scores or ground truth anywhere)
Run from `{WINDOW_DIR}` with `{PYTHON} evidence.py <cmd>`:
- `events A B` - a code-made list of candidate moments: the first frame a player of the other team stays at the ball,
  the ball outside the pitch, stretches where the TV broadcast shows a replay / close-up (it usually cuts to a replay
  right after a foul, a chance or the ball going out), slow-play stretches, set-piece tags. These are hints to check, not answers: proximity is not control, and many listed touches are not
  changes of possession.
- `summary A B [STEP]` - table: ball position in metres on a 105x68 pitch, ball speed m/s, distance of the nearest
  Barcelona (dH) and Athletic (dA) player to the ball, `src` = where the ball position comes from (`video` = our ball
  detector on the full-resolution video, best; `track` = Metrica's tracked ball where a ball is visible there; `TVtrack`
  = the ball tracked in the TV broadcast; `-` unknown), players tracked, mean player speed `plv`, `tv` = the TV
  broadcast shows live play (`live`) or a replay/close-up (`repl`), Smart-Tagging set pieces. The ball position is in the
  ground plane: a ball in the air looks displaced. The tracking has no player identities.
- `sheet A B N` - N frames of the wide tactical camera with tracked players (blue = Barcelona, red = Athletic, grey x =
  a track with nobody under it), the CYAN ring = best ball position (see `src`), and Metrica's tracked ball (yellow; ORANGE
  = no ball visible there, i.e. Metrica's ball is wrong), plus a zoom cut from the full-resolution (1920x1080) frame
  around the cyan ball. Open the printed image with the Read tool.
- `bview A B N` - the TV BROADCAST at the same N moments: a closer camera where touches, control, tackles, fouls and the
  referee are much easier to see. Cyan ring = the ball, on live views only. It cuts to replays / close-ups (labelled);
  a replay shows an earlier moment again - use it to understand what happened (e.g. the foul), never as live time. The
  TV picture can be up to ~1 s off in time: judge WHAT happens there, take exact frames from `sheet`/`summary`.
- `zoom F` - a larger full-resolution zoom around the ball at frame F; `zoomat F X Y` - the same around pixel X,Y of the
  960x540 sheet frame (sheet images are shown at 2/3 size: multiply what you read by 1.5). Barcelona wear dark blue/red (blaugrana) shirts; Athletic wear
  light/white shirts in this match (check on the zooms); referees yellow/black.
- `minimap A B N` - top-down pitch views with ball trail.
- `tags A B` - Metrica's automatic Smart Tagging phases (team in possession phases, set pieces); automatic, can be wrong.
Video frames exist only for frames {WA}-{WB}; you may look up to 250 frames before A for context.

## The tracking makes mistakes - trust the video
- Metrica's tracked ball is sometimes a "ghost" on empty grass (orange ring) while the real ball is elsewhere. The cyan
  ring and dH/dA use our own ball detector first, so they are usually right - but still check the picture when a
  decision depends on them.
- Team colours of the tracks were corrected from the shirt colours (the goalkeepers wear green; Barcelona's keeper
  was tracked as Athletic). Still check shirt colours in the zoom when a change depends on one player.

## Dead balls are easy to miss
Not every stoppage has the ball leaving the pitch: fouls, offsides, injuries and the referee's whistle stop play in the
middle of the field. Signs: the mean player speed (`plv` in `summary`) drops from ~3-5 m/s in play to ~1-1.5 m/s and
stays low; the ball sits still; players stand or walk; the TV cuts to a replay or a close-up of a player; the referee
signals; afterwards a set piece (free kick, throw-in, corner, goal kick) restarts play. Look at `sheet` and `bview` frames
in every period of low `plv` and every replay, and decide whether play was stopped. A stoppage is `D` from the moment the
ball goes out / play is stopped (usually where the players start to slow down, a second or so before `plv` bottoms
out) until the restart kick/throw.

## Control, not touches
A change of possession needs the new team to CONTROL the ball, but control can be brief: a player who receives or wins
the ball and then plays it on deliberately (a pass, a dribble, a shot, a controlled clearance) gives his team a
possession, even if it lasts only 1-3 s before they lose it again. Short possessions like that are real and must get
their own run. What does NOT count: a deflection, a block, a poke that goes straight to an opponent, or a 50/50 the
team does not come away with (roughly < 0.5 s on the ball). If play is stopped (foul) during the duel, the label goes
straight from the old team to `D`.
Timing: put the change at the moment the new team's player has the ball under control (his first controlled touch has
been taken, the ball is at his feet or played on by him), not when the opponent's pass/clearance leaves the foot and not
while the ball is still travelling to him. For a dead ball, D starts when the ball crosses the line / the whistle stops
play and ends at the restart touch.

## Method
1. Get the overview: `events A B`, `summary A B 10`, `tags A B`, a `minimap` and a `sheet` of the piece.
2. Find candidate changes: the `events` list, nearest team switching and staying switched, ball leaving the pitch (x<0, x>105, y<0, y>68 or
   ball vanishing near a line), set pieces, long gaps. Check each candidate with `summary` at STEP 1-2 and with
   `sheet`/`zoom` frames and `bview` (TV) frames around it - the TV view is the best place to see who controls the ball.
3. Sweep the whole piece with `sheet A B 12` (every ~2.5 s) and look for stoppages you have not explained, and for
   short possessions of the other team (a 2-4 s spell where they win the ball and play one or two passes is easy to miss
   between two long ones, especially where the tracked ball is not visible).
4. Decide the run list. Use your football understanding: a long pass that is intercepted changes possession at the
   interception; a pass in flight stays with the passing team; a throw-in / corner means the ball was dead before it.

## Output
Write `{WINDOW_DIR}/labels/<A>_<B>.json` (create the folder if needed):
`{"runs": [{"start": F0, "end": F1, "label": "H"|"A"|"D"}, ...], "notes": "short reasoning per change"}`
Runs must be contiguous, in order, and cover exactly A..B (first start = A, last end = B, each start = previous end + 1).
Save a first version early and refine it. Reply only "done" plus the run list.
