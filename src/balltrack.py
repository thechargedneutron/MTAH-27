"""One ball track from the full-res RF-DETR ball candidates (detect.py): candidates >= CONF whose pixel maps
(inverse homography) inside the pitch +- 4 m, then a Viterbi pass over frames: cost = -log-ish confidence, plus a jump
cost by distance in metres (the ball moves <= 1.4 m per frame = 35 m/s), with a 'no ball' state for gaps.
Writes work/det/balltrack_<a>_<b>.npy: [frame, x_m, y_m, px960, py960, conf] (NaN where no ball)."""
import os, sys, numpy as np
from paths import WORK
import evidence as T
CONF, K, VMAX, MISS = 0.25, 8, 1.4, 1.2

def to_pitch(f, px, py):
    q = np.linalg.inv(T.HM[f].astype(float)) @ np.array([px, py, 1.0]); return q[0] / q[2] * T.L, q[1] / q[2] * T.W

def build(a, b):
    d = np.load(f"{WORK}/det/dets_{a}_{b}.npz")["d"]; d = d[(d[:, 1] == 37) & (d[:, 6] >= CONF)]
    C = []
    for f in range(a, b + 1):
        r = d[d[:, 0] == f]; r = r[np.argsort(-r[:, 6])][:K]; c = []
        for x1, y1, x2, y2, cf in r[:, 2:7]:
            px, py = (x1 + x2) / 2, y2 - (y2 - y1) * 0.5; X, Y = to_pitch(f, px, py)
            if -4 < X < T.L + 4 and -4 < Y < T.W + 4: c.append((X, Y, px / 2, py / 2, cf))
        C.append(c)
    # Viterbi: state 0 = none, i+1 = candidate i. last real position carried through "none" for the jump test
    n = len(C); cost = [np.array([MISS] + [1 - c[4] for c in C[0]])]; back = []; lastpos = [[None] + [c[:2] for c in C[0]]]; gap = [[1] + [0] * len(C[0])]
    for t in range(1, n):
        prev = cost[-1]; cs = [np.inf]; bk = [0]; lp = [None]; gp = [0]
        j = int(np.argmin(prev)); cs[0] = prev[j] + MISS; bk[0] = j; lp[0] = lastpos[-1][j]; gp[0] = gap[-1][j] + 1
        for c in C[t]:
            best, bj = np.inf, 0
            for j in range(len(prev)):
                p = lastpos[-1][j]; g = gap[-1][j] + 1
                jump = 0.0 if p is None else max(0.0, np.hypot(c[0] - p[0], c[1] - p[1]) - VMAX * g) * 0.5
                v = prev[j] + (1 - c[4]) + jump
                if v < best: best, bj = v, j
            cs.append(best); bk.append(bj); lp.append(c[:2]); gp.append(0)
        cost.append(np.array(cs)); back.append(bk); lastpos.append(lp); gap.append(gp)
    s = int(np.argmin(cost[-1])); out = np.full((n, 6), np.nan); out[:, 0] = np.arange(a, b + 1)
    for t in range(n - 1, -1, -1):
        if s > 0: out[t, 1:] = C[t][s - 1]
        if t > 0: s = back[t - 1][s]
    return out

if __name__ == "__main__":
    a, b = map(int, os.environ.get("MTAH_WIN", "11185,14934").split(",")); o = build(a, b); np.save(f"{WORK}/det/balltrack_{a}_{b}.npy", o)
    v = ~np.isnan(o[:, 1]); print("ball found on", round(v.mean(), 3), "of frames")
    tb = ~np.isnan(T.B[a:b + 1, 0]); dm = np.hypot(o[:, 1] - T.B[a:b + 1, 0] * T.L, o[:, 2] - T.B[a:b + 1, 1] * T.W)
    ok = v & tb; print("vs Metrica ball (m): pct", np.nanpercentile(dm[ok], [25, 50, 75, 90]).round(1), " >5 m on", round((dm[ok] > 5).mean(), 3))
