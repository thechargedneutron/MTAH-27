"""Ball evidence for frames a..b, every source in pitch metres:
  det_anchor : RF-DETR ball chains seeded from high-confidence boxes (>= HI), grown with >= LO boxes while physically
               reachable (<= 35 m/s); one ball at a time (longest chains win)
  det_vit    : Viterbi track over all >= 0.25 boxes (balltrack.py)
  met        : Metrica tactical ball, kept only where a ball is visible there (ballcheck)
  bc         : Metrica BROADCAST ball, mapped to tactical frames with the fixed lag (tactical = broadcast + 4014)
  attn       : where the players converge: least-squares intersection of the movement lines of running players,
               weighted by speed and by closeness to the play (no ball information at all)
  fused      : first available of det_anchor, met, bc, det_vit; attn only as a fallback hint (not in 'fused')."""
import os, sys, numpy as np
from paths import WORK
import evidence as T
import balltrack, bcast
L, W = T.L, T.W; WK = WORK; LAG = int(os.environ.get("MTAH_BLAG", 4014))   # broadcast lag of this half (bsync.py)
HI, LO, VMAX = 0.70, 0.35, 1.4

def det_rows(a, b):
    d = np.load(f"{WK}/det/dets_{a}_{b}.npz")["d"]; d = d[d[:, 1] == 37]; out = {}
    for r in d:
        f = int(r[0]); px, py = (r[2] + r[4]) / 2, (r[3] + r[5]) / 2; X, Y = balltrack.to_pitch(f, px, py)
        if -4 < X < L + 4 and -4 < Y < W + 4: out.setdefault(f, []).append((X, Y, float(r[6])))
    return out

def anchors(a, b):
    R = det_rows(a, b); n = b - a + 1; out = np.full((n, 3), np.nan)
    hi = sorted(((c, f, X, Y) for f, rs in R.items() for X, Y, c in rs if c >= HI), reverse=True)
    used = np.zeros(n, bool)
    for c, f, X, Y in hi:
        i = f - a
        if used[i]: continue
        out[i] = (X, Y, c); used[i] = True
        for step in (1, -1):                                   # grow both ways with relaxed boxes
            px, py, j, miss = X, Y, i, 0
            while 0 <= j + step < n and miss <= 5:
                j += step; cand = [r for r in R.get(a + j, []) if r[2] >= LO and np.hypot(r[0] - px, r[1] - py) <= VMAX * (miss + 1) + 0.5]
                if used[j] and not np.isnan(out[j, 0]): break
                if cand:
                    r = max(cand, key=lambda r: r[2]); out[j] = r; used[j] = True; px, py, miss = r[0], r[1], 0
                else: miss += 1
    return out

def attention(f, horizon=1.0):
    """players' movement lines -> least-squares meeting point (m). Players with speed > 1.5 m/s; weight speed^2 and
    a Gaussian on distance to the players' median position (focus on the play)."""
    if f < 5: return np.nan, np.nan
    p = T.P[f].astype(float) * [L, W]; q = T.P[f - 5].astype(float) * [L, W]; v = (p - q) * 5.0
    sp = np.hypot(v[:, 0], v[:, 1]); m = ~np.isnan(sp) & (sp > 1.5)
    if m.sum() < 4: return np.nan, np.nan
    p, v, sp = p[m], v[m], sp[m]; u = v / sp[:, None]; nrm = np.c_[-u[:, 1], u[:, 0]]
    c = np.median(p, 0); w = sp ** 2 * np.exp(-np.sum((p - c) ** 2, 1) / (2 * 20.0 ** 2))
    A = np.einsum("i,ij,ik->jk", w, nrm, nrm) + np.eye(2) * 1e-3 * w.sum(); bb = np.einsum("i,ij,ik,ik->j", w, nrm, nrm, p)
    x = np.linalg.solve(A, bb + 1e-3 * w.sum() * c)
    return (x[0], x[1]) if (-10 < x[0] < L + 10 and -10 < x[1] < W + 10) else (np.nan, np.nan)

def build(a, b):
    n = b - a + 1; S = {}
    S["det_anchor"] = anchors(a, b)[:, :2]
    S["det_vit"] = np.load(f"{WK}/det/balltrack_{a}_{b}.npy")[:, 1:3]
    met = T.B[a:b + 1].astype(float) * [L, W]; sc = np.load(f"{WK}/ballcheck_{a}_{b}.npy"); met[~(np.nan_to_num(sc) >= 15)] = np.nan; S["met"] = met
    BD = bcast.load(); bi = np.arange(a, b + 1) - LAG; bc = BD["B"][bi].astype(float) * [L, W]; S["bc"] = bc
    S["attn"] = np.array([attention(f) for f in range(a, b + 1)])
    # two other sources agreeing (< 3 m) overrule the detector when it is > 5 m away from both
    da = S["det_anchor"].copy()
    for i in range(n):
        o = [S[k][i] for k in ("met", "bc") if not np.isnan(S[k][i, 0])]
        if not np.isnan(da[i, 0]) and len(o) == 2 and np.hypot(*(o[0] - o[1])) < 3 and all(np.hypot(*(da[i] - y)) > 5 for y in o): da[i] = np.nan
    S["det_anchor"] = da
    fused = np.full((n, 2), np.nan); src = np.full(n, "", object)
    for k in ("det_anchor", "met", "bc", "det_vit"):
        m = np.isnan(fused[:, 0]) & ~np.isnan(S[k][:, 0]); fused[m] = S[k][m]; src[m] = k
    S["fused"] = fused; S["src"] = src; return S

if __name__ == "__main__":
    a, b = map(int, os.environ.get("MTAH_WIN", "11185,14934").split(",")); S = build(a, b); np.save(f"{WK}/det/ballfuse_{a}_{b}.npy", S, allow_pickle=True)
    np.savez(f"{WK}/det/fused_{a}_{b}.npz", xy=S["fused"], src=S["src"].astype(str))
    # reference: Metrica ball strongly visible AND det_vit agrees within 2 m
    sc = np.load(f"{WK}/ballcheck_{a}_{b}.npy"); met = T.B[a:b + 1].astype(float) * [L, W]
    ref = np.where(((np.nan_to_num(sc) >= 30) & (np.hypot(*(met - S["det_vit"]).T) < 2))[:, None], met, np.nan)
    print("reference frames", int((~np.isnan(ref[:, 0])).sum()))
    P = T.P[a:b + 1].astype(float) * [L, W]; cen = np.nanmean(P, 1)
    for k in ("det_anchor", "det_vit", "met", "bc", "fused", "attn"):
        x = S[k]; cov = (~np.isnan(x[:, 0])).mean(); e = np.hypot(*(x - ref).T); e = e[~np.isnan(e)]
        print(f"{k:11s} coverage {cov:.2f}  vs reference: median {np.median(e):.1f} m, within 3 m {np.mean(e < 3):.2f} (n={len(e)})")
    e = np.hypot(*(cen - ref).T); e = e[~np.isnan(e)]; print(f"{'centroid':11s} (all players' mean) median {np.median(e):.1f} m, within 3 m {np.mean(e < 3):.2f}")
    print("fused source mix", {k: int((S['src'] == k).sum()) for k in ('det_anchor', 'met', 'bc', 'det_vit', '')})
