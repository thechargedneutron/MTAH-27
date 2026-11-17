"""Broadcast -> tactical frame mapping from the tracking: for a broadcast frame, the tactical frame whose players (same
team) sit closest to the broadcast players (mean nearest-neighbour distance in metres, Chamfer)."""
import os, sys, numpy as np
import evidence as T
import bcast
BD = bcast.load(); BP, BT, BV = BD["P"], BD["team"], BD["valid"]
L, W = 105.3, 68.0
def cost(b, tf):
    """b: broadcast frame; tf: array of tactical frames -> Chamfer cost (m) per tactical frame."""
    m = ~np.isnan(BP[b, :, 0])
    if m.sum() < 5: return None
    out = np.zeros(len(tf))
    for t in (0, 1):
        q = BP[b, m & (BT == t)]
        if len(q) == 0: continue
        P = T.P[tf][:, T.TEAM == t]                                   # (n, k, 2)
        d = np.hypot((P[:, None, :, 0] - q[None, :, None, 0]) * L, (P[:, None, :, 1] - q[None, :, None, 1]) * W)
        out += np.nanmin(np.where(np.isnan(d), 99, d), 2).sum(1)
    return out / m.sum()
if __name__ == "__main__":
    tf = np.arange(0, len(T.P), 5); res = []
    for b in range(0, len(BP), 1000):
        if not BV[b]: continue
        c = cost(b, tf)
        if c is None: continue
        k = int(np.argmin(c)); s = np.sort(c)
        res.append((b, int(tf[k]), round(float(c[k]), 2), round(float(np.median(c)), 2)))
    for r in res: print(r, "lag", r[1] - r[0])
