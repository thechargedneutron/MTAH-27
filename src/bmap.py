"""Piecewise broadcast lag for tactical frames a..b: every 10 tactical frames, best broadcast frame within +-250 of the
current lag (Chamfer on players); accepted when cost < 1.2 m; lag per frame = nearest accepted sample (lags only change
at cuts). live = the broadcast frame at that lag matches the tactical players (cost < 2 m) and has a valid homography."""
import numpy as np, align_trk as A, bcast, os
from paths import WORK
BL0 = int(os.environ.get("MTAH_BLAG", 4014))                                  # fixed broadcast lag of this half (bsync.py)
BD = bcast.load(); a, b = map(int, os.environ.get("MTAH_WIN", "11185,14934").split(","))
samples = []
for f in range(a, b + 1, 10):
    bf = np.arange(f - BL0 - 250, f - BL0 + 250); bf = bf[(bf >= 0) & (bf < len(A.BP))]
    best = (None, 9e9)
    for x in bf[::2]:
        if not A.BV[x]: continue
        c = A.cost(x, np.array([f]))
        if c is not None and c[0] < best[1]: best = (x, c[0])
    if best[0] is not None and best[1] < 1.2:
        x0 = best[0]
        for x in (x0 - 1, x0 + 1):
            c = A.cost(x, np.array([f])) if A.BV[x] else None
            if c is not None and c[0] < best[1]: best = (x, c[0])
        samples.append((f, f - best[0], best[1]))
if not samples: samples = [(a, BL0, 9.0)]                                     # no TV frame matched anywhere (long stoppage): fixed lag of the half
S = np.array(samples); print("accepted samples", len(S), "lags:", np.unique(S[:, 1].astype(int), return_counts=True))
lag = np.interp(np.arange(a, b + 1), S[:, 0], S[:, 1]); 
# nearest-sample (not interpolated) lag, then median filter to kill single outliers
idx = np.abs(np.arange(a, b + 1)[:, None] - S[None, :, 0]).argmin(1); lag = S[idx, 1].astype(int)
from collections import Counter
lag2 = lag.copy()
for i in range(len(lag)): w = lag[max(0, i - 50):i + 51]; lag2[i] = Counter(w).most_common(1)[0][0]
live = np.zeros(len(lag2), bool)
for i, f in enumerate(range(a, b + 1)):
    x = f - lag2[i]
    if 0 <= x < len(A.BP) and A.BV[x]:
        c = A.cost(x, np.array([f])); live[i] = c is not None and c[0] < 2.0
np.savez(f"{WORK}/det/bmap_{a}_{b}.npz", lag=lag2, live=live, H=BD["H"][np.arange(a, b + 1) - lag2])
print("live", live.mean().round(3)); ch = np.where(np.diff(lag2) != 0)[0]; print("lag changes at", [(a + i + 1, int(lag2[i]), int(lag2[i + 1])) for i in ch])
