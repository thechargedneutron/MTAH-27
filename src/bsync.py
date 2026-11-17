"""Fixed broadcast lag (tactical frame = broadcast frame + lag) for tactical frames a..b: the lag where Metrica's
tactical and broadcast balls agree best (median distance). One lag per half; bmap.py refines it piecewise."""
import os, sys, numpy as np
import evidence as T; import bcast
def lag(a, b, lo=0, hi=60000, step=4):
    BB = bcast.load()["B"]; fs = np.arange(a, b, 25)
    def cost(l):
        bi = fs - l; ok = (bi >= 0) & (bi < len(BB)); d = np.linalg.norm((T.B[fs[ok]] - BB[bi[ok]]) * [T.L, T.W], axis=1); d = d[~np.isnan(d)]
        return np.median(d) if len(d) > 100 else 99.0
    c = min(range(lo, hi, step), key=cost); fs = np.arange(a, b, 3)
    return min(range(c - step, c + step + 1), key=cost)
if __name__ == "__main__": print(lag(int(sys.argv[1]), int(sys.argv[2])))
