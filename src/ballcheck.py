"""Is there a ball-like bright blob at the tracked ball's pixel? (960x540 frames; ball ~2-4 px). Score = brightest 3x3
mean within R px of the projected ball minus the median of a 31x31 neighbourhood. Returns array over frames (NaN = untracked)."""
import os, sys, numpy as np
from paths import WORK
from PIL import Image
import evidence as T
from scipy.ndimage import uniform_filter

def score(f, R=6):
    if np.isnan(T.B[f, 0]): return np.nan
    x, y = T.proj(f, T.B[f])[0]; x, y = int(round(x)), int(round(y))
    if not (16 <= x < 944 and 16 <= y < 524): return np.nan
    im = np.asarray(Image.open(f"{T.FRAMES}/f{f:06d}.jpg").convert("L"), np.float32)[y - 15:y + 16, x - 15:x + 16]
    sm = uniform_filter(im, 3); c = sm[15 - R:15 + R + 1, 15 - R:15 + R + 1]
    return float(c.max() - np.median(im))

if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2])
    s = np.array([score(f) for f in range(a, b + 1)])
    np.save(f"{WORK}/ballcheck_{a}_{b}.npy", s); print(np.nanpercentile(s, [5, 25, 50, 75, 95]))
