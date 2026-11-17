"""Kit colour of every tracked player from the video (torso patch above the projected foot point), every STEP frames.
Saves work/kit_<a>_<b>.npz: frames, rgb[frame, slot, 3] (NaN where untracked / off-frame)."""
import sys, os, numpy as np
from paths import WORK
from PIL import Image
import evidence as T

def ppm(f, x, y):
    """pixels per metre (960 px frame) at normalised pitch point x,y, along the pitch's x axis."""
    a = T.proj(f, [x, y])[0]; b = T.proj(f, [x + 1 / T.L, y])[0]; return float(np.hypot(*(a - b)))

def sample(a, b, step=5):
    fs = list(range(a, b + 1, step)); S = T.P.shape[1]; rgb = np.full((len(fs), S, 3), np.nan, np.float32); fg = np.full((len(fs), S), np.nan, np.float32)
    for k, f in enumerate(fs):
        im = np.asarray(Image.open(f"{T.FRAMES}/f{f:06d}.jpg").convert("RGB"), np.float32)
        for s in range(S):
            if np.isnan(T.P[f, s, 0]): continue
            x, y = T.proj(f, T.P[f, s])[0]; h = 1.8 * ppm(f, *T.P[f, s])          # body height in px (approx)
            x0, x1 = int(x - 0.12 * h), int(x + 0.12 * h) + 1; y0, y1 = int(y - 0.75 * h), int(y - 0.45 * h) + 1
            if x0 < 0 or y0 < 0 or x1 > 960 or y1 > 540 or x1 - x0 < 1 or y1 - y0 < 1: continue
            px = im[y0:y1, x0:x1].reshape(-1, 3)
            w = int(0.6 * h) + 2; bg = np.r_[im[y0:y1, max(0, x0 - w):x0].reshape(-1, 3), im[y0:y1, x1:min(960, x1 + w)].reshape(-1, 3)]
            n0 = len(px)
            if len(bg): px = px[np.linalg.norm(px - np.median(bg, 0), axis=1) > 35]       # drop pixels that look like the local grass
            fg[k, s] = len(px) / n0
            if len(px) >= 2: rgb[k, s] = np.median(px, 0)
    return np.array(fs), rgb, fg

if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2]); step = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    fs, rgb, fg = sample(a, b, step)
    np.savez(f"{WORK}/kit_{a}_{b}.npz", frames=fs, rgb=rgb, fg=fg)
    print(fs.shape, np.isfinite(rgb[..., 0]).mean())
