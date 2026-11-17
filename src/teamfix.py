"""Per-frame team of every track slot from the kit colour (kit.py samples), plus ghost tracks (no person under the slot).
Kit classes: field kits = the colour medians of the two teams' slots (Metrica team labels as weak supervision); goalkeeper
kits = samples far from both field kits; a keeper belongs to the team that defends the goal in his half.
Returns team[f, s] in {0, 1, -1 (ghost)} for frames a..b, Metrica's team where there is no evidence."""
import os, sys, numpy as np
from paths import WORK
import evidence as T

def home_defends_left(f):
    """True when Barcelona defends x=0 at frame f: their players' mean x is below Athletic's (over +-10 s)."""
    a, b = max(0, f - 250), f + 250; X = T.P[a:b, :, 0]
    return np.nanmean(X[:, T.TEAM == 0]) < np.nanmean(X[:, T.TEAM == 1])

def fix(a, b, WIN=4, GHOST_FG=0.35, FAR=20, FIELD=False):
    z = np.load(f"{WORK}/kit_{a}_{b}.npz"); fs, rgb, fg = z["frames"], z["rgb"], z["fg"]
    S = rgb.shape[1]; ok = np.isfinite(rgb[..., 0]) & (fg >= GHOST_FG)
    cen = [np.median(rgb[ok & (T.TEAM[None] == t)], 0) for t in (0, 1)]
    dist = np.stack([np.linalg.norm(rgb - c, axis=-1) for c in cen], -1)          # [k, s, 2]
    cls = np.where(dist[..., 0] < dist[..., 1], 0, 1).astype(int)
    gk = ok & (rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2]) > FAR)              # green-dominant shirt = a keeper
    left = home_defends_left((a + b) // 2)
    for k, f in enumerate(fs):
        for s in np.where(gk[k])[0]:
            x = T.P[f, s, 0]; inleft = x < 0.5; cls[k, s] = 0 if inleft == left else 1
    if not FIELD: cls[~gk] = T.TEAM[None].repeat(len(fs), 0)[~gk]                  # trust Metrica's team for field kits
    cls[~ok] = -2                                                                  # no evidence
    ghost = np.isfinite(fg) & (fg < GHOST_FG)
    cls[ghost] = -1
    # smooth per slot: majority over +-WIN samples among evidence; ghost if majority ghost
    sm = np.full_like(cls, -2)
    for s in range(S):
        for k in range(len(fs)):
            w = cls[max(0, k - WIN):k + WIN + 1, s]; w = w[w != -2]
            if len(w): v, c = np.unique(w, return_counts=True); sm[k, s] = v[np.argmax(c)]
    team = np.repeat(T.TEAM[None].astype(int), b - a + 1, 0)
    idx = np.clip(np.round((np.arange(a, b + 1) - fs[0]) / (fs[1] - fs[0])).astype(int), 0, len(fs) - 1)
    e = sm[idx]; team[e != -2] = e[e != -2]
    return team

if __name__ == "__main__":
    a, b = map(int, os.environ.get("MTAH_WIN", "11185,14934").split(",")); t = fix(a, b)
    for s in range(t.shape[1]):
        v, c = np.unique(t[:, s], return_counts=True)
        if np.isfinite(T.P[a:b + 1, s, 0]).any(): print(s, T.TEAM[s], dict(zip(v.tolist(), c.tolist())))
    np.save(f"{WORK}/teamfix_{a}_{b}.npy", t.astype(np.int8))
