"""Match data access and inspection views for one window of frames (MTAH_WIN = "A,B").

Inputs are the provided match assets only (no ground truth): tactical tracking (ATD), homography, Smart Tagging, the
tactical and broadcast video frames extracted for the window, and the per-window features written by kit.py,
teamfix.py, ballcheck.py, ballfuse.py and bmap.py. Frame f is video frame f = submission row f (25 fps).

  python evidence.py events A B           candidate moments: opponent staying at the ball, ball outside the pitch,
                                          broadcast replays, slow play, Smart Tagging set pieces
  python evidence.py summary A B [STEP]   table every STEP frames: ball position (m), ball speed, nearest home / away
                                          player distance, ball source, players tracked, mean player speed, live/replay
  python evidence.py sheet A B [N]        N tactical frames with tracks and ball drawn, plus a full-resolution zoom
  python evidence.py bview A B [N]        the broadcast frames at the same moments
  python evidence.py zoom F [W]           full-resolution zoom around the ball at frame F
  python evidence.py zoomat F X Y [W]     the same around pixel (X, Y) of the 960x540 frame
  python evidence.py minimap A B [N]      top-down pitch views with the ball trail
  python evidence.py tags A B             Smart Tagging intervals overlapping A..B
Image commands print the path of the image they write. Home = Barcelona (H), away = Athletic Club (A)."""
import os, sys, pickle, xml.etree.ElementTree as ET
from paths import DATA, WORK
import numpy as np
from PIL import Image, ImageDraw

FRAMES = os.environ.get("MTAH_FRAMES", f"{WORK}/frames")
OUT = os.environ.get("MTAH_OUT", f"{WORK}/img")
CACHE = f"{WORK}/atd_cache.pkl"
L, W, FPS = 105.3, 68.0, 25
os.makedirs(OUT, exist_ok=True)


def load():
    if os.path.exists(CACHE): return pickle.load(open(CACHE, "rb"))
    root = ET.parse(f"{DATA}/tactical/BAR-ATH_tactical_atd.xml").getroot()
    home = [t.get("id") for t in root.iter("Team")][0]
    team = np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")], np.int8)
    n = 199319; P = np.full((n, len(team), 2), np.nan, np.float32); B = np.full((n, 2), np.nan, np.float32)
    for line in open(f"{DATA}/tactical/BAR-ATH_tactical_atd.txt"):
        a = line.split(":")
        if len(a) != 3: continue
        i = int(a[0]) - 1
        for s, g in enumerate(a[1].split(";")):
            if g[0] != "N": x, y = g.split(",")[:2]; P[i, s] = (float(x), float(y))
        if a[2][0] != "N": x, y = a[2].split(",")[:2]; B[i] = (float(x), float(y))
    H = np.genfromtxt(f"{DATA}/tactical/BAR-ATH_tactical_homography.csv", delimiter=",", skip_header=1)[:, 1:10].reshape(-1, 3, 3)
    tags = []
    for ins in ET.parse(f"{DATA}/tactical/BAR-ATH_tactical_smart_tagging.xml").iter("instance"):
        lab = {l.findtext("group"): l.findtext("text") for l in ins.iter("label")}
        tags.append((round(float(ins.findtext("start")) * FPS), round(float(ins.findtext("end")) * FPS), ins.findtext("code"), lab))
    d = {"team": team, "P": P, "B": B, "H": H.astype(np.float32), "tags": tags}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True); pickle.dump(d, open(CACHE, "wb")); return d


D = load(); TEAM, P, B, HM, TAGS = D["team"], D["P"], D["B"], D["H"], D["tags"]
VA, VB = map(int, os.environ.get("MTAH_WIN", "11185,14934").split(","))                  # window (MTAH_WIN), first window by default
_tf = f"{WORK}/teamfix_{VA}_{VB}.npy"; TFIX = np.load(_tf) if os.path.exists(_tf) else None
_bc = f"{WORK}/ballcheck_{VA}_{VB}.npy"; BCHK = np.load(_bc) if os.path.exists(_bc) else None
_fp = f"{WORK}/det/fused_{VA}_{VB}.npz"         # absent only while a window is being prepared
_fu = np.load(_fp) if os.path.exists(_fp) else None; FUS, FSRC = (_fu["xy"], _fu["src"]) if _fu is not None else (None, None)      # metres
_bp = f"{WORK}/det/bmap_{VA}_{VB}.npz"
_bv = np.load(_bp) if os.path.exists(_bp) else None; BVH, BLIVE, BLAG = (_bv["H"], _bv["live"], _bv["lag"]) if _bv is not None else (None, None, None)
FULL = os.environ.get("MTAH_FULL", f"{WORK}/tac"); BCF = os.environ.get("MTAH_BCF", f"{WORK}/bc")
SRCN = {"det_anchor": "video", "det_vit": "video?", "met": "track", "bc": "TVtrack", "": "-"}


def ball(f):
    """best ball position (metres) at frame f: our ball detector on the full-res video, else Metrica's tracked ball where a
    ball is visible there, else the ball tracked in the TV broadcast; None if unknown."""
    if VA <= f <= VB:
        x = FUS[f - VA]; return None if np.isnan(x[0]) else (float(x[0]), float(x[1]))
    return None if np.isnan(B[f, 0]) else (float(B[f, 0] * L), float(B[f, 1] * W))


def team_at(f):
    """team of every slot at frame f from the video kit check (0 home, 1 away, -1 = no person under the track); Metrica's otherwise."""
    if TFIX is not None and VA <= f <= VB: return TFIX[f - VA]
    return TEAM.astype(int)


def ball_seen(f):
    """'yes' / 'NO' = is there a ball-like spot in the video at the tracked ball position; '' if unknown."""
    if BCHK is None or not VA <= f <= VB or np.isnan(BCHK[f - VA]): return ""
    return "yes" if BCHK[f - VA] >= 15 else "NO"
mm = lambda f: f"{int(f / FPS // 60)}:{f / FPS % 60:05.2f}"


def proj(f, xy):
    xy = np.asarray(xy, float).reshape(-1, 2); q = np.c_[xy, np.ones(len(xy))] @ HM[f].T
    return q[:, :2] / q[:, 2:3] * 0.5                        # homography is for 1920x1080; frames are 960x540


def nearest(f):
    bl = ball(f)
    if bl is None: return None
    d = np.hypot(P[f, :, 0] * L - bl[0], P[f, :, 1] * W - bl[1]); out = []; tm = team_at(f)
    for t in (0, 1):
        m = (tm == t) & ~np.isnan(d); out.append(float(d[m].min()) if m.any() else np.nan)
    return out


def tags_at(a, b): return [t for t in TAGS if t[0] <= b and t[1] >= a]


def summary(a, b, step=5):
    print("frame   time      ball_x  ball_y  speed  dH    dA    near  src     nH nA  plv  tv   tags")
    for f in range(a, b + 1, step):
        tm = team_at(f); nH = int(np.sum(~np.isnan(P[f, tm == 0, 0]))); nA = int(np.sum(~np.isnan(P[f, tm == 1, 0])))
        tg = ",".join(f"{t[3].get('Type')}/{t[3].get('Team', '')}" for t in tags_at(f, f) if t[2] == "SET PIECES")   # set pieces only; see `tags`
        pl = player_speed(f); tv = ("live" if BLIVE[f - VA] else "repl") if VA <= f <= VB else ""; bl = ball(f)
        src = SRCN.get(str(FSRC[f - VA]), "-") if VA <= f <= VB else "track"
        if bl is None: print(f"{f:6d}  {mm(f):9s}  {'-':>6s}  {'-':>6s}  {'-':>5s}  {'-':5s} {'-':5s} {'-':4s}  {'-':7s} {nH:2d} {nA:2d}  {pl:4.1f} {tv:4s} {tg}"); continue
        g = ball(f - 5) if f >= 5 else None
        sp = np.hypot(bl[0] - g[0], bl[1] - g[1]) * FPS / 5 if g is not None else np.nan
        dH, dA = nearest(f)
        print(f"{f:6d}  {mm(f):9s}  {bl[0]:6.1f}  {bl[1]:6.1f}  {sp:5.1f}  {dH:5.1f} {dA:5.1f} {'H' if dH < dA else 'A':4s}  {src:7s} {nH:2d} {nA:2d}  {pl:4.1f} {tv:4s} {tg}")


def player_speed(f):
    """mean speed (m/s) of the players tracked at both f-5 and f: everyone slows down when play is stopped."""
    if f < 5: return np.nan
    d = np.hypot((P[f, :, 0] - P[f - 5, :, 0]) * L, (P[f, :, 1] - P[f - 5, :, 1]) * W) * FPS / 5
    d = d[~np.isnan(d)]; return float(d.mean()) if len(d) else np.nan


def ball_gap(f):
    """length (s) of the run of untracked-ball frames that contains f (0 if the ball is tracked)."""
    if not np.isnan(B[f, 0]): return 0.0
    a = f
    while a > 0 and np.isnan(B[a - 1, 0]): a -= 1
    b = f
    while b + 1 < len(B) and np.isnan(B[b + 1, 0]): b += 1
    return (b - a + 1) / FPS


def last_ball(f):
    for g in range(f, max(0, f - 250), -1):
        if not np.isnan(B[g, 0]): return g
    return None


def draw(f, scale=1):
    im = Image.open(f"{FRAMES}/f{f:06d}.jpg").convert("RGB"); d = ImageDraw.Draw(im)
    tm = team_at(f)
    for s in range(len(TEAM)):
        if np.isnan(P[f, s, 0]): continue
        x, y = proj(f, P[f, s])[0]
        if tm[s] < 0: d.line([x - 3, y - 3, x + 3, y + 3], fill=(160, 160, 160)); d.line([x - 3, y + 3, x + 3, y - 3], fill=(160, 160, 160)); continue
        c = (40, 120, 255) if tm[s] == 0 else (255, 60, 60)
        d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=c, width=2)
    g = last_ball(f); bx = None
    if g is not None:
        bx, by = proj(f, B[g])[0]; c = (255, 230, 0) if g == f else (150, 150, 150)
        if g == f and ball_seen(f) == "NO": c = (255, 140, 0)
        d.ellipse([bx - 7, by - 7, bx + 7, by + 7], outline=c, width=2)
    bl = ball(f) if VA <= f <= VB else None
    if bl is not None:
        cx, cy = proj(f, [bl[0] / L, bl[1] / W])[0]; d.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], outline=(0, 255, 255), width=2); bx, by = cx, cy
    return im, (bx, by) if bx is not None else None, g


def full_zoom(f, c, w=440, out_w=880):
    """crop from the native 1920x1080 frame (nearest extracted frame) around 960-px point c, with the same overlays."""
    g = f - ((f - VA) % 2); fp = f"{FULL}/p{g - VA:06d}.jpg"
    if not (VA <= f <= VB and os.path.exists(fp)): return None
    im = Image.open(fp).convert("RGB"); d = ImageDraw.Draw(im); tm = team_at(g)
    for s_ in range(len(TEAM)):
        if np.isnan(P[g, s_, 0]) or tm[s_] < 0: continue
        x, y = proj(g, P[g, s_])[0] * 2; col = (40, 120, 255) if tm[s_] == 0 else (255, 60, 60)
        d.ellipse([x - 5, y + 2, x + 5, y + 8], outline=col, width=2)
    bl = ball(g)
    if bl is not None: cx, cy = proj(g, [bl[0] / L, bl[1] / W])[0] * 2; d.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], outline=(0, 255, 255), width=2)
    if c is None: c = (480, 270)
    h = int(w * 9 / 16); x0 = int(min(max(c[0] * 2 - w / 2, 0), 1920 - w)); y0 = int(min(max(c[1] * 2 - h / 2, 0), 1080 - h))
    return im.crop((x0, y0, x0 + w, y0 + h)).resize((out_w, int(out_w * 9 / 16)), Image.LANCZOS)


def zoom_of(im, c, w=240, k=3):
    if c is None: c = (480, 270)
    x0 = int(min(max(c[0] - w / 2, 0), 960 - w)); y0 = int(min(max(c[1] - w * 9 / 32, 0), 540 - w * 9 / 16))
    return im.crop((x0, y0, x0 + w, y0 + int(w * 9 / 16))).resize((int(w * k), int(w * 9 / 16 * k)), Image.LANCZOS)


def label(im, txt):
    d = ImageDraw.Draw(im); d.rectangle([0, 0, 8 * len(txt) + 8, 16], fill=(0, 0, 0)); d.text((4, 2), txt, fill=(255, 255, 255)); return im


def sheet(a, b, n=6):
    fs = [int(round(a + i * (b - a) / max(1, n - 1))) for i in range(n)]; rows = []
    for f in fs:
        im, c, g = draw(f); z = full_zoom(f, c) or zoom_of(im, c)
        if z.height > 540: z = z.resize((int(z.width * 540 / z.height), 540))
        ball = ("ball tracked" + (" BUT NOT VISIBLE THERE" if ball_seen(f) == "NO" else "")) if g == f else (f"ball last tracked {f - g} fr ago" if g is not None else "no ball")
        row = Image.new("RGB", (960 + z.width, 540)); row.paste(label(im, f"frame {f} {mm(f)}  {ball}"), (0, 0)); row.paste(z, (960, 0)); rows.append(row)
    S = Image.new("RGB", (rows[0].width, 540 * len(rows)))
    for i, r in enumerate(rows): S.paste(r, (0, 540 * i))
    S = S.resize((S.width * 2 // 3, S.height * 2 // 3)); p = f"{OUT}/sheet_{a}_{b}_{n}.jpg"; S.save(p, quality=85); print(p)


def zoom(f, w=240):
    im, c, g = draw(f); z = full_zoom(f, c, w * 2, 1200) or zoom_of(im, c, w, 4); p = f"{OUT}/zoom_{f}.jpg"; label(z, f"frame {f} {mm(f)}").save(p, quality=90); print(p)


def zoomat(f, x, y, w=240):
    """4x zoom of a W-px box centred on pixel (x, y) of the 960x540 frame: use it to find the REAL ball when the tracked
    ball is wrong (orange ring) - read x, y off the sheet's left image (sheet images are shown at 2/3 size: multiply by 1.5)."""
    im, c, g = draw(f); z = full_zoom(f, (x, y), w * 2, 1200) or zoom_of(im, (x, y), w, 4); p = f"{OUT}/zoomat_{f}_{x}_{y}.jpg"; label(z, f"frame {f} {mm(f)} at {x},{y}").save(p, quality=90); print(p)


def bview(a, b, n=6):
    """the TV broadcast at the same moments (closer camera; it cuts to replays / close-ups, marked REPLAY OR CLOSE-UP).
    Cyan ring = the best ball position, drawn only on live pitch views."""
    fs = [int(round(a + i * (b - a) / max(1, n - 1))) for i in range(n)]; rows = []
    for f in fs:
        k = f - int(BLAG[f - VA]) + int(os.environ.get("MTAH_BLAG", 4014)); k = k if os.path.exists(f"{BCF}/f{k:06d}.jpg") else f      # extracted with the fixed lag of the half
        im = Image.open(f"{BCF}/f{k:06d}.jpg").convert("RGB"); d = ImageDraw.Draw(im); live = bool(BLIVE[f - VA]); bl = ball(f)
        if live and bl is not None:
            q = BVH[f - VA].astype(float) @ np.array([bl[0] / L, bl[1] / W, 1.0]); x, y = q[0] / q[2] * 0.5, q[1] / q[2] * 0.5
            d.ellipse([x - 12, y - 12, x + 12, y + 12], outline=(0, 255, 255), width=2)
        rows.append(label(im, f"frame {f} {mm(f)}  TV broadcast " + ("live view" if live else "REPLAY OR CLOSE-UP (not live play)")))
    cols = 2; S = Image.new("RGB", (960 * cols, 540 * ((len(rows) + 1) // cols)))
    for i, r in enumerate(rows): S.paste(r, ((i % cols) * 960, (i // cols) * 540))
    S = S.resize((S.width * 2 // 3, S.height * 2 // 3)); p = f"{OUT}/bview_{a}_{b}_{n}.jpg"; S.save(p, quality=85); print(p)


def minimap(a, b, n=6):
    fs = [int(round(a + i * (b - a) / max(1, n - 1))) for i in range(n)]; s = 4; pw, ph = int(L * s) + 20, int(W * s) + 30
    S = Image.new("RGB", (pw * 2, ph * ((n + 1) // 2)), (30, 90, 40))
    for k, f in enumerate(fs):
        ox, oy = (k % 2) * pw + 10, (k // 2) * ph + 20; d = ImageDraw.Draw(S)
        d.rectangle([ox, oy, ox + L * s, oy + W * s], outline=(230, 230, 230)); d.line([ox + L * s / 2, oy, ox + L * s / 2, oy + W * s], fill=(230, 230, 230))
        d.text((ox, oy - 16), f"frame {f} {mm(f)}", fill=(255, 255, 255))
        tr = [B[g] for g in range(max(a, f - 50), f + 1) if not np.isnan(B[g, 0])]
        for p0, p1 in zip(tr, tr[1:]): d.line([ox + p0[0] * L * s, oy + p0[1] * W * s, ox + p1[0] * L * s, oy + p1[1] * W * s], fill=(255, 230, 0))
        for t_ in range(len(TEAM)):
            if np.isnan(P[f, t_, 0]): continue
            tt = team_at(f)[t_]
            if tt < 0: continue
            x, y = ox + P[f, t_, 0] * L * s, oy + P[f, t_, 1] * W * s; c = (60, 140, 255) if tt == 0 else (255, 70, 70)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=c)
        if not np.isnan(B[f, 0]): x, y = ox + B[f, 0] * L * s, oy + B[f, 1] * W * s; d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 0), outline=(0, 0, 0))
    p = f"{OUT}/minimap_{a}_{b}_{n}.jpg"; S.save(p, quality=85); print(p)


def events(a, b):
    """code-made candidate list (hints only, often wrong): who touches the ball, ball leaving the pitch, tracked ball not
    visible in the video, slow-play stretches, set-piece tags."""
    out = []; last = None; run_ = [0, 0]; inv = None; slow = None; oob = None
    for f in range(a, b + 1):
        nd = nearest(f)
        for t in (0, 1):
            run_[t] = run_[t] + 1 if nd is not None and nd[t] <= 1.5 and nd[t] <= nd[1 - t] else 0
            if run_[t] == 5 and last != t: out.append((f - 4, f"{'H' if t == 0 else 'A'} player at the ball (<=1.5 m for 0.2 s+), first since {'H' if last == 0 else 'A' if last == 1 else '-'}")); last = t
        bl = ball(f); o = bl is not None and (bl[0] < -0.3 or bl[0] > L + 0.3 or bl[1] < -0.3 or bl[1] > W + 0.3)
        if o and oob is None: oob = f
        if not o and oob is not None: out.append((oob, f"ball tracked OUTSIDE the pitch until {f - 1}")); oob = None
        v = VA <= f <= VB and not BLIVE[f - VA]
        if v and inv is None: inv = f
        if not v and inv is not None:
            if f - inv >= 25: out.append((inv, f"TV broadcast shows a REPLAY or close-up until {f - 1} (often right after a foul, a chance or the ball going out)"))
            inv = None
        sl = player_speed(f) < 1.6
        if sl and slow is None: slow = f
        if not sl and slow is not None:
            if f - slow >= 25: out.append((slow, f"players slow (plv < 1.6 m/s) until {f - 1}: stoppage?"))
            slow = None
    for f0, txt in ((oob, "ball tracked OUTSIDE the pitch until end"), (slow, "players slow until end"), (inv, "TV broadcast replay / close-up until end")):
        if f0 is not None: out.append((f0, txt))
    for t in tags_at(a, b):
        if t[2] == "SET PIECES": out.append((t[0], f"Smart Tagging set piece {t[3].get('Type')} team {t[3].get('Team')}"))
    for f, txt in sorted(out): print(f"{f:6d} {mm(f):9s} {txt}")


def tags(a, b):
    for s, e, c, lab in tags_at(a, b): print(f"{s}-{e} ({mm(s)}-{mm(e)}) {c} {lab}")


if __name__ == "__main__":
    c, *x = sys.argv[1:]; x = [int(v) for v in x]
    {"summary": summary, "events": events, "sheet": sheet, "bview": bview, "zoom": zoom, "zoomat": zoomat, "minimap": minimap, "tags": tags}[c](*x)
