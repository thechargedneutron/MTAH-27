"""Ball and person detection on the tactical video (GPU). For each batch A:B of frames:
  1. RF-DETR Large (COCO weights, Apache-2.0) on 3x2 overlapping tiles of the 1920x1080 frame: balls >= 0.05,
     persons >= 0.3, merged by class-wise NMS.
  2. Every 2nd frame: shirt / shorts colour features of each person box, and the frame saved as JPEG.
Outputs in OUT/<A>_<B>/: dets_<A>_<B>.npz (frame, cls, x1, y1, x2, y2, conf), feat_<A>_<B>.npz, frames/f<frame>.jpg.
usage: python detect.py VIDEO OUT A:B [A:B ...] [--batch 12]
The video is read straight through from frame 0 (seeking in long mp4 files can land on the wrong frame); frames outside
the batches are grabbed but not decoded."""
import sys, os, time, argparse, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("out"); ap.add_argument("ranges", nargs="+")
ap.add_argument("--batch", type=int, default=12); A = ap.parse_args()
import rfdetr, torch, torchvision
model = rfdetr.RFDETRLarge()
OV = 64; xs = [0, 640 - OV // 2, 1280 - OV]; tw = 640 + OV; ys = [0, 540 - OV]; th = 540 + OV; tiles = [(x, y) for y in ys for x in xs]

def lab(rgb):
    c = rgb / 255.0; c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]); xyz = c @ M.T / [0.9505, 1.0, 1.089]
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.c_[116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])]

def box_feat(im, x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    if h < 14: return None
    gx1, gx2 = int(max(0, x1 - w)), int(min(1920, x2 + w)); gy1, gy2 = int(max(0, y2 - 0.3 * h)), int(min(1080, y2 + 0.2 * h))
    g = im[gy1:gy2, gx1:gx2].reshape(-1, 3)
    if len(g) < 10: return None
    gm = np.median(g, 0); out = []
    for t0, t1 in ((0.20, 0.50), (0.55, 0.72)):
        px = im[int(y1 + t0 * h):int(y1 + t1 * h), int(x1 + 0.25 * w):int(x2 - 0.25 * w) + 1].reshape(-1, 3)
        px = px[np.linalg.norm(px - gm, axis=1) > 35]
        if len(px) < 5: return None
        out.append(px)
    sh, so = out; L1 = lab(np.median(sh, 0)[None])[0]; L2 = lab(np.median(so, 0)[None])[0]
    mx, mn = sh.max(1), sh.min(1); sat = (mx - mn) / (mx + 1)
    return (*L1, *L2, np.mean(sat > 0.4), np.mean((sat < 0.2) & (mx > 150)), np.mean(mx < 70), len(sh), *lab(gm[None])[0][:2])

cap = cv2.VideoCapture(A.video); pos = 0
for rg in A.ranges:
    a, b = map(int, rg.split(":")); out = f"{A.out}/{a}_{b}"; FR = f"{out}/frames"; os.makedirs(FR, exist_ok=True)
    while pos < a: cap.grab(); pos += 1
    rows, feats = [], []; t0 = time.time(); f = a; buf, fids = [], []
    def flush():
        ims = [im[y:y + th, x:x + tw] for im in buf for (x, y) in tiles]
        res = model.predict(ims, threshold=0.05)
        for i, fid in enumerate(fids):
            bx, cf, cl = [], [], []
            for j, (x, y) in enumerate(tiles):
                r = res[i * len(tiles) + j]
                for c, thr in ((37, 0.05), (1, 0.3)):
                    m = (r.class_id == c) & (r.confidence >= thr)
                    if m.any(): bx.append(r.xyxy[m] + [x, y, x, y]); cf.append(r.confidence[m]); cl.append(np.full(m.sum(), c))
            if not bx: continue
            bx, cf, cl = np.vstack(bx), np.concatenate(cf), np.concatenate(cl)
            keep = torchvision.ops.batched_nms(torch.tensor(bx, dtype=torch.float32), torch.tensor(cf, dtype=torch.float32), torch.tensor(cl), 0.5).numpy()
            sample = (fid - a) % 2 == 0; imf = buf[i].astype(np.float32) if sample else None
            for k in keep:
                if sample and cl[k] == 1:
                    ft = box_feat(imf, *bx[k])
                    if ft is not None: feats.append((len(rows), *ft))
                rows.append((fid, cl[k], *bx[k], cf[k]))
        buf.clear(); fids.clear()
    while f <= b:
        ok, im = cap.read(); pos += 1
        if not ok: break
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if (f - a) % 2 == 0: cv2.imwrite(f"{FR}/f{f:06d}.jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        buf.append(im); fids.append(f); f += 1
        if len(buf) >= A.batch: flush()
        if (f - a) % 1000 == 0: print(f, f"{(f - a) / (time.time() - t0):.1f} fps", len(rows), flush=True)
    if buf: flush()
    np.savez_compressed(f"{out}/dets_{a}_{b}.npz", d=np.array(rows, np.float32))
    np.savez_compressed(f"{out}/feat_{a}_{b}.npz", f=np.array(feats, np.float32))
    print("DONE", a, b, f - a, "frames", len(rows), "dets", len(feats), "feats", round(time.time() - t0), "s", flush=True)
