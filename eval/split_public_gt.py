"""Split the public ground truth (frames 11185-18684) into a development part (11185-14934) and a held-out part
(14935-18684); every other frame becomes X, so the official scorer grades only that part.
usage: python eval/split_public_gt.py BAR-ATH_possession_gt_public.csv OUT_DIR"""
import csv, os, sys
PARTS = {"gt_dev.csv": (11185, 14934), "gt_holdout.csv": (14935, 18684)}
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
for name, (a, b) in PARTS.items():
    with open(src) as fi, open(os.path.join(out, name), "w", newline="") as fo:
        r = csv.reader(fi); w = csv.writer(fo, lineterminator="\n"); w.writerow(next(r))
        for f, lab in r: w.writerow([f, lab if a <= int(f) <= b else "X"])
    print(os.path.join(out, name), a, b)
