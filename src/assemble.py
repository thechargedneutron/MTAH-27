"""Join the per-segment label files (A_B.json, runs of H/A/D) into the submission CSV; frames outside the two halves
are X. usage: python src/assemble.py OUT.csv LABEL_DIR"""
import json, glob, sys
N = 199319; lab = ["X"] * N
for fn in sorted(glob.glob(f"{sys.argv[2]}/*_*.json")):
    for r in json.load(open(fn))["runs"]:
        for f in range(int(r["start"]), int(r["end"]) + 1): lab[f] = r["label"]
with open(sys.argv[1], "w") as fo:
    fo.write("frame,label\n"); fo.writelines(f"{i},{l}\n" for i, l in enumerate(lab))
print(sys.argv[1], {k: lab.count(k) for k in "HADX"})
