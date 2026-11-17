"""Broadcast tracking loader (same ATD format as tactical): P[frame, slot, 2] normalised pitch coords, team per slot, ball."""
import os, pickle, numpy as np, xml.etree.ElementTree as ET
from paths import DATA, WORK
D = f"{DATA}/broadcast"; C = f"{WORK}/bcast_cache.pkl"
def load():
    if os.path.exists(C): return pickle.load(open(C, "rb"))
    root = ET.parse(f"{D}/BAR-ATH_broadcast_atd.xml").getroot()
    teams = [t.get("id") for t in root.iter("Team")]; names = [t.findtext("Name") for t in root.iter("Team")]
    team = np.array([teams.index(p.get("teamId")) for p in root.iter("Player")], np.int8)
    n = sum(1 for _ in open(f"{D}/BAR-ATH_broadcast_atd.txt"))
    P = np.full((n, len(team), 2), np.nan, np.float32); B = np.full((n, 2), np.nan, np.float32)
    for line in open(f"{D}/BAR-ATH_broadcast_atd.txt"):
        a = line.split(":")
        if len(a) != 3: continue
        i = int(a[0]) - 1
        for s, g in enumerate(a[1].split(";")):
            if g and g[0] != "N": x, y = g.split(",")[:2]; P[i, s] = (float(x), float(y))
        if a[2][0] != "N": x, y = a[2].split(",")[:2]; B[i] = (float(x), float(y))
    H = np.genfromtxt(f"{D}/BAR-ATH_broadcast_homography.csv", delimiter=",", skip_header=1)
    d = dict(team=team, names=names, P=P, B=B, H=H[:, 1:10].reshape(-1, 3, 3).astype(np.float32), valid=H[:, -1].astype(bool))
    pickle.dump(d, open(C, "wb")); return d
if __name__ == "__main__":
    d = load(); print(d["names"], d["P"].shape, np.bincount(d["team"]), "valid", d["valid"].mean().round(3))
    print("players per frame", np.nanmean((~np.isnan(d["P"][:, :, 0])).sum(1)))
