"""Writes pipeline/windows.json: each half (frame ranges from the data bundle's DATA.md) cut into 3750-frame windows,
each with the broadcast lag of its half (tactical frame = broadcast frame + lag) from bsync.py.
usage: python src/make_windows.py [OUT]"""
import json, os, sys
import bsync
HALVES = [(11185, 90651), (110808, 190197)]                 # DATA.md: half_video_blocks("BAR-ATH")
out = []
for a0, b0 in HALVES:
    lag = bsync.lag(a0, b0); print("half", a0, b0, "lag", lag)
    out += [dict(a=a, b=min(a + 3749, b0), lag=lag) for a in range(a0, b0 + 1, 3750)]
p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline", "windows.json")
json.dump(out, open(p, "w")); print(p, len(out), "windows")
