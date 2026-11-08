"""How big is the misalignment compared with the objects it has to move?

Outdoors a 0.3 m pose error slides a car by a fifteenth of its own length and
the box still overlaps itself. Indoors the objects are small enough that the
same error moves them clear of themselves. This script measures the object
footprints from the annotations and reports, per class, the displacement at
which a detection stops counting as a hit at each IoU threshold we report.
"""
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import yaml

LOADER = yaml.CSafeLoader
ROOT = "/nfs/home/charlie/InCoP/dataset/IsaacSimOPV2V_hospital/test"
IOUS = (0.3, 0.5, 0.7)
# Measured actual misalignment for the project's sigma_pos = 0.2 setting
# (runlogs/ + figures/overlap_noise_model): 0.43 m near the robot, 1.26 m at
# the edge of the sensing range.
ACTUAL = {"sigma_pos=0.2, near robot": 0.43, "sigma_pos=0.2, range edge": 1.26}


def box_iou_shifted(w, l, d, n_dir=72):
    """Mean IoU of an axis-aligned w x l box against itself shifted by d,
    averaged over shift directions."""
    if d <= 0:
        return 1.0
    vals = []
    for th in np.linspace(0, 2 * math.pi, n_dir, endpoint=False):
        dx, dy = abs(d * math.cos(th)), abs(d * math.sin(th))
        ox, oy = max(0.0, w - dx), max(0.0, l - dy)
        inter = ox * oy
        vals.append(inter / (2 * w * l - inter))
    return float(np.mean(vals))


def critical_d(w, l, iou_target, hi=6.0):
    """Smallest displacement whose mean IoU falls below iou_target."""
    lo = 0.0
    if box_iou_shifted(w, l, hi) > iou_target:
        return float("inf")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if box_iou_shifted(w, l, mid) > iou_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


sizes = defaultdict(list)
for scene in sorted(os.listdir(ROOT)):
    for agent in ("0", "1"):
        for f in sorted(glob.glob(os.path.join(ROOT, scene, agent, "*.yaml")))[:6]:
            try:
                with open(f) as fh:
                    y = yaml.load(fh, Loader=LOADER)
            except Exception:
                continue
            for v in (y.get("vehicles") or {}).values():
                e = v.get("extent")
                cn = v.get("class_name", "?")
                if e:
                    # extent holds half-extents, OPV2V convention
                    sizes[cn].append((2 * e[0], 2 * e[1], 2 * e[2]))

print(f"{'class':18s} {'n':>6s} {'width':>7s} {'length':>7s} "
      f"{'sqrt(area)':>10s}   displacement that breaks IoU")
print(f"{'':18s} {'':>6s} {'(m)':>7s} {'(m)':>7s} {'(m)':>10s}   "
      f"{'@0.3':>7s} {'@0.5':>7s} {'@0.7':>7s}")
rows = []
for cn in sorted(sizes):
    a = np.array(sizes[cn])
    w, l = float(np.median(a[:, 0])), float(np.median(a[:, 1]))
    s = math.sqrt(w * l)
    ds = [critical_d(w, l, t) for t in IOUS]
    rows.append((cn, len(a), w, l, s, ds))
    print(f"{cn:18s} {len(a):6d} {w:7.2f} {l:7.2f} {s:10.2f}   "
          + " ".join(f"{d:7.2f}" for d in ds))

allw = np.array([r[2] for r in rows])
alll = np.array([r[3] for r in rows])
w, l = float(np.median(allw)), float(np.median(alll))
d03 = critical_d(w, l, 0.3)
print(f"\n{'MEDIAN CLASS':18s} {'':6s} {w:7.2f} {l:7.2f} "
      f"{math.sqrt(w*l):10.2f}   " + " ".join(f"{critical_d(w,l,t):7.2f}" for t in IOUS))

print("\n=== indoor objects vs the misalignment they actually get ===")
for label, d in ACTUAL.items():
    broken = sum(1 for r in rows if d >= r[5][0])
    print(f"  {label:28s} = {d:.2f} m  ->  "
          f"ratio to median object {d / math.sqrt(w*l):.2f}x, "
          f"breaks IoU@0.3 for {broken}/{len(rows)} classes")

print("\n=== outdoor reference (typical cooperative-perception benchmark) ===")
CAR_W, CAR_L = 1.8, 4.5
cd = critical_d(CAR_W, CAR_L, 0.3)
print(f"  car {CAR_W} x {CAR_L} m: breaks IoU@0.3 at {cd:.2f} m displacement")
for d in (0.2, 0.5):
    print(f"    benchmark pose noise {d:.1f} m -> ratio to object "
          f"{d / math.sqrt(CAR_W*CAR_L):.3f}x, "
          f"{'SAFE' if d < cd else 'BROKEN'} "
          f"({100.0 * d / cd:.0f}% of the breaking distance)")
print(f"\n  indoor median object: 0.43 m actual misalignment is "
      f"{100.0 * 0.43 / d03:.0f}% of its breaking distance; "
      f"1.26 m is {100.0 * 1.26 / d03:.0f}%.")

print("\n=== why coarsening cannot rescue it ===")
print(f"  To tolerate a {1.26:.2f} m shift you must blur to roughly that scale.")
print(f"  The median indoor object is {math.sqrt(w*l):.2f} m across, i.e. "
      f"{math.sqrt(w*l)/1.26:.2f}x the blur scale - it is erased.")
print(f"  A {CAR_W}x{CAR_L} m car is {math.sqrt(CAR_W*CAR_L)/0.5:.1f}x the "
      f"0.5 m blur scale an outdoor method needs - it survives.")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "size_ratio.json")
with open(out, "w") as fh:
    json.dump({"per_class": [{"class": r[0], "n": r[1], "w": r[2], "l": r[3],
                              "sqrt_area": r[4],
                              "break_iou": dict(zip(map(str, IOUS), r[5]))}
                             for r in rows],
               "median_class": {"w": w, "l": l, "break_iou_0.3": d03}}, fh, indent=1)
print(f"\nwrote {out}")
