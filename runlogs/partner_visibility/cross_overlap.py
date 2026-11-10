"""Does seeing the partner robot help exactly when the scene overlap does not?

The free-space signal has a known weakness: the evidence it needs is most
plentiful when it is least necessary. If direct sight of the partner robot has
the same shape, it is not an escape from the problem. If it is independent, it
is a strong baseline we have to acknowledge.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
vis = pd.read_csv(os.path.join(HERE, "partner_visibility_test.csv"))
ov = pd.read_csv(os.path.join(HERE, "..", "overlap_geometry", "perframe_overlap.csv"))
ov["frame"] = ov["frame"].astype(str).str.zfill(7)
vis["frame"] = vis["frame"].astype(str).str.zfill(7)
d = vis.merge(ov, on=["case" if "case" in ov else "scene", "frame"], how="inner") \
    if "case" in ov.columns else vis.merge(
        ov.rename(columns={"scene": "case"}), on=["case", "frame"], how="inner")
print(f"joined {len(d)} frames of {len(vis)} visibility rows "
      f"and {len(ov)} overlap rows")
if not len(d):
    raise SystemExit("join failed")

d["sees"] = d.ego_sees_partner.astype(bool)
print(f"\nsplits present: {sorted(d.split.unique())}")
d = d[d.split == "low"] if "low" in set(d.split) else d
print(f"using {len(d)} low-overlap frames\n")

print("=== co-visible scene objects vs seeing the partner robot ===")
print(f"{'n_shared':>10} {'frames':>7} {'sees partner':>13}")
for k in sorted(d.n_shared.unique())[:9]:
    s = d[d.n_shared == k]
    print(f"{k:>10} {len(s):>7} {100*s.sees.mean():12.1f}%")
hi = d[d.n_shared >= 9]
if len(hi):
    print(f"{'>=9':>10} {len(hi):>7} {100*hi.sees.mean():12.1f}%")

print("\n=== the question that matters: the hard frames ===")
hard = d[d.n_shared <= 2]
easy = d[d.n_shared > 2]
print(f"  frames with <=2 co-visible objects (the ones that break CoAlign/HERO):")
print(f"    {len(hard)} frames, {100*len(hard)/len(d):.1f}% of the split")
print(f"    ego sees the partner robot in {100*hard.sees.mean():.1f}% of them")
print(f"  frames with >2 co-visible objects:")
print(f"    {len(easy)} frames, ego sees the partner in {100*easy.sees.mean():.1f}%")
print(f"\n  zero co-visible objects: {len(d[d.n_shared==0])} frames, "
      f"sees partner in {100*d[d.n_shared==0].sees.mean():.1f}%")

r = np.corrcoef(d.n_shared, d.sees.astype(float))[0, 1]
sp = pd.Series(d.n_shared).rank().corr(pd.Series(d.sees.astype(float)).rank())
print(f"\n  correlation between n_shared and seeing the partner: "
      f"Pearson {r:+.3f}, Spearman {sp:+.3f}")

print("\n=== geometry of the two cases ===")
for lab, s in (("sees partner", d[d.sees]), ("does not", d[~d.sees])):
    print(f"  {lab:14} n={len(s):5d}  dist median {s.robot_dist_m.median():5.2f} m"
          f"  heading diff median {s.heading_diff_deg.median():6.1f} deg"
          f"  n_shared median {s.n_shared.median():.1f}"
          f"  n_partner_only median {s.n_partner_only.median():.1f}")

print("\n=== but is SEEING the partner accurate enough? ===")
# A partner heading error of delta radians displaces an object the partner
# reports at range r from itself by r * delta. The budget comes from the size
# analysis: the median indoor object stops counting as a hit past 0.161 m.
import json
sz = json.load(open(os.path.join(HERE, "..", "size_ratio", "size_ratio.json")))
BUDGET = sz["median_class"]["break_iou_0.3"]
CAR = 1.23
print(f"  displacement budget: {BUDGET:.3f} m (median indoor object, IoU@0.3)")
print(f"\n  {'object range from partner':>26} {'indoor heading budget':>22}"
      f" {'outdoor (car) budget':>21}")
for r in (4.0, 8.0, 15.0, 22.4):
    ind = np.degrees(BUDGET / r)
    out = np.degrees(CAR / r)
    print(f"  {r:>22.1f} m {ind:>20.2f} deg {out:>18.2f} deg")
print(f"\n  So even with the partner's POSITION known exactly, its HEADING has")
print(f"  to be right to about 1 degree for mid-range objects and 0.4 degrees")
print(f"  at the edge of its range. A car outdoors gets 8.8 and 3.1 degrees.")
print(f"  Median inter-robot distance here is {d.robot_dist_m.median():.2f} m, so a")
print(f"  robot is a small, roughly symmetric blob at that range - recovering")
print(f"  its yaw to a degree from one LiDAR view is not a solved problem.")
print(f"\n  This is the honest answer to 'why not just look at the other robot':")
print(f"  seeing it gives you position, and position is not what breaks us.")
