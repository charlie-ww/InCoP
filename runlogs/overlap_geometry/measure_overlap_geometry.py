"""Per-frame co-visibility: can a correspondence-based method even run on this frame?

For every test frame of both splits: how many objects each robot annotates, how
many they share, and how far apart the robots stand. Objects are counted inside
the evaluation range in the ego frame, matching the detector's own gt_range.
"""
import glob
import math
import os

import numpy as np
import yaml

LOADER = yaml.CSafeLoader
ROOT = "/nfs/home/charlie/InCoP/dataset"
XMIN, YMIN, XMAX, YMAX = 0.0, -11.2, 22.4, 11.2
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "perframe_overlap.csv")


def in_range(loc, pose):
    """World point -> ego lidar frame, then the detector's BEV window."""
    yaw = math.radians(pose[4])
    dx, dy = loc[0] - pose[0], loc[1] - pose[1]
    c, s = math.cos(-yaw), math.sin(-yaw)
    x, y = c * dx - s * dy, s * dx + c * dy
    return XMIN <= x <= XMAX and YMIN <= y <= YMAX


def frames(split_dir):
    for scene in sorted(os.listdir(split_dir)):
        d0, d1 = (os.path.join(split_dir, scene, a) for a in ("0", "1"))
        if not (os.path.isdir(d0) and os.path.isdir(d1)):
            continue
        for f0 in sorted(glob.glob(os.path.join(d0, "*.yaml"))):
            f1 = os.path.join(d1, os.path.basename(f0))
            if os.path.exists(f1):
                yield scene, os.path.basename(f0)[:-5], f0, f1


rows = []
for tag, sub in (("low", "IsaacSimOPV2V_hospital"),
                 ("high", "IsaacSimOPV2V_hospital_highoverlap")):
    split_dir = os.path.join(ROOT, sub, "test")
    for scene, stem, f0, f1 in frames(split_dir):
        try:
            with open(f0) as fh:
                a = yaml.load(fh, Loader=LOADER)
            with open(f1) as fh:
                b = yaml.load(fh, Loader=LOADER)
        except Exception:
            continue
        pose, bp = a["lidar_pose"], b["lidar_pose"]
        ea = {k for k, v in (a.get("vehicles") or {}).items()
              if in_range(v["location"], pose)}
        eb = {k for k, v in (b.get("vehicles") or {}).items()
              if in_range(v["location"], pose)}
        dh = abs(pose[4] - bp[4]) % 360.0
        rows.append((tag, scene, stem, len(ea), len(eb), len(ea & eb),
                     len(ea - eb), len(eb - ea),
                     round(math.hypot(pose[0] - bp[0], pose[1] - bp[1]), 4),
                     round(min(dh, 360.0 - dh), 2)))
    print(f"{tag}: {sum(1 for r in rows if r[0] == tag)} frames", flush=True)

with open(OUT, "w") as fh:
    fh.write("split,scene,frame,n_ego,n_partner,n_shared,n_ego_only,"
             "n_partner_only,robot_dist_m,heading_diff_deg\n")
    for r in rows:
        fh.write(",".join(str(x) for x in r) + "\n")

print(f"\nwrote {OUT}\n")
for tag in ("low", "high"):
    sh = np.array([r[5] for r in rows if r[0] == tag])
    po = np.array([r[7] for r in rows if r[0] == tag])
    di = np.array([r[8] for r in rows if r[0] == tag])
    hd = np.array([r[9] for r in rows if r[0] == tag])
    if len(sh) == 0:
        continue
    print(f"=== {tag} overlap  ({len(sh)} frames) ===")
    print(f"  shared objects/frame:       median {np.median(sh):.0f}  "
          f"mean {sh.mean():.2f}  p10 {np.percentile(sh, 10):.0f}  "
          f"p90 {np.percentile(sh, 90):.0f}")
    for k in (0, 1, 2, 3):
        print(f"    frames with <={k} shared: {100.0 * (sh <= k).mean():5.1f}%")
    print(f"  partner-only objects/frame: median {np.median(po):.0f}  "
          f"mean {po.mean():.2f}   frames with >=1: "
          f"{100.0 * (po >= 1).mean():.1f}%")
    print(f"  robot distance (m):   median {np.median(di):.2f}  "
          f"p10 {np.percentile(di, 10):.2f}  p90 {np.percentile(di, 90):.2f}")
    print(f"  |heading diff| (deg): median {np.median(hd):.1f}  "
          f"p10 {np.percentile(hd, 10):.1f}  p90 {np.percentile(hd, 90):.1f}")
