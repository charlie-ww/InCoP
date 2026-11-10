"""Is "co-visible" the wrong notion for a static scene?

Every pose-correction method in the literature matches what two agents see in
the SAME frame, because outdoors the cars move and only simultaneous
observations are comparable. Indoors the furniture does not move. A chair the
ego saw thirty frames ago and the partner sees now is a valid correspondence.

This counts correspondences both ways for every frame of the test split:

  same-frame : objects both robots see right now
  memory     : objects the partner sees right now that the ego has seen at any
               time up to now in this episode (and the reverse)

It first checks the two premises: object ids persist across frames, and the
objects are static.
"""
import glob, math, os
from collections import defaultdict
import numpy as np
import yaml

LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
ROOT = "/nfs/home/charlie/InCoP/dataset/IsaacSimOPV2V_hospital/test"
HERE = os.path.dirname(os.path.abspath(__file__))

rows, moved, steps = [], [], []
for case in sorted(os.listdir(ROOT)):
    d0, d1 = os.path.join(ROOT, case, "0"), os.path.join(ROOT, case, "1")
    if not (os.path.isdir(d0) and os.path.isdir(d1)):
        continue
    stems = sorted(os.path.basename(f)[:-5] for f in glob.glob(d0 + "/*.yaml")
                   if os.path.exists(os.path.join(d1, os.path.basename(f))))
    ego_mem, par_mem = set(), set()
    first_loc, prev = {}, None
    for t, st in enumerate(stems):
        with open(os.path.join(d0, st + ".yaml")) as fh:
            a = yaml.load(fh, Loader=LOADER)
        with open(os.path.join(d1, st + ".yaml")) as fh:
            b = yaml.load(fh, Loader=LOADER)
        E = set((a.get("vehicles") or {}).keys())
        P = set((b.get("vehicles") or {}).keys())
        for src in (a, b):
            for k, v in (src.get("vehicles") or {}).items():
                loc = v["location"]
                if k in first_loc:
                    moved.append(math.dist(loc[:2], first_loc[k][:2]))
                else:
                    first_loc[k] = loc
        pose = a["lidar_pose"]
        if prev is not None:
            steps.append(math.hypot(pose[0] - prev[0], pose[1] - prev[1]))
        prev = pose
        same = len(E & P)
        # what can be matched NOW if each robot remembers what it has seen
        mem_p = len(P & (ego_mem | E))      # partner's current view vs ego memory
        mem_e = len(E & (par_mem | P))      # ego's current view vs partner memory
        mem_any = len((ego_mem | E) & (par_mem | P))
        rows.append((case, st, t, len(E), len(P), same, mem_p, mem_e, mem_any))
        ego_mem |= E
        par_mem |= P

r = np.array([x[2:] for x in rows], dtype=float)
t, nE, nP, same, mem_p, mem_e, mem_any = r.T
n = len(r)
moved = np.array(moved)
print(f"{n} frames, {len(set(x[0] for x in rows))} episodes, "
      f"median episode length {np.median([sum(1 for x in rows if x[0]==c) for c in set(x[0] for x in rows)]):.0f} frames")
print("\n=== premises ===")
print(f"  same object id re-observed {len(moved)} times; displacement from first "
      f"sighting: median {np.median(moved):.4f} m, 99th pct {np.percentile(moved,99):.4f} m, "
      f"max {moved.max():.4f} m")
print(f"  robot step between consecutive frames: median {np.median(steps):.2f} m, "
      f"90th pct {np.percentile(steps,90):.2f} m")

def share(x, k):
    return 100.0 * (x >= k).mean()

print("\n=== correspondences available per frame ===")
print(f"{'':34s} {'>=1':>7} {'>=2':>7} {'>=3':>7} {'median':>7}")
for name, x in (("same frame (what the field counts)", same),
                ("partner now vs ego memory", mem_p),
                ("ego now vs partner memory", mem_e),
                ("both memories", mem_any)):
    print(f"{name:34s} {share(x,1):6.1f}% {share(x,2):6.1f}% {share(x,3):6.1f}% "
          f"{np.median(x):7.1f}")

print("\n=== the frames that defeat single-frame methods (same-frame <= 2) ===")
hard = same <= 2
print(f"  {hard.sum()} frames ({100*hard.mean():.1f}%)")
for name, x in (("partner now vs ego memory", mem_p), ("both memories", mem_any)):
    print(f"  rescued to >=3 by {name:26s}: {100*(x[hard]>=3).mean():5.1f}%")
zero = same == 0
print(f"  zero same-frame ({zero.sum()} frames): >=3 via both memories "
      f"{100*(mem_any[zero]>=3).mean():.1f}%, >=1 {100*(mem_any[zero]>=1).mean():.1f}%")

print("\n=== how fast does memory build up? (frames into the episode) ===")
for lo, hi in ((0, 1), (1, 5), (5, 10), (10, 20), (20, 40), (40, 999)):
    m = (t >= lo) & (t < hi)
    if m.sum():
        print(f"  frames {lo:>2}-{min(hi,99):<2}: n={m.sum():4d}  same-frame>=3 "
              f"{share(same[m],3):5.1f}%   both-memories>=3 {share(mem_any[m],3):5.1f}%")

with open(os.path.join(HERE, "cross_time.csv"), "w") as fh:
    fh.write("case,frame,t,n_ego,n_partner,same,partner_vs_ego_mem,ego_vs_partner_mem,both_mem\n")
    for x in rows:
        fh.write(",".join(str(v) for v in x) + "\n")
