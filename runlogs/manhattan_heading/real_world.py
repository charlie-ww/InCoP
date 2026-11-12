"""Same estimator on the real-robot recordings (dataset/real_world).

The two robots differ: floor at about -0.9 and -0.7 m, ceiling at about 1.8 and
2.1 m in their sensor frames. One wall band that fits both is used: -0.45 to
1.40 m. Poses are the recordings' own lidar_pose.

Robot 1's points are layered at fixed forward distances (about 700 distinct x
values in 10k points, typical of a depth camera). Each layer is a line at
exactly 0 deg in the sensor frame, which the projection score mistakes for a
wall facing the robot: the raw estimate sticks at 0.00. --dequantize spreads
each point uniformly within the gap to the neighbouring layers before the
estimate; the LiDAR robot (0) is unaffected because its x values are distinct.
"""
import glob, math, os, sys
import numpy as np, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_heading as M
from opencood.utils import pcd_utils
M.Z_LO, M.Z_HI = -0.45, 1.40
wrap = M.wrap
ROOT = "/nfs/home/charlie/InCoP/dataset/real_world"

RNG = np.random.default_rng(0)
DEQ = "--dequantize" in sys.argv


def dequantize(p):
    lv, inv = np.unique(p[:, 0], return_inverse=True)
    if len(lv) > 0.5 * len(p):
        return p                                  # not layered
    gap = np.diff(lv)
    lo = np.concatenate([[gap[0] if len(gap) else 0], gap])      # gap below each level
    hi = np.concatenate([gap, [gap[-1] if len(gap) else 0]])     # gap above
    q = p.copy()
    q[:, 0] += RNG.uniform(-0.5, 0.5, len(p)) * np.where(RNG.random(len(p)) < 0.5, lo[inv], hi[inv])
    return q


def circ_med90(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    if len(x) == 0: return np.nan
    return x[np.argmin(np.abs(wrap(x[:, None] - x[None, :], 90)).sum(1))]

for sp in ("train", "validate"):
    d0, d1 = f"{ROOT}/{sp}/case_0/0", f"{ROOT}/{sp}/case_0/1"
    stems = sorted(os.path.basename(f)[:-5] for f in glob.glob(d0 + "/*.yaml")
                   if os.path.exists(os.path.join(d1, os.path.basename(f))))
    B, Y, D = [[], []], [[], []], []
    for st in stems:
        poses = []
        for i, d in enumerate((d0, d1)):
            y = yaml.load(open(f"{d}/{st}.yaml"), Loader=M.LOADER)["lidar_pose"]
            pts = pcd_utils.pcd_to_np(f"{d}/{st}.pcd")
            a = M.manhattan_angle(dequantize(pts) if DEQ else pts)["alpha"]
            B[i].append((a + y[4]) % 90); Y[i].append(y[4]); poses.append(y)
        D.append(math.hypot(poses[1][0] - poses[0][0], poses[1][1] - poses[0][1]))
    B = [np.array(b) for b in B]
    print(f"\n=== real_world/{sp} ({'dequantized' if DEQ else 'raw'}): {len(stems)} frame pairs, robots {np.median(D):.1f} m apart (median)")
    for i in range(2):
        m = circ_med90(B[i]); dev = np.abs(wrap(B[i] - m, 90)); dev = np.where(np.isfinite(dev), dev, np.inf)
        turn = np.ptp(np.unwrap(np.radians(Y[i])))
        print(f"  robot {i}: turned {math.degrees(turn):5.1f} deg in total; wall angle in world {m:5.2f} deg, "
              f"deviation median {np.median(dev):.2f}, <1.15 {100*(dev<1.15).mean():.1f}%, >5 {100*(dev>5).mean():.1f}%")
    for k in (1, 10, 20):
        p0 = np.array([circ_med90(B[0][max(0, t-k+1):t+1]) for t in range(len(stems))])
        p1 = np.array([circ_med90(B[1][max(0, t-k+1):t+1]) for t in range(len(stems))])
        e = np.abs(wrap(p0 - p1, 90)); e = np.where(np.isfinite(e), e, np.inf)
        print(f"  relative heading, k={k:2d}: no estimate {100*np.isinf(e).mean():.1f}%, median {np.median(e):.2f} deg, <0.41 {100*(e<0.41).mean():5.1f}%, "
              f"<1.15 {100*(e<1.15).mean():5.1f}%, <2 {100*(e<2).mean():5.1f}%, >5 {100*(e>5).mean():5.1f}%")
