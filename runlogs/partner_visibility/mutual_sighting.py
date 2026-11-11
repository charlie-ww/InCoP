"""Who can see whom, measured from the LiDAR returns themselves.

Seeing the other robot gives its position in your own frame, independent of
either robot's global localisation error. One-sided sighting fixes the relative
POSITION. If both robots see each other, the two bearings also fix the relative
HEADING (yaw = bearing A->B - bearing B->A + 180 deg), so the full planar pose
is determined. This counts how often each case happens.
"""
import glob, math, os, sys
import numpy as np, yaml
sys.path.insert(0, "/nfs/home/charlie/InCoP")
from opencood.utils import pcd_utils
L = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
ROOT = "dataset/IsaacSimOPV2V_hospital/test"
MIN_PTS = 3

def rel(p_from, p_to):
    yaw = math.radians(p_from[4]); dx, dy = p_to[0]-p_from[0], p_to[1]-p_from[1]
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c*dx - s*dy, s*dx + c*dy

def hits(pcd_path, x, y):
    p = pcd_utils.pcd_to_np(pcd_path)[:, :3]
    near = np.hypot(p[:, 0]-x, p[:, 1]-y) <= 0.45
    body = (p[:, 2] > -0.40) & (p[:, 2] < 0.05)
    return int((near & body).sum())

rows = []
for f0 in sorted(glob.glob(f"{ROOT}/*/0/*.yaml")):
    f1 = f0.replace("/0/", "/1/")
    if not os.path.exists(f1):
        continue
    a = yaml.load(open(f0), Loader=L)["lidar_pose"]; b = yaml.load(open(f1), Loader=L)["lidar_pose"]
    ex, ey = rel(a, b)          # partner in ego frame
    px, py = rel(b, a)          # ego in partner frame
    e_sees = hits(f0[:-5] + ".pcd", ex, ey) >= MIN_PTS
    p_sees = hits(f1[:-5] + ".pcd", px, py) >= MIN_PTS
    rows.append((e_sees, p_sees, math.hypot(ex, ey)))
r = np.array(rows, dtype=float)
n = len(r); E, P = r[:, 0] > 0, r[:, 1] > 0
print(f"{n} test frames, a robot counts as seen with >= {MIN_PTS} LiDAR returns on its body")
print(f"  ego sees partner              {100*E.mean():5.1f}%")
print(f"  partner sees ego              {100*P.mean():5.1f}%")
print(f"  only the partner sees         {100*(P & ~E).mean():5.1f}%   <- what one-sided ego checking misses")
print(f"  at least one sees the other   {100*(E | P).mean():5.1f}%   -> relative POSITION measurable")
print(f"  both see each other           {100*(E & P).mean():5.1f}%   -> relative HEADING measurable too")
print(f"  neither                       {100*(~E & ~P).mean():5.1f}%")
