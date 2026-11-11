"""Recheck: is the other robot really absent from the LiDAR?

The earlier test used an obstacle band starting at z = -0.20 (sensor frame).
The camera images show the robots ARE rendered, and the robot is low: its body
sits between the floor (z about -0.47) and roughly the sensor height. A band
starting at -0.20 throws most of that away as floor. This looks again with a
fine height histogram just above the floor.
"""
import glob, math, os, sys
import numpy as np, yaml
sys.path.insert(0, "/nfs/home/charlie/InCoP")
from opencood.utils import pcd_utils
LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
FOV, R_NEAR = 89.0, 0.45

def rel(p_from, p_to):
    yaw = math.radians(p_from[4]); dx, dy = p_to[0]-p_from[0], p_to[1]-p_from[1]
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c*dx - s*dy, s*dx + c*dy

def visible(x, y):
    return x > 0.3 and math.hypot(x, y) < 20 and abs(math.degrees(math.atan2(y, x))) <= FOV/2 - 3

def run(root, label, max_frames=400):
    edges = np.round(np.arange(-0.50, 0.31, 0.05), 2)
    at, ctrl, n = np.zeros(len(edges)-1), np.zeros(len(edges)-1), 0
    hit_at, hit_ctrl, dists = [], [], []
    files = sorted(glob.glob(os.path.join(root, "*", "0", "*.yaml")))
    step = max(1, len(files) // 1500)
    for f0 in files[::step]:
        f1 = f0.replace(os.sep + "0" + os.sep, os.sep + "1" + os.sep)
        if not os.path.exists(f1):
            continue
        a = yaml.load(open(f0), Loader=LOADER); b = yaml.load(open(f1), Loader=LOADER)
        for me, other, fme in ((a, b, f0), (b, a, f1)):
            x, y = rel(me["lidar_pose"], other["lidar_pose"])
            if not visible(x, y):
                continue
            pts = pcd_utils.pcd_to_np(fme[:-5] + ".pcd")[:, :3]
            d = math.hypot(x, y)
            def grab(cx, cy):
                m = np.hypot(pts[:, 0]-cx, pts[:, 1]-cy) <= R_NEAR
                return pts[m, 2]
            za = grab(x, y)
            # control: same range, mirrored bearing, must also be in the wedge
            zc = grab(x, -y) if abs(y) > 2*R_NEAR else grab(d*math.cos(math.radians(30)), d*math.sin(math.radians(30)))
            at += np.histogram(za, edges)[0]; ctrl += np.histogram(zc, edges)[0]
            hit_at.append(int(((za > -0.40) & (za < 0.25)).sum()))
            hit_ctrl.append(int(((zc > -0.40) & (zc < 0.25)).sum()))
            dists.append(d); n += 1
            if n >= max_frames:
                break
        if n >= max_frames:
            break
    hit_at, hit_ctrl = np.array(hit_at), np.array(hit_ctrl)
    print(f"\n=== {label}: {n} views where the other robot is inside the wedge "
          f"(distance median {np.median(dists):.2f} m) ===")
    print(f"  returns between 7 cm above the floor and z=+0.25, within {R_NEAR} m:")
    print(f"    at the other robot : median {np.median(hit_at):5.0f}  mean {hit_at.mean():7.1f}  none in {100*(hit_at==0).mean():5.1f}%")
    print(f"    at control position: median {np.median(hit_ctrl):5.0f}  mean {hit_ctrl.mean():7.1f}  none in {100*(hit_ctrl==0).mean():5.1f}%")
    print("  height histogram (sensor frame; floor is near -0.47):")
    for lo, x1, x2 in zip(edges[:-1], at, ctrl):
        print(f"    z {lo:+.2f}  robot {int(x1):7d}  control {int(x2):7d}")

run("dataset/IsaacSimOPV2V_hospital_highoverlap/train", "high-overlap train")
run("dataset/IsaacSimOPV2V_hospital/test", "low-overlap test")
