"""Can the ego see the partner robot itself?

The project's central tension is that the partner is useful precisely because it
looks where the ego cannot, which is also why there is no shared observation to
align with. Seeing the partner *robot* would break that tension for free: it is
one correspondence that needs no scene overlap at all.

So this measures three things, in increasing order of how much they commit to:

  1. geometry   - is the partner inside the ego's sensing wedge at all?
  2. occlusion  - is the line to it clear of the ego's own returns?
  3. returns    - are there actually LiDAR points where the partner should be?

Test 3 decides whether this is usable in THIS dataset or only in principle: the
partner robot is not in the annotations (the classes are chairs, extinguishers
and so on), so whether the simulator rendered it is an open question.
"""
import argparse, glob, math, os, sys
from collections import defaultdict

import numpy as np
import yaml

sys.path.insert(0, "/nfs/home/charlie/InCoP")
from opencood.utils import pcd_utils

LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
ROOT = "/nfs/home/charlie/InCoP/dataset/IsaacSimOPV2V_hospital"
# cav_lidar_range from the training config: a forward-only BEV crop.
XMIN, YMIN, XMAX, YMAX = 0.0, -11.2, 22.4, 11.2
FOV_DEG = 89.0            # IsaacSimGenScript/visualize.py, config default 90.0
CELL = 0.2                # occupancy cell for the ray walk
# Measured from the clouds: the floor sits at z about -0.4 (45.9% of returns)
# and the ceiling at +2.6 (38.4%). Only the band between them is an obstacle;
# without this filter the floor marks every cell occupied and every ray reads
# as blocked, which is exactly what the first run reported.
Z_LO, Z_HI = -0.20, 2.30
Z_FLOOR_HI = -0.25        # below this is floor, used to test "inside the room"
NEAR_R = 0.5              # radius around the partner counted as "its returns"
SKIRT = 0.8               # ray stops this far short, so the partner's own
                          # returns cannot count as the thing occluding it


def pose_xy_yaw(p):
    """lidar_pose is [x, y, z, roll, yaw, pitch] with angles in degrees."""
    return float(p[0]), float(p[1]), math.radians(float(p[4]))


def to_ego(px, py, ex, ey, eyaw):
    dx, dy = px - ex, py - ey
    c, s = math.cos(-eyaw), math.sin(-eyaw)
    return c * dx - s * dy, s * dx + c * dy


def in_wedge(x, y):
    if x <= 0.05:
        return False
    if not (XMIN <= x < XMAX and YMIN <= y < YMAX):
        return False
    return abs(math.degrees(math.atan2(y, x))) <= FOV_DEG / 2.0


def obstacle_band(points):
    return points[(points[:, 2] > Z_LO) & (points[:, 2] < Z_HI)]


def floor_band(points):
    return points[points[:, 2] <= Z_FLOOR_HI]


def occupancy(points):
    """Cells of the ego's own cloud that returned something, in the ego frame."""
    nx = int(math.ceil((XMAX - XMIN) / CELL))
    ny = int(math.ceil((YMAX - YMIN) / CELL))
    occ = np.zeros((nx, ny), dtype=bool)
    p = points[(points[:, 0] >= XMIN) & (points[:, 0] < XMAX)
               & (points[:, 1] >= YMIN) & (points[:, 1] < YMAX)]
    if len(p):
        ix = ((p[:, 0] - XMIN) / CELL).astype(int)
        iy = ((p[:, 1] - YMIN) / CELL).astype(int)
        occ[ix, iy] = True
    return occ


def ray_blocked(occ, tx, ty):
    """Walk from the ego origin toward (tx, ty); is anything in the way?"""
    d = math.hypot(tx, ty)
    stop = max(0.0, d - SKIRT)
    n = max(1, int(stop / (CELL * 0.5)))
    for k in range(1, n + 1):
        r = stop * k / n
        x, y = tx * r / d, ty * r / d
        ix = int((x - XMIN) / CELL)
        iy = int((y - YMIN) / CELL)
        if 0 <= ix < occ.shape[0] and 0 <= iy < occ.shape[1] and occ[ix, iy]:
            return True
    return False


def count_near(points, tx, ty, r=NEAR_R):
    d = np.hypot(points[:, 0] - tx, points[:, 1] - ty)
    return int((d <= r).sum())


def control_positions(tx, ty):
    """Same range, different bearings inside the wedge - the null for test 3.

    If the simulator renders the partner, the count at the partner's bearing
    should stand out against these. The caller keeps only controls where the
    ego also saw the floor, so a control cannot be empty merely because it is
    behind a wall or outside the room.
    """
    d = math.hypot(tx, ty)
    out = []
    for deg in (-35, -25, -15, 15, 25, 35):
        a = math.atan2(ty, tx) + math.radians(deg)
        if abs(math.degrees(a)) <= FOV_DEG / 2.0:
            cx, cy = d * math.cos(a), d * math.sin(a)
            if in_wedge(cx, cy):
                out.append((cx, cy))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_frames_per_case", type=int, default=0,
                    help="0 = every frame")
    ap.add_argument("--pcd_frames", type=int, default=400,
                    help="how many in-wedge frames to run tests 2 and 3 on")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    base = os.path.join(ROOT, args.split)
    rows = []
    for case in sorted(os.listdir(base)):
        cdir = os.path.join(base, case)
        if not os.path.isdir(cdir):
            continue
        agents = sorted(d for d in os.listdir(cdir)
                        if os.path.isdir(os.path.join(cdir, d)))
        if len(agents) < 2:
            continue
        a0, a1 = agents[0], agents[1]
        frames = sorted(os.path.basename(f)[:-5]
                        for f in glob.glob(os.path.join(cdir, a0, "*.yaml")))
        if args.max_frames_per_case:
            frames = frames[:args.max_frames_per_case]
        for fr in frames:
            try:
                with open(os.path.join(cdir, a0, fr + ".yaml")) as fh:
                    y0 = yaml.load(fh, Loader=LOADER)
                with open(os.path.join(cdir, a1, fr + ".yaml")) as fh:
                    y1 = yaml.load(fh, Loader=LOADER)
            except Exception:
                continue
            if "lidar_pose" not in y0 or "lidar_pose" not in y1:
                continue
            ex, ey, eyaw = pose_xy_yaw(y0["lidar_pose"])
            px, py, pyaw = pose_xy_yaw(y1["lidar_pose"])
            # both directions: ego->partner and partner->ego are different
            # questions, because the wedge is forward-only
            fx, fy = to_ego(px, py, ex, ey, eyaw)
            bx, by = to_ego(ex, ey, px, py, pyaw)
            rows.append({
                "case": case, "frame": fr,
                "dist": math.hypot(px - ex, py - ey),
                "fx": fx, "fy": fy,
                "fwd": in_wedge(fx, fy), "bwd": in_wedge(bx, by),
                "bearing": math.degrees(math.atan2(fy, fx)),
                "pcd": os.path.join(cdir, a0, fr + ".pcd"),
            })

    n = len(rows)
    fwd = sum(r["fwd"] for r in rows)
    bwd = sum(r["bwd"] for r in rows)
    either = sum(r["fwd"] or r["bwd"] for r in rows)
    both = sum(r["fwd"] and r["bwd"] for r in rows)
    print(f"=== 1. geometry: is the partner inside the sensing wedge? "
          f"({args.split}, {n} frames) ===")
    print(f"  ego sees partner's position      {fwd:5d}  {100*fwd/n:5.1f}%")
    print(f"  partner sees ego's position      {bwd:5d}  {100*bwd/n:5.1f}%")
    print(f"  at least one direction           {either:5d}  {100*either/n:5.1f}%")
    print(f"  both directions                  {both:5d}  {100*both/n:5.1f}%")
    print(f"  (wedge = {FOV_DEG} deg forward, x in [{XMIN},{XMAX}], "
          f"y in [{YMIN},{YMAX}])")

    d = np.array([r["dist"] for r in rows])
    print(f"\n  inter-robot distance: median {np.median(d):.2f} m, "
          f"range {d.min():.2f}-{d.max():.2f} m")
    inw = [r for r in rows if r["fwd"]]
    if inw:
        dd = np.array([r["dist"] for r in inw])
        bb = np.abs([r["bearing"] for r in inw])
        print(f"  when in the wedge: distance median {np.median(dd):.2f} m, "
              f"|bearing| median {np.median(bb):.1f} deg")

    # --- tests 2 and 3, on a sample, because they read point clouds
    sample = inw[:: max(1, len(inw) // max(1, args.pcd_frames))][:args.pcd_frames]
    print(f"\n=== 2/3. occlusion and returns, on {len(sample)} in-wedge frames ===")
    blocked = 0
    near_at_partner, near_at_control, zprof = [], [], []
    used = 0
    floor_seen_at_partner = 0
    for r in sample:
        if not os.path.exists(r["pcd"]):
            continue
        try:
            pts = pcd_utils.pcd_to_np(r["pcd"])[:, :3]
        except Exception:
            continue
        used += 1
        obst = obstacle_band(pts)
        flr = floor_band(pts)
        occ = occupancy(obst)
        if ray_blocked(occ, r["fx"], r["fy"]):
            blocked += 1
        near_at_partner.append(count_near(obst, r["fx"], r["fy"]))
        zprof.extend(pts[np.hypot(pts[:, 0] - r["fx"],
                                  pts[:, 1] - r["fy"]) <= NEAR_R, 2].tolist())
        # keep only controls the ego could actually see the floor at
        ctrl = [(cx, cy) for cx, cy in control_positions(r["fx"], r["fy"])
                if count_near(flr, cx, cy) >= 3]
        if ctrl:
            near_at_control.append(float(np.mean(
                [count_near(obst, cx, cy) for cx, cy in ctrl])))
        if count_near(flr, r["fx"], r["fy"]) >= 3:
            floor_seen_at_partner += 1
    if not used:
        print("  no point clouds read")
        return
    ap_ = np.array(near_at_partner, dtype=float)
    ac = np.array(near_at_control, dtype=float) if near_at_control else np.zeros(0)
    print(f"  ego also saw the FLOOR at the partner's position  "
          f"{floor_seen_at_partner}/{used}  {100*floor_seen_at_partner/used:.1f}%"
          f"   (a robot standing there would block the floor)")
    print(f"  line of sight blocked by the ego's own returns   "
          f"{blocked}/{used}  {100*blocked/used:.1f}%")
    print(f"  clear line of sight                              "
          f"{used-blocked}/{used}  {100*(used-blocked)/used:.1f}%")
    print(f"\n  LiDAR points within {NEAR_R} m of where the partner is:")
    print(f"    at the partner's position   median {np.median(ap_):6.1f}   "
          f"mean {ap_.mean():7.1f}   zero in {100*(ap_==0).mean():5.1f}% of frames")
    if len(ac):
        print(f"    at control bearings         median {np.median(ac):6.1f}   "
              f"mean {ac.mean():7.1f}   zero in {100*(ac==0).mean():5.1f}% of frames")
        print(f"    ratio of medians            "
              f"{(np.median(ap_)+1e-9)/(np.median(ac)+1e-9):.2f}x")
    if zprof:
        z = np.array(zprof)
        print(f"\n  height profile of the {len(z)} returns within "
              f"{NEAR_R} m of the partner:")
        for lo in np.arange(-0.5, 2.75, 0.25):
            c = int(((z >= lo) & (z < lo + 0.25)).sum())
            if c:
                print(f"    z {lo:+.2f}..{lo+0.25:+.2f}  {c:7d}  "
                      f"{100*c/len(z):5.1f}%  " + "#" * int(50 * c / max(1, c)) 
                      if False else
                      f"    z {lo:+.2f}..{lo+0.25:+.2f}  {c:7d}  {100*c/len(z):5.1f}%")
        print("    (a rendered robot would show a bump away from the floor and")
        print("     ceiling bands; walls alone give a flat spread)")
    print("\n  Reading this: a clearly higher count at the partner's position")
    print("  than at control bearings means the simulator rendered the partner")
    print("  robot, so the ego could in principle detect and use it. Similar")
    print("  counts mean the returns are just the room, and the partner is")
    print("  invisible in this dataset however good the geometry is.")

    out = os.path.join(args.out, f"partner_visibility_{args.split}.csv")
    with open(out, "w") as fh:
        fh.write("case,frame,dist_m,bearing_deg,ego_sees_partner,partner_sees_ego\n")
        for r in rows:
            fh.write(f"{r['case']},{r['frame']},{r['dist']:.4f},"
                     f"{r['bearing']:.2f},{int(r['fwd'])},{int(r['bwd'])}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
