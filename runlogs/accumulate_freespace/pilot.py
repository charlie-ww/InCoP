"""Pilot: if the pose error is the SAME across frames, does weak evidence add up?

The benchmark's noise is drawn fresh every frame (0.2 m), while the robots move
0.08 m per frame. A real localisation error cannot jump like that; it is close
to constant over an episode. If it is constant, every frame is evidence about
ONE unknown offset, and the free-space contradiction - too weak to act on in a
single frame - can be summed.

Test: the truth is offset 0. Score a grid of candidate world-frame offsets on
every frame, pool the contradiction over the first N frames of each episode,
and see how far the best candidate is from the truth as N grows. No ground
truth is used to pick the candidate; it is used only to grade it.
"""
import glob, math, os, sys, time
import numpy as np
import yaml
sys.path.insert(0, "/nfs/home/charlie/InCoP")
from opencood.utils.freespace_uncertainty import FreeSpaceContradiction
from opencood.utils.transformation_utils import x_to_world

LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
ROOT = "/nfs/home/charlie/InCoP/dataset/IsaacSimOPV2V_hospital/test"
RANGE = [0.0, -11.2, -1, 22.4, 11.2, 3]
SHIFTS = np.round(np.arange(-0.6, 0.601, 0.1), 2)          # metres
YAWS = np.arange(-6.0, 6.01, 1.0)                          # degrees
MAXEP = int(sys.argv[1]) if len(sys.argv) > 1 else 999

est = FreeSpaceContradiction(RANGE)

def offset_world(dx, dy, dyaw_deg, pivot):
    """Planar offset of the partner's world pose, rotating about the partner."""
    th = math.radians(dyaw_deg)
    c, s = math.cos(th), math.sin(th)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    px, py = pivot
    T[0, 3] = px - (c * px - s * py) + dx
    T[1, 3] = py - (s * px + c * py) + dy
    return T

cases = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))[:MAXEP]
G = len(SHIFTS)
res_xy, res_yaw = {}, {}
t0 = time.time()
for case in cases:
    d0, d1 = os.path.join(ROOT, case, "0"), os.path.join(ROOT, case, "1")
    stems = sorted(os.path.basename(f)[:-5] for f in glob.glob(d0 + "/*.yaml")
                   if os.path.exists(os.path.join(d1, os.path.basename(f))))
    hits_xy = np.zeros((len(stems), G, G)); tot_xy = np.zeros_like(hits_xy)
    hits_yw = np.zeros((len(stems), len(YAWS))); tot_yw = np.zeros_like(hits_yw)
    for t, st in enumerate(stems):
        with open(os.path.join(d0, st + ".yaml")) as fh:
            a = yaml.load(fh, Loader=LOADER)
        with open(os.path.join(d1, st + ".yaml")) as fh:
            b = yaml.load(fh, Loader=LOADER)
        Te, Tp = x_to_world(a["lidar_pose"]), x_to_world(b["lidar_pose"])
        Tei = np.linalg.inv(Te)
        piv = (b["lidar_pose"][0], b["lidar_pose"][1])
        e_pcd, p_pcd = os.path.join(d0, st + ".pcd"), os.path.join(d1, st + ".pcd")
        for i, dx in enumerate(SHIFTS):
            for j, dy in enumerate(SHIFTS):
                s, n = est.score(e_pcd, p_pcd, Tei @ offset_world(dx, dy, 0.0, piv) @ Tp)
                if n:
                    hits_xy[t, i, j], tot_xy[t, i, j] = s * n, n
        for k, yw in enumerate(YAWS):
            s, n = est.score(e_pcd, p_pcd, Tei @ offset_world(0.0, 0.0, yw, piv) @ Tp)
            if n:
                hits_yw[t, k], tot_yw[t, k] = s * n, n
    res_xy[case] = (hits_xy, tot_xy); res_yaw[case] = (hits_yw, tot_yw)
print(f"scored {len(cases)} episodes in {time.time()-t0:.0f}s", flush=True)

def best_xy(h, n):
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(n > 0, h / n, np.nan)
    if np.all(np.isnan(c)):
        return np.nan
    m = np.nanmin(c)
    ii, jj = np.where(c <= m + 1e-12)           # ties: average them
    return math.hypot(SHIFTS[ii].mean(), SHIFTS[jj].mean())

def best_yaw(h, n):
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(n > 0, h / n, np.nan)
    if np.all(np.isnan(c)):
        return np.nan
    m = np.nanmin(c)
    return abs(YAWS[np.where(c <= m + 1e-12)[0]].mean())

print("\nerror of the best candidate vs how many frames are pooled")
print("(truth is 0; chance for the translation grid is about 0.46 m, yaw 3.2 deg)")
print(f"{'frames pooled':>14} {'transl. median':>15} {'<=0.16 m':>9} {'<=0.30 m':>9}"
      f" {'yaw median':>11} {'<=1.15 deg':>11} {'episodes':>9}")
for N in (1, 3, 5, 10, 20, 30, 999):
    ex, ey = [], []
    for case in cases:
        hx, nx = res_xy[case]; hy, ny = res_yaw[case]
        if N != 999 and len(hx) < N:
            continue
        k = len(hx) if N == 999 else N
        ex.append(best_xy(hx[:k].sum(0), nx[:k].sum(0)))
        ey.append(best_yaw(hy[:k].sum(0), ny[:k].sum(0)))
    ex, ey = np.array(ex), np.array(ey)
    ex, ey = ex[~np.isnan(ex)], ey[~np.isnan(ey)]
    lab = "whole episode" if N == 999 else str(N)
    print(f"{lab:>14} {np.median(ex):13.3f} m {100*(ex<=0.16).mean():8.1f}% "
          f"{100*(ex<=0.30).mean():8.1f}% {np.median(ey):9.2f}deg "
          f"{100*(ey<=1.15).mean():10.1f}% {len(ex):>9}")

np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot_costs.npz"),
         shifts=SHIFTS, yaws=YAWS,
         **{f"xy_h_{c}": res_xy[c][0] for c in cases},
         **{f"xy_n_{c}": res_xy[c][1] for c in cases},
         **{f"yw_h_{c}": res_yaw[c][0] for c in cases},
         **{f"yw_n_{c}": res_yaw[c][1] for c in cases})
