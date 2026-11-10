"""If the partner reports an object the ego cannot see together with a radius,
is an ESTIMATED radius better than one fixed radius for everything?

No detector and no GPU. The partner's reports are the annotated objects that
only the partner sees (inside the ego's BEV window, absent from the ego's own
list). A pose error on the partner moves each report; the question is whether
the true object still falls inside the reported circle (coverage) and how big
the circles have to be to achieve that (size).

Every method is calibrated to the same target, 90% coverage, on half of the
episodes and graded on the other half, so the only thing that differs is how
the radius is chosen:

  fixed          one radius rule for everything - no uncertainty estimate
  frame score    radius picked from this frame's free-space contradiction
  window score   radius picked from the median score of the previous 20 frames
  oracle level   radius picked knowing the true noise level
  oracle object  radius = the true displacement (unreachable lower bound)

Noise: a planar error on the partner's pose, sigma_pos metres on x and y and
10 * sigma_pos degrees on yaw, redrawn each frame, as in the benchmark runs.
"""
import glob, math, os, sys, time
import numpy as np
import yaml
sys.path.insert(0, "/nfs/home/charlie/InCoP")
from opencood.utils.freespace_uncertainty import FreeSpaceContradiction
from opencood.utils.transformation_utils import x_to_world

LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
ROOT = "/nfs/home/charlie/InCoP/dataset/IsaacSimOPV2V_hospital/test"
HERE = os.path.dirname(os.path.abspath(__file__))
RANGE = [0.0, -11.2, -1, 22.4, 11.2, 3]
LEVELS = [0.0, 0.1, 0.2, 0.3, 0.6, 0.8]
TARGET, WINDOW = 0.90, 20
rng = np.random.default_rng(0)
est = FreeSpaceContradiction(RANGE)

def offset_world(dx, dy, dyaw_deg, pivot):
    th = math.radians(dyaw_deg); c, s = math.cos(th), math.sin(th)
    T = np.eye(4); T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    px, py = pivot
    T[0, 3] = px - (c * px - s * py) + dx
    T[1, 3] = py - (s * px + c * py) + dy
    return T

def in_window(loc, pose):
    yaw = math.radians(pose[4]); dx, dy = loc[0] - pose[0], loc[1] - pose[1]
    c, s = math.cos(-yaw), math.sin(-yaw)
    x, y = c * dx - s * dy, s * dx + c * dy
    return RANGE[0] <= x <= RANGE[3] and RANGE[1] <= y <= RANGE[4]

def shape(r):
    """How displacement grows with the object's range from the partner, up to a
    scale: translation error on two axes plus yaw error times range, with the
    benchmark's 10 degrees per metre ratio. Only the scale is estimated."""
    return np.sqrt(2.0 + (math.radians(10.0) * r) ** 2)

# ---- pass 1: per frame and level, the score and the displaced reports
rec = []          # one row per (level, episode, t, object)
t0 = time.time()
cases = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
for ci, case in enumerate(cases):
    d0, d1 = os.path.join(ROOT, case, "0"), os.path.join(ROOT, case, "1")
    stems = sorted(os.path.basename(f)[:-5] for f in glob.glob(d0 + "/*.yaml")
                   if os.path.exists(os.path.join(d1, os.path.basename(f))))
    for t, st in enumerate(stems):
        with open(os.path.join(d0, st + ".yaml")) as fh:
            a = yaml.load(fh, Loader=LOADER)
        with open(os.path.join(d1, st + ".yaml")) as fh:
            b = yaml.load(fh, Loader=LOADER)
        ego_ids = set((a.get("vehicles") or {}).keys())
        par = b.get("vehicles") or {}
        n_shared = len(ego_ids & set(par.keys()))
        only = [(k, v["location"]) for k, v in par.items()
                if k not in ego_ids and in_window(v["location"], a["lidar_pose"])]
        Te, Tp = x_to_world(a["lidar_pose"]), x_to_world(b["lidar_pose"])
        Tei = np.linalg.inv(Te)
        piv = (b["lidar_pose"][0], b["lidar_pose"][1])
        e_pcd, p_pcd = os.path.join(d0, st + ".pcd"), os.path.join(d1, st + ".pcd")
        for li, sig in enumerate(LEVELS):
            dx, dy = rng.normal(0, sig, 2) if sig else (0.0, 0.0)
            dyaw = rng.normal(0, 10.0 * sig) if sig else 0.0
            Off = offset_world(dx, dy, dyaw, piv)
            score, n = est.score(e_pcd, p_pcd, Tei @ Off @ Tp)
            for k, loc in only:
                x = np.array([loc[0], loc[1], 0.0, 1.0])
                d = float(np.hypot(*((Off @ x)[:2] - x[:2])))
                r = math.hypot(loc[0] - piv[0], loc[1] - piv[1])
                rec.append((li, ci, t, d, r, score if n else np.nan, n_shared))
print(f"{len(rec)} partner-only reports from {len(cases)} episodes "
      f"in {time.time()-t0:.0f}s", flush=True)

R = np.array(rec, dtype=float)
lvl, epi, tt, d, r, score, nsh = R.T
g = shape(r)
ratio = d / g                       # what conformal calibrates

# window score: median of the previous WINDOW frames' scores, in order, within
# one run at one noise level (rolling in order, never resampled)
frame_key = {}
for i in range(len(R)):
    frame_key.setdefault((int(lvl[i]), int(epi[i]), int(tt[i])), score[i])
win = {}
for li in range(len(LEVELS)):
    keys = sorted(k for k in frame_key if k[0] == li)
    hist = []
    for k in keys:
        prev = [s for s in hist[-WINDOW:] if not np.isnan(s)]
        win[k] = float(np.median(prev)) if len(prev) >= 5 else np.nan
        hist.append(frame_key[k])
wscore = np.array([win[(int(lvl[i]), int(epi[i]), int(tt[i]))] for i in range(len(R))])

cal = (epi % 2 == 0)
tst = ~cal

def cq(x):
    """Conformal quantile with the finite-sample correction."""
    x = np.sort(x); n = len(x)
    if n == 0:
        return np.inf
    return float(x[min(n - 1, int(math.ceil((n + 1) * TARGET)) - 1)])

def binned(sig_cal, sig_all, nb=5):
    """A multiplier per bin of the signal; nan gets its own bin."""
    ok = ~np.isnan(sig_cal)
    edges = np.quantile(sig_cal[ok], np.linspace(0, 1, nb + 1)[1:-1])
    def which(s):
        return np.where(np.isnan(s), nb, np.digitize(s, edges))
    bc, ba = which(sig_cal), which(sig_all)
    mult = np.array([cq(ratio[cal][bc == b]) for b in range(nb + 1)])
    return mult[ba]

mult = {
    "fixed (no estimate)": np.full(len(R), cq(ratio[cal])),
    "frame score": binned(score[cal], score),
    "window score (20 frames)": binned(wscore[cal], wscore),
    "oracle: true noise level": np.array(
        [cq(ratio[cal & (lvl == l)]) for l in range(len(LEVELS))])[lvl.astype(int)],
}
radius = {k: m * g for k, m in mult.items()}
radius["oracle: per object"] = d.copy()

print(f"\ncalibration reports {cal.sum()}, test reports {tst.sum()}; "
      f"target coverage {100*TARGET:.0f}%")
print(f"\n{'method':28s} {'coverage':>9} {'mean radius':>12} {'median':>8} "
      f"{'<=0.5 m':>8}")
for k, rad in radius.items():
    c = (d[tst] <= rad[tst] + 1e-12).mean()
    print(f"{k:28s} {100*c:8.1f}% {rad[tst].mean():10.2f} m {np.median(rad[tst]):6.2f} m "
          f"{100*(rad[tst] <= 0.5).mean():7.1f}%")

print("\ncoverage by true noise level (honest calibration means ~90% in EVERY column)")
print(f"{'method':28s}" + "".join(f"{s:>8.1f}" for s in LEVELS))
for k, rad in radius.items():
    if k.startswith("oracle: per"):
        continue
    row = [100 * (d[tst & (lvl == l)] <= rad[tst & (lvl == l)] + 1e-12).mean()
           for l in range(len(LEVELS))]
    print(f"{k:28s}" + "".join(f"{v:7.1f}%" for v in row))
print("\nmean radius by true noise level (m)")
for k, rad in radius.items():
    row = [rad[tst & (lvl == l)].mean() for l in range(len(LEVELS))]
    print(f"{k:28s}" + "".join(f"{v:8.2f}" for v in row))

z = tst & (nsh == 0)
print(f"\nframes with zero co-visible objects only ({z.sum()} test reports):")
for k, rad in radius.items():
    print(f"  {k:28s} coverage {100*(d[z] <= rad[z] + 1e-12).mean():5.1f}%   "
          f"mean radius {rad[z].mean():.2f} m")
np.savez(os.path.join(HERE, "reports.npz"), R=R, wscore=wscore)
