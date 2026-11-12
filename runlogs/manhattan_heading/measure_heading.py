"""Can each robot read its heading off the building, with no shared objects?

Indoor buildings are mostly built on two perpendicular directions (walls,
corridors, shelf rows). If each robot measures, from its own LiDAR alone, the
angle of those directions in its own frame (alpha), then

    alpha_ego - alpha_partner = yaw_partner - yaw_ego   (mod 90 deg)

which is the relative heading the fusion needs, obtained without the two robots
seeing anything in common. The 90-degree ambiguity is resolved by the reported
(noisy) heading, whose error here is a few degrees, far below 45.

This measures, per frame, how far that estimate is from the truth (GT poses),
and whether a confidence computed from the scan says when to believe it.
CPU only.

Per scan:
  1. keep the wall band (sensor-frame z between -0.30 and 2.25 m: above the
     floor at about -0.47, below the ceiling at about 2.4)
  2. collapse to 5 cm xy cells, so a wall counts by its length, not by how many
     laser rings hit it
  3. rotate the cells by each candidate angle in [0, 90) and project them on the
     two rotated axes; walls aligned with the axes pile up into a few bins, so
     the sum of squared bin counts peaks at the building's angle
     (0.5 deg grid, then 0.05 deg refinement averaged over 3, 4, 5 cm bins)
  4. confidence: peak score over the median score (how much structure there is)
     and peak over the best score more than 3 deg away (is there a rival angle)

The estimator was chosen on 8 validation cases (compare_variants.py) against a
local-line PCA variant: median relative error 0.50 deg against 0.98 deg.
"""
import argparse, glob, math, os, sys
from multiprocessing import Pool
import numpy as np, yaml

sys.path.insert(0, "/nfs/home/charlie/InCoP")
from opencood.utils import pcd_utils

LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
Z_LO, Z_HI, CELL = -0.30, 2.25, 0.05


def wrap(a, period):
    """wrap to [-period/2, period/2)"""
    return (a + period / 2) % period - period / 2


def _score(xy, th, b):
    c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
    u = xy[:, 0] * c + xy[:, 1] * s
    v = -xy[:, 0] * s + xy[:, 1] * c
    hu = np.bincount(np.floor((u - u.min()) / b).astype(np.int64)).astype(float)
    hv = np.bincount(np.floor((v - v.min()) / b).astype(np.int64)).astype(float)
    return float((hu ** 2).sum() + (hv ** 2).sum())


COARSE = np.arange(0, 90, 0.5)


def manhattan_angle(pts):
    band = pts[(pts[:, 2] > Z_LO) & (pts[:, 2] < Z_HI), :2]
    out = dict(n_band=len(band), n_cells=0, alpha=np.nan, peak_over_median=np.nan,
               peak_over_rival=np.nan)
    if len(band) < 50:
        return out
    xy = (np.unique(np.floor(band / CELL).astype(np.int64), axis=0) + 0.5) * CELL
    sc = np.array([_score(xy, t, CELL) for t in COARSE])
    i = int(np.argmax(sc))
    fine = COARSE[i] + np.arange(-0.75, 0.76, 0.05)
    scf = np.array([np.mean([_score(xy, t, b) for b in (0.03, 0.04, 0.05)]) for t in fine])
    far = np.abs(wrap(COARSE - COARSE[i], 90)) > 3.0
    out.update(n_cells=len(xy), alpha=float(fine[int(np.argmax(scf))] % 90),
               peak_over_median=float(sc[i] / np.median(sc)),
               peak_over_rival=float(sc[i] / sc[far].max()))
    return out


def one_case(args):
    root, case, split = args
    d0, d1 = os.path.join(root, case, "0"), os.path.join(root, case, "1")
    rows = []
    stems = sorted(os.path.basename(f)[:-5] for f in glob.glob(d0 + "/*.yaml")
                   if os.path.exists(os.path.join(d1, os.path.basename(f))))
    for t, st in enumerate(stems):
        meta, est = [], []
        for d in (d0, d1):
            with open(os.path.join(d, st + ".yaml")) as fh:
                y = yaml.load(fh, Loader=LOADER)
            meta.append(y)
            est.append(manhattan_angle(pcd_utils.pcd_to_np(os.path.join(d, st + ".pcd"))))
        E = set((meta[0].get("vehicles") or {}).keys())
        P = set((meta[1].get("vehicles") or {}).keys())
        p0, p1 = meta[0]["lidar_pose"], meta[1]["lidar_pose"]
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        rows.append(dict(split=split, case=case, scene=case.split("__")[0], stem=st, t=t,
                         yaw0=p0[4], yaw1=p1[4], dist=dist, covis=len(E & P),
                         **{f"{k}0": v for k, v in est[0].items()},
                         **{f"{k}1": v for k, v in est[1].items()}))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["IsaacSimOPV2V_hospital/test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "heading.csv"))
    a = ap.parse_args()
    jobs = []
    for sp in a.splits:
        root = os.path.join("/nfs/home/charlie/InCoP/dataset", sp)
        cases = sorted(c for c in os.listdir(root) if os.path.isdir(os.path.join(root, c, "0")))
        if a.max_cases:
            cases = cases[:a.max_cases]
        jobs += [(root, c, sp) for c in cases]
    import csv
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one_case, jobs)):
            rows += r
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(jobs)} cases", flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} frames to {a.out}")
