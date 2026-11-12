"""Pick the estimator on a few VALIDATION cases (warehouse/office), not on test."""
import sys, glob, math, os, numpy as np, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_heading as M
from scipy.spatial import cKDTree
from opencood.utils import pcd_utils

def proj_angle(pts, zlo, zhi, cell=0.05):
    band = pts[(pts[:, 2] > zlo) & (pts[:, 2] < zhi), :2]
    if len(band) < 50: return np.nan, 0.0
    xy = (np.unique(np.floor(band / cell).astype(np.int64), axis=0) + 0.5) * cell
    def score(th, b):
        c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
        u = xy[:, 0] * c + xy[:, 1] * s; v = -xy[:, 0] * s + xy[:, 1] * c
        hu = np.bincount(np.floor((u - u.min()) / b).astype(int)); hv = np.bincount(np.floor((v - v.min()) / b).astype(int))
        return float((hu.astype(float) ** 2).sum() + (hv.astype(float) ** 2).sum())
    ths = np.arange(0, 90, 0.5); sc = np.array([score(t, 0.05) for t in ths])
    best = ths[np.argmax(sc)]
    fine = best + np.arange(-0.75, 0.76, 0.05); scf = np.array([np.mean([score(t, b) for b in (0.03, 0.04, 0.05)]) for t in fine])
    conf = float(np.sort(sc)[-1] / np.median(sc))
    return float(fine[np.argmax(scf)] % 90), conf

def _circ_mean_mod90(a):
    return (math.degrees(np.angle(np.exp(1j * np.radians(4 * a)).mean())) / 4) % 90

def pca_angle(pts, zlo, zhi, cell=0.05, rad=0.25, k=12, lin=0.10):
    """the first estimator tried: local line direction of each 5 cm cell (PCA of
    its neighbours within 0.25 m), histogram mod 90, peak refined by circular mean"""
    band = pts[(pts[:, 2] > zlo) & (pts[:, 2] < zhi), :2]
    if len(band) < 50: return np.nan, 0.0
    xy = (np.unique(np.floor(band / cell).astype(np.int64), axis=0) + 0.5) * cell
    if len(xy) < 20: return np.nan, 0.0
    d, idx = cKDTree(xy).query(xy, k=k, distance_upper_bound=rad)
    valid = np.isfinite(d); nn = valid.sum(1)
    nb = xy[np.where(valid, idx, 0)]; w = valid[..., None].astype(float)
    mu = (nb * w).sum(1) / np.maximum(nn, 1)[:, None]
    c = (nb - mu[:, None, :]) * w
    ev, evec = np.linalg.eigh(np.einsum("nki,nkj->nij", c, c) / np.maximum(nn, 1)[:, None, None])
    line = (nn >= 5) & (ev[:, 0] / np.maximum(ev[:, 1], 1e-12) < lin)
    if line.sum() < 10: return np.nan, 0.0
    v = evec[line, :, 1]; a = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 90
    h = np.histogram(a, bins=180, range=(0, 90))[0]
    peak = (np.argmax(h + 0.5 * (np.roll(h, 1) + np.roll(h, -1))) + 0.5) * 0.5
    for half in (3.0, 1.5):
        sel = np.abs(M.wrap(a - peak, 90)) <= half
        if sel.sum() < 5: break
        peak = _circ_mean_mod90(a[sel])
    return float(peak), float((np.abs(M.wrap(a - peak, 90)) <= 2.0).mean())

variants = {
    "pca  full  band": lambda p: pca_angle(p, -0.30, 2.25),
    "pca  upper band": lambda p: pca_angle(p, 0.80, 2.25),
    "proj full  band": lambda p: proj_angle(p, -0.30, 2.25),
    "proj upper band": lambda p: proj_angle(p, 0.80, 2.25),
}
root = "dataset/IsaacSimOPV2V_hospital/validate"
cases = sorted(os.listdir(root))
cases = [c for c in cases if "warehouse" in c][:4:] + [c for c in cases if "office" in c][:4]
err = {k: [] for k in variants}
for case in cases:
    stems = sorted(f.split("/")[-1][:-4] for f in glob.glob(f"{root}/{case}/0/*.pcd"))[::3]
    for st in stems:
        if not os.path.exists(f"{root}/{case}/1/{st}.pcd"): continue
        P = [pcd_utils.pcd_to_np(f"{root}/{case}/{a}/{st}.pcd") for a in "01"]
        Y = [yaml.load(open(f"{root}/{case}/{a}/{st}.yaml"), Loader=M.LOADER)["lidar_pose"][4] for a in "01"]
        for k, f in variants.items():
            (a0, _), (a1, _) = f(P[0]), f(P[1])
            err[k].append(abs(M.wrap((a0 + Y[0]) - (a1 + Y[1]), 90)))
for k, e in err.items():
    e = np.array(e); ok = np.isfinite(e); e = e[ok]
    print(f"{k}: n={len(e):4d} median {np.median(e):5.2f} deg  <0.5 {100*(e<0.5).mean():5.1f}%  <1 {100*(e<1).mean():5.1f}%  <2 {100*(e<2).mean():5.1f}%  >5 {100*(e>5).mean():5.1f}%  no-estimate {100*(1-ok.mean()):.1f}%")
