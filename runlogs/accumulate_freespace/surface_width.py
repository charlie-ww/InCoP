"""Two follow-ups on the single-frame cost surface.

1. Does free-space localisation work in the frames with no co-visible objects?
2. Does the WIDTH of the low-cost region know when the answer is wrong? If it
   does, the width is an alignment uncertainty in metres, with no threshold to
   tune and no ground truth involved.
"""
import csv, math, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "pilot_costs.npz"))
S = z["shifts"]
GX, GY = np.meshgrid(S, S, indexing="ij")
D = np.hypot(GX, GY)
same = {}
with open(os.path.join(HERE, "..", "cross_time_covisibility", "cross_time.csv")) as fh:
    for r in csv.DictReader(fh):
        same[(r["case"], int(r["t"]))] = int(r["same"])

err, width, shared = [], [], []
for k in z.files:
    if not k.startswith("xy_h_"):
        continue
    c = k[5:]
    H, N = z[k], z[f"xy_n_{c}"]
    for t in range(len(H)):
        if not (N[t] > 0).any():
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            cost = np.where(N[t] > 0, H[t] / N[t], np.nan)
        m = np.nanmin(cost)
        tied = cost <= m + 1e-12
        ax, ay = GX[tied].mean(), GY[tied].mean()
        err.append(float(D[tied].mean()))
        # how far from the best candidate do "nearly as good" candidates reach?
        near = cost <= m + 0.05
        width.append(float(np.sqrt(((GX[near]-ax)**2 + (GY[near]-ay)**2).mean())))
        shared.append(same.get((c, t), -1))
err, width, shared = map(np.array, (err, width, shared))

print("=== 1. free-space localisation by number of co-visible objects ===")
print(f"{'co-visible':>11} {'frames':>7} {'median err':>11} {'<=0.16 m':>9} {'<=0.30 m':>9}")
for lab, m in (("0", shared == 0), ("1-2", (shared >= 1) & (shared <= 2)),
               ("3-5", (shared >= 3) & (shared <= 5)), (">=6", shared >= 6)):
    e = err[m]
    print(f"{lab:>11} {m.sum():>7} {np.median(e):9.3f} m {100*(e<=0.16).mean():8.1f}% "
          f"{100*(e<=0.30).mean():8.1f}%")
print("  (chance: median 0.496 m, <=0.16 m 5.3%)")

def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

print("\n=== 2. does the width of the low-cost region predict the error? ===")
print(f"  Spearman(width, error) over {len(err)} frames: {spearman(width, err):+.3f}")
q = np.quantile(width, [1/3, 2/3])
for lab, m in (("narrowest third", width <= q[0]),
               ("middle third", (width > q[0]) & (width <= q[1])),
               ("widest third", width > q[1])):
    e = err[m]
    print(f"  {lab:16s} width median {np.median(width[m]):.2f} m -> "
          f"error median {np.median(e):.3f} m, <=0.16 m {100*(e<=0.16).mean():5.1f}%")
z0 = shared == 0
if z0.sum() > 30:
    print(f"  within zero-co-visible frames only: Spearman {spearman(width[z0], err[z0]):+.3f}")
