"""Regrade the pilot without the tie-break that favoured the truth.

pilot.py averaged tied minima. The truth sits at the centre of the grid, so a
flat cost surface averaged to the centre and scored as a perfect answer. Here a
tie is graded by the MEAN DISTANCE of the tied candidates from the truth, which
is what picking one of them at random would give.
"""
import math, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "pilot_costs.npz"))
S, Y = z["shifts"], z["yaws"]
cases = sorted(k[5:] for k in z.files if k.startswith("xy_h_"))
D = np.hypot(*np.meshgrid(S, S, indexing="ij"))
chance_xy, chance_yaw = D.mean(), np.abs(Y).mean()

def grade(h, n, dist):
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(n > 0, h / n, np.nan)
    if np.all(np.isnan(c)):
        return np.nan, 0
    tied = c <= np.nanmin(c) + 1e-12
    return float(dist[tied].mean()), int(tied.sum())

print(f"chance: translation {chance_xy:.3f} m (<=0.16 m by chance "
      f"{100*(D<=0.16).mean():.1f}%), yaw {chance_yaw:.2f} deg "
      f"(<=1.15 by chance {100*(np.abs(Y)<=1.15).mean():.1f}%)\n")
print(f"{'frames':>14} {'transl. median':>15} {'<=0.16 m':>9} {'<=0.30 m':>9} "
      f"{'tied (median)':>14} {'yaw median':>11} {'<=1.15':>7} {'n':>4}")
for N in (1, 3, 5, 10, 20, 30, 999):
    ex, ties, ey = [], [], []
    for c in cases:
        hx, nx, hy, ny = z[f"xy_h_{c}"], z[f"xy_n_{c}"], z[f"yw_h_{c}"], z[f"yw_n_{c}"]
        if N != 999 and len(hx) < N:
            continue
        k = len(hx) if N == 999 else N
        e, t = grade(hx[:k].sum(0), nx[:k].sum(0), D)
        y, _ = grade(hy[:k].sum(0), ny[:k].sum(0), np.abs(Y))
        if not np.isnan(e):
            ex.append(e); ties.append(t)
        if not np.isnan(y):
            ey.append(y)
    ex, ey = np.array(ex), np.array(ey)
    lab = "whole episode" if N == 999 else str(N)
    print(f"{lab:>14} {np.median(ex):13.3f} m {100*(ex<=0.16).mean():8.1f}% "
          f"{100*(ex<=0.30).mean():8.1f}% {np.median(ties):14.0f} "
          f"{np.median(ey):9.2f}deg {100*(ey<=1.15).mean():6.1f}% {len(ex):>4}")

# sliding windows anywhere in the episode, not just from its start
print("\nany window of N consecutive frames (all positions, all episodes):")
for N in (1, 10, 30):
    ex = []
    for c in cases:
        hx, nx = z[f"xy_h_{c}"], z[f"xy_n_{c}"]
        for s in range(0, max(1, len(hx) - N + 1)):
            e, _ = grade(hx[s:s+N].sum(0), nx[s:s+N].sum(0), D)
            if not np.isnan(e):
                ex.append(e)
    ex = np.array(ex)
    print(f"  N={N:>2}: median {np.median(ex):.3f} m, <=0.16 m {100*(ex<=0.16).mean():.1f}%, "
          f"<=0.30 m {100*(ex<=0.30).mean():.1f}%  ({len(ex)} windows)")
