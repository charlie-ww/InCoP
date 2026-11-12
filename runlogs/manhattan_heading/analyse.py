"""Tables for README.md from heading.csv.

Tuning (confidence cut points, gate threshold, calibrated ranges) is done on the
warehouse and office scenes (train + validate); everything reported for the
hospital test split is out-of-building.
"""
import math, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "heading.csv"))
wrap = lambda a, p: (a + p / 2) % p - p / 2

d["beta0"] = (d.alpha0 + d.yaw0) % 90          # building axis in the world, seen by robot 0
d["beta1"] = (d.alpha1 + d.yaw1) % 90
d["e_rel"] = wrap(d.beta0 - d.beta1, 90)        # = wall relative heading - true relative heading
d["abs_e"] = d.e_rel.abs()
d["conf"] = np.minimum(d.peak_over_rival0, d.peak_over_rival1)
d["struct"] = np.minimum(d.peak_over_median0, d.peak_over_median1)
d["test"] = d.split.str.endswith("/test")
tune, test = d[~d.test], d[d.test]

def row(e):
    e = np.abs(np.asarray(e))
    return (f"{len(e):6d} | {np.median(e):5.2f} | {100*(e<0.41).mean():5.1f}% | {100*(e<1.15).mean():5.1f}% | "
            f"{100*(e<2).mean():5.1f}% | {100*(e>5).mean():5.1f}%")
HDR = "| n | median deg | <0.41 deg | <1.15 deg | <2 deg | >5 deg |"

print("## 1. Is the building axis-aligned? world angle of the walls, per scene")
for sc, g in d.groupby("scene"):
    b = np.concatenate([g.beta0, g.beta1])
    m = (math.degrees(np.angle(np.exp(1j*np.radians(4*b)).mean()))/4) % 90
    dev = np.abs(wrap(b - m, 90))
    print(f"  {sc:18s} axis {m:6.2f} deg (mod 90); single robot vs axis: median {np.median(dev):.2f}, "
          f"<1.15 {100*(dev<1.15).mean():.1f}%, >5 {100*(dev>5).mean():.1f}%")

print("\n## 2. Relative heading from the walls alone vs truth")
print(HDR)
for (sp, sc), g in d.groupby(["split", "scene"]):
    print(f"| {sp.split('/')[-1]:8s} {sc:18s} " + row(g.e_rel) + " |")

print("\n## 3. Hospital test, by objects both robots see in this frame")
print("| co-visible " + HDR[1:])
for lab, lo, hi in (("0", 0, 0), ("1-2", 1, 2), ("3-5", 3, 5), (">=6", 6, 999)):
    g = test[(test.covis >= lo) & (test.covis <= hi)]
    print(f"| {lab:4s} | " + row(g.e_rel) + " |")
print("| by distance between robots:")
for lo, hi in ((0, 4), (4, 8), (8, 12), (12, 99)):
    g = test[(test.dist >= lo) & (test.dist < hi)]
    print(f"| {lo:2d}-{hi:2d} m | " + row(g.e_rel) + " |")

print("\n## 4. Does the scan know when it is wrong?")
for k in ("conf", "struct"):
    for nm, g in (("tune", tune), ("test", test)):
        r = spearmanr(g[k], g.abs_e).correlation
        print(f"  Spearman({k}, |error|) on {nm}: {r:+.3f}")
cuts = np.quantile(tune.conf, [1/3, 2/3])
print(f"  confidence terciles cut on warehouse/office at {cuts[0]:.3f}, {cuts[1]:.3f}")
lab = lambda c: np.digitize(c, cuts)
print("| confidence | split " + HDR[1:])
for b, nm in enumerate(("low", "mid", "high")):
    for snm, g in (("tune", tune), ("test", test)):
        gg = g[lab(g.conf) == b]
        print(f"| {nm:4s} | {snm} | " + row(gg.e_rel) + " |")

print("\n## 5. Calibrated 90% heading range per confidence tercile (calibrated on warehouse/office)")
q = {}
for b in range(3):
    e = np.sort(tune.abs_e[lab(tune.conf) == b].values); n = len(e)
    q[b] = e[min(n - 1, int(math.ceil(0.9 * (n + 1))) - 1)]
w = np.array([q[b] for b in lab(test.conf)])
cov = (test.abs_e.values <= w)
print(f"  half-widths: low {q[0]:.2f}, mid {q[1]:.2f}, high {q[2]:.2f} deg")
print(f"  hospital coverage {100*cov.mean():.1f}%, mean half-width {w.mean():.2f} deg")
for b, nm in enumerate(("low", "mid", "high")):
    m = lab(test.conf) == b
    print(f"    {nm:4s}: coverage {100*cov[m].mean():5.1f}%  ({m.sum()} frames)")
for lab_, lo, hi in (("0 co-visible", 0, 0), (">=1 co-visible", 1, 999)):
    m = ((test.covis >= lo) & (test.covis <= hi)).values
    print(f"    {lab_}: coverage {100*cov[m].mean():5.1f}%, mean half-width {w[m].mean():.2f}")
print("  for comparison, the 90% range of the injected relative heading noise (1.645 x sqrt2 x 10 sigma):")
print("   ", ", ".join(f"sigma {s}: {1.645*math.sqrt(2)*10*s:.2f}" for s in (0.1, 0.2, 0.3, 0.6, 0.8)))

print("\n## 6. Replacing the reported relative heading (simulated noise, 20 draws per frame)")
rng = np.random.default_rng(0)
def after(g, s, gate=None):
    tr = (g.yaw1 - g.yaw0).values[None, :]
    rep = tr + rng.normal(0, 10*s, (20, len(g))) - rng.normal(0, 10*s, (20, len(g)))
    wall = (g.alpha0 - g.alpha1).values[None, :]
    res = rep + wrap(wall - rep, 90)              # 90-deg ambiguity resolved by the reported heading
    use = np.ones_like(res, bool) if gate is None else np.broadcast_to(gate[None, :], res.shape)
    est = np.where(use, res, rep)
    return np.abs(wrap(rep - tr, 360)), np.abs(wrap(est - tr, 360))
gate_tau = None
best = -1
for tau in np.quantile(tune.conf, np.linspace(0, 0.9, 19)):
    sc_ = np.mean([(after(tune, s, (tune.conf >= tau).values)[1] < 1.15).mean() for s in (0.1, 0.2, 0.3, 0.6, 0.8)])
    if sc_ > best: best, gate_tau = sc_, tau
print(f"  confidence gate chosen on warehouse/office: use the walls when conf >= {gate_tau:.3f} "
      f"({100*(test.conf >= gate_tau).mean():.1f}% of hospital frames)")
print("| sigma | relative-heading noise std | within 1.15 deg: reported | walls always | walls if confident | median deg: reported | walls always | ambiguity wrong |")
for s in (0.1, 0.2, 0.3, 0.6, 0.8):
    rb, ra = after(test, s)
    _, rg = after(test, s, (test.conf >= gate_tau).values)
    print(f"| {s:.1f} | {math.sqrt(2)*10*s:5.2f} | {100*(rb<1.15).mean():5.1f}% | {100*(ra<1.15).mean():5.1f}% | "
          f"{100*(rg<1.15).mean():5.1f}% | {np.median(rb):5.2f} | {np.median(ra):5.2f} | {100*(ra>45).mean():.3f}% |")

print("\n## 7. Pooling each robot's wall angle over its own past frames")
print("  Assumes the robot knows how much it has itself turned since k frames ago (gyro /")
print("  odometry is accurate over ~1.6 m); taken here from the true yaw increments.")
print("  Window = the current frame and the k-1 before it, in order, within the episode.")
def circ_med90(x):
    # circular median mod 90: the sample minimising summed wrapped distance
    x = np.asarray(x); D = np.abs(wrap(x[:, None] - x[None, :], 90)).sum(1)
    return x[np.argmin(D)]
def pooled(df, k):
    out_b0, out_b1, out_s = [], [], []
    for _, g in df.sort_values(["case", "t"]).groupby("case", sort=False):
        b0, b1 = g.beta0.values, g.beta1.values
        for i in range(len(g)):
            lo = max(0, i - k + 1)
            w0, w1 = b0[lo:i+1], b1[lo:i+1]
            m0, m1 = circ_med90(w0), circ_med90(w1)
            out_b0.append(m0); out_b1.append(m1)
            out_s.append(max(np.median(np.abs(wrap(w0 - m0, 90))), np.median(np.abs(wrap(w1 - m1, 90)))) if i > lo else np.nan)
    df = df.sort_values(["case", "t"]).copy()
    df["e_pool"] = wrap(np.array(out_b0) - np.array(out_b1), 90)
    df["spread"] = out_s
    return df
print(HDR)
pooled_by_k = {}
for k in (1, 5, 10, 20):
    tt = pooled(test, k); pooled_by_k[k] = tt
    print(f"| k={k:2d} hospital " + row(tt.e_pool) + " |")
for k in (10, 20):
    tt = pooled_by_k[k]
    for lab_, lo, hi in (("0 co-visible", 0, 0), (">=1", 1, 999)):
        g = tt[(tt.covis >= lo) & (tt.covis <= hi)]
        print(f"| k={k:2d} {lab_} " + row(g.e_pool) + " |")
    ok = tt.spread.notna()
    print(f"  k={k}: Spearman(window spread, |error|) on hospital = {spearmanr(tt.spread[ok], tt.e_pool.abs()[ok]).correlation:+.3f}")
    tu = pooled(tune, k); oku = tu.spread.notna()
    print(f"  k={k}: Spearman(window spread, |error|) on warehouse/office = {spearmanr(tu.spread[oku], tu.e_pool.abs()[oku]).correlation:+.3f}")
    cuts_s = np.quantile(tu.spread[oku], [1/3, 2/3])
    qs = {}
    labs_u = np.digitize(tu.spread[oku], cuts_s)
    for b in range(3):
        e = np.sort(tu.e_pool.abs()[oku].values[labs_u == b]); n = len(e)
        qs[b] = e[min(n - 1, int(math.ceil(0.9 * (n + 1))) - 1)]
    labs_t = np.digitize(tt.spread[ok], cuts_s)
    w_ = np.array([qs[b] for b in labs_t]); c_ = tt.e_pool.abs()[ok].values <= w_
    print(f"  calibrated 90% half-widths by spread tercile (warehouse/office): {qs[0]:.2f} / {qs[1]:.2f} / {qs[2]:.2f} deg")
    print(f"  hospital coverage {100*c_.mean():.1f}%, mean half-width {w_.mean():.2f} deg; by tercile "
          + ", ".join(f"{100*c_[labs_t==b].mean():.1f}%" for b in range(3)))

print("\n## 8. Reject wall readings that disagree with the reported heading by > 3 noise std")
print("  (needs the noise level, which the regime detector estimates; true level used here)")
print("| sigma | within 1.15 deg: reported | walls k=1 gated | walls k=10 | walls k=10 gated | median: k=10 gated |")
for s in (0.1, 0.2, 0.3, 0.6, 0.8):
    sr = math.sqrt(2) * 10 * s
    res = {}
    for k, ecol in ((1, "e_rel"), (10, "e_pool")):
        g = test if k == 1 else pooled_by_k[10]
        e = g[ecol].values[None, :]
        noise = rng.normal(0, 10*s, (20, len(g))) - rng.normal(0, 10*s, (20, len(g)))
        # wall estimate error relative to truth is e (mod 90, ambiguity resolved by reported)
        e_res = wrap(e - noise, 90) + noise
        keep = np.abs(e_res - noise) <= 3 * sr
        res[(k, False)] = np.abs(e_res)
        res[(k, True)] = np.where(keep, np.abs(e_res), np.abs(noise))
        res["rep"] = np.abs(noise)
    print(f"| {s:.1f} | {100*(res['rep']<1.15).mean():5.1f}% | {100*(res[(1,True)]<1.15).mean():5.1f}% | "
          f"{100*(res[(10,False)]<1.15).mean():5.1f}% | {100*(res[(10,True)]<1.15).mean():5.1f}% | {np.median(res[(10,True)]):.2f} |")
