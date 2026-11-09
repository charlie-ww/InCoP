"""Hedging versus correcting: two action classes with very different ceilings.

The sigma sweep measured a cap and I first read it as a cap on what pose
uncertainty can buy. It is not. It caps *hedging* - blurring, discarding or
reweighting what the partner sent under a pose you do not trust. Correcting the
pose is a different action class, and the same runs already bound it, because a
perfectly corrected pose is the sigma_pos = 0 operating point we measured.

The point of the comparison is not that correction wins. It is how tight the
correction has to be before it wins, because that requirement is what makes
this an indoor problem rather than a port of an outdoor one.
"""
import json, os, re
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
LEVELS = ["0.00", "0.10", "0.20", "0.30", "0.60", "0.80"]
HEDGES = ("none", "fixed1", "fixed2")
PAT = re.compile(r"mAP@0\.3=([0-9.]+)")

def read(v, p):
    val = None
    for line in open(os.path.join(ROOT, f"{v}_{p}.log")):
        if "Isaac multi-class" in line:
            m = PAT.search(line)
            if m:
                val = float(m.group(1))
    return val

T = {v: {p: read(v, p) for p in LEVELS}
     for v in HEDGES + ("estimated", "est_regime", "oracle")}
best_hedge = {p: max(T[v][p] for v in HEDGES) for p in LEVELS}
one_constant = max(np.mean([T[v][p] for p in LEVELS]) for v in HEDGES)

print("=== 1. what hedging can buy, against two baselines ===")
print(f"  best single constant across all levels      "
      f"{one_constant:.4f}")
print(f"  best constant chosen per level              "
      f"{np.mean(list(best_hedge.values())):.4f}"
      f"   {np.mean(list(best_hedge.values())) - one_constant:+.4f}")
print(f"  perfect PER-FRAME hedging (oracle)          "
      f"{np.mean([T['oracle'][p] for p in LEVELS]):.4f}"
      f"   {np.mean([T['oracle'][p] for p in LEVELS]) - one_constant:+.4f}")
print("  -> everything hedging can do, including a perfect per-frame choice,")
print(f"     is worth {np.mean([T['oracle'][p] for p in LEVELS]) - one_constant:+.4f} "
      f"over one constant.")

print("\n=== 2. what correcting the pose buys, by how far it gets ===")
print("  Reading down a column: if correction takes you from the row's noise")
print("  level to the column's, and you then hedge optimally, you score the")
print("  column's best-hedge value.")
print(f"\n  {'from sigma_pos':>14} {'best hedge':>11}   "
      + "".join(f"{'->'+p:>9}" for p in LEVELS[:-1]))
for i, p in enumerate(LEVELS):
    if i == 0:
        continue
    cells = []
    for j, q in enumerate(LEVELS[:-1]):
        cells.append(f"{best_hedge[q] - best_hedge[p]:+9.4f}" if j < i else f"{'':>9}")
    print(f"  {p:>14} {best_hedge[p]:11.4f}   " + "".join(cells))

perfect = best_hedge["0.00"]
fix_gain = [perfect - best_hedge[p] for p in LEVELS[1:]]
print(f"\n  perfect correction, averaged over the five levels with real "
      f"misalignment: {np.mean(fix_gain):+.4f}")

print("\n=== 3. the gradient: partial correction is nearly worthless here ===")
halvings = [("0.20", "0.10"), ("0.60", "0.30")]
for a, b in halvings:
    print(f"  halving {a} -> {b}: {best_hedge[b] - best_hedge[a]:+.4f}")
print(f"  mean of the two true halvings available in the grid: "
      f"{np.mean([best_hedge[b] - best_hedge[a] for a, b in halvings]):+.4f}")
print("  Halving the error from 0.60 is worth +0.0098; from 0.20 it is +0.0304.")
print("  The payoff is concentrated at the very end of the correction, because")
print("  the objects stop overlapping themselves long before the error is gone.")

print("\n=== 4. how tight does the correction have to be? ===")
sz = json.load(open(os.path.join(ROOT, "..", "size_ratio", "size_ratio.json")))
d_break = sz["median_class"]["break_iou_0.3"]
FACTOR = 2.15        # measured sigma_pos -> actual misalignment near the robot
CAR_BREAK = 1.23     # a 1.8 x 4.5 m car at IoU@0.3
print(f"  median indoor object breaks IoU@0.3 at   {d_break:.3f} m")
print(f"  -> required residual error               {d_break:.3f} m"
      f"  (sigma_pos < {d_break / FACTOR:.3f})")
print(f"  outdoor car breaks IoU@0.3 at            {CAR_BREAK:.2f} m")
print(f"  -> required residual error               {CAR_BREAK:.2f} m"
      f"  (sigma_pos < {CAR_BREAK / FACTOR:.2f})")
print(f"\n  The indoor requirement is {CAR_BREAK / d_break:.1f}x tighter.")
print(f"  Per class, the tightest is the fire extinguisher at "
      f"{min(r['break_iou']['0.3'] for r in sz['per_class']):.3f} m "
      f"-> {CAR_BREAK / min(r['break_iou']['0.3'] for r in sz['per_class']):.0f}x tighter.")
print("\n  And it has to be met with fewer correspondences, not more: 53.3% of")
print("  low-overlap frames have at most two co-visible objects, against 10.9%")
print("  in the high-overlap split (runlogs/overlap_geometry).")
