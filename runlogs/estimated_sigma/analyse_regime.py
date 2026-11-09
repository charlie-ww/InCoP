"""Did the regime selector escape the cap, and what is the cap actually made of?"""
import glob, os, re
import numpy as np

ROOT = "/nfs/home/charlie/InCoP/runlogs/estimated_sigma"
LEVELS = ["0.00", "0.10", "0.20", "0.30", "0.60", "0.80"]
VARIANTS = ["none", "fixed1", "fixed2", "estimated", "est_regime", "oracle"]
PAT = re.compile(r"mAP@0\.3=([0-9.]+), mAP@0\.5=([0-9.]+), mAP@0\.7=([0-9.]+)")

def read(v, p):
    f = os.path.join(ROOT, f"{v}_{p}.log")
    if not os.path.exists(f):
        return None
    m = None
    for line in open(f):
        if "Isaac multi-class" in line:
            got = PAT.search(line)
            if got:
                m = tuple(float(x) for x in got.groups())
    return m

tab = {v: [read(v, p) for p in LEVELS] for v in VARIANTS}

print("mAP@0.3 by injected pose noise (sigma_pos, metres)")
print(f"{'variant':12s}" + "".join(f"{p:>9s}" for p in LEVELS) + f"{'mean':>9s}")
for v in VARIANTS:
    row = [r[0] for r in tab[v]]
    print(f"{v:12s}" + "".join(f"{x:9.4f}" for x in row) + f"{np.mean(row):9.4f}")

# The per-level best constant: one global decision per operating regime, no
# per-frame estimation at all.
const = [max(tab[v][i][0] for v in ("none", "fixed1", "fixed2"))
         for i in range(len(LEVELS))]
pick = [max(("none", "fixed1", "fixed2"), key=lambda v: tab[v][i][0])
        for i in range(len(LEVELS))]
print(f"\n{'per-level best constant':30s}" + "".join(f"{x:9.4f}" for x in const)
      + f"{np.mean(const):9.4f}")
print(f"{'  (which constant)':30s}" + "".join(f"{p:>9s}" for p in pick))

f2 = np.mean([r[0] for r in tab["fixed2"]])
orc = np.mean([r[0] for r in tab["oracle"]])
est = np.mean([r[0] for r in tab["estimated"]])
reg = np.mean([r[0] for r in tab["est_regime"]])
cst = np.mean(const)

print("\n=== how much is each decision worth, over the best single constant ===")
print(f"  best single constant (fixed sigma=2)      {f2:.4f}   +0.0000")
for name, val in (("per-frame estimated sigma", est),
                  ("regime-adaptive calibration", reg),
                  ("perfect per-LEVEL constant", cst),
                  ("perfect per-FRAME oracle", orc)):
    print(f"  {name:41s} {val:.4f}   {val - f2:+.4f}"
          f"   ({100*(val-f2)/(orc-f2):5.1f}% of oracle gain)")

print(f"\n  per-frame knowledge beyond per-level knowledge: "
      f"{orc - cst:+.4f} ({100*(orc-cst)/(orc-f2):.1f}% of the oracle gain)")

print("\n=== at which levels does ANY estimation beat the constant? ===")
for i, p in enumerate(LEVELS):
    f2i, oi, ri = tab["fixed2"][i][0], tab["oracle"][i][0], tab["est_regime"][i][0]
    print(f"  sigma={p}  fixed2={f2i:.4f}  oracle={oi:.4f} ({oi-f2i:+.4f})"
          f"  regime={ri:.4f} ({ri-f2i:+.4f})"
          f"   {'oracle beats constant' if oi > f2i + 1e-4 else 'constant is already optimal'}")

print("\n=== where regime gains and where it pays ===")
for i, p in enumerate(LEVELS):
    d = tab["est_regime"][i][0] - tab["estimated"][i][0]
    print(f"  sigma={p}  regime - single calibration = {d:+.4f}")
