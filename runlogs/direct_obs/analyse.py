"""What does seeing the other robot buy, and which part of the error matters?"""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
ES = os.path.join(HERE, "..", "estimated_sigma")
PAT = re.compile(r"mAP@0\.3=([0-9.]+)")
def m(path):
    v = None
    if os.path.exists(path):
        for line in open(path):
            if "Isaac multi-class" in line:
                g = PAT.search(line); v = float(g.group(1)) if g else v
    return v
LEVELS = ["0.10", "0.20", "0.30", "0.60", "0.80"]
ZERO = m(os.path.join(ES, "none_0.00.log"))
rows = []
for p in LEVELS:
    none = m(os.path.join(ES, f"none_{p}.log"))
    hedge = max(m(os.path.join(ES, f"{v}_{p}.log")) for v in ("none", "fixed1", "fixed2"))
    oracle = m(os.path.join(ES, f"oracle_{p}.log"))
    vis = m(os.path.join(HERE, f"pos_visible_{p}.log"))
    alw = m(os.path.join(HERE, f"pos_always_{p}.log"))
    yaw = m(os.path.join(HERE, f"yaw_always_{p}.log"))
    rows.append((p, none, hedge, oracle, vis, alw, yaw))
print(f"zero-noise mAP@0.3 = {ZERO:.4f}\n")
print(f"{'sigma':>6} {'no fix':>7} {'best hedge':>10} {'hedge oracle':>12} "
      f"{'pos if seen':>11} {'pos always':>10} {'yaw always':>10}")
for p, n, h, o, v, a, y in rows:
    f = lambda x: f"{x:.4f}" if x is not None else "   -  "
    print(f"{p:>6} {f(n):>7} {f(h):>10} {f(o):>12} {f(v):>11} {f(a):>10} {f(y):>10}")
print("\nshare of the pose-error loss recovered (relative to no fix, of the gap to zero noise)")
print(f"{'sigma':>6} {'pos if seen':>11} {'pos always':>10} {'yaw always':>10}   "
      f"{'seen / always':>13}")
for p, n, h, o, v, a, y in rows:
    loss = ZERO - n
    r = lambda x: f"{100*(x-n)/loss:9.1f}%" if x is not None else "      -   "
    print(f"{p:>6} {r(v):>11} {r(a):>10} {r(y):>10}   {100*(v-n)/(a-n):12.0f}%")
print("\nagainst the best existing hedge (blur), same noise level")
for p, n, h, o, v, a, y in rows:
    print(f"  sigma={p}: pos if seen {v-h:+.4f}   pos always {a-h:+.4f}")
