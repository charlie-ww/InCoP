import numpy as np, pandas as pd
df = pd.read_csv("/nfs/home/charlie/InCoP/runlogs/freespace/contradiction.csv")
T, W = 0.55, 20
d = df.dropna(subset=["contradiction_obs"]).copy()
print(f"usable rows {len(d)} of {len(df)}  "
      f"({df.contradiction_obs.isna().sum()} unjudgeable -> deployed falls back)\n")
print(f"{'sigma':>6} {'n':>6} {'median':>8} {'frame<T':>9} {'roll20<T':>9} {'iid20<T':>8}")
rates = {}
for sig in sorted(d.sigma.unique()):
    s = d[d.sigma == sig].contradiction_obs.to_numpy()
    med = pd.Series(s).rolling(W).median().shift(1)
    rng = np.random.default_rng(0)
    boot = np.mean([np.median(rng.choice(s, W, replace=False)) < T
                    for _ in range(4000)])
    rates[sig] = (100*(med < T).mean(), 100*boot)
    print(f"{sig:6.2f} {len(s):6d} {np.median(s):8.3f} {100*(s<T).mean():8.1f}% "
          f"{100*(med<T).mean():8.1f}% {100*boot:7.1f}%")

dep = {0.00: 71.4, 0.10: 40.7, 0.20: 20.2, 0.30: 8.0, 0.60: 3.0, 0.80: 0.0}
print(f"\n{'sigma':>6} {'csv iid-20':>11} {'deployed':>9}  gap")
for sig, v in dep.items():
    if sig in rates:
        print(f"{sig:6.2f} {rates[sig][1]:10.1f}% {v:8.1f}%  {v - rates[sig][1]:+.1f} pts")
print(f"\ncsv frames {d.frame.nunique()}, cases {d.case.nunique()}; "
      f"NaN share by level:")
for sig in sorted(df.sigma.unique()):
    sub = df[df.sigma == sig]
    print(f"  sigma={sig:<5} unjudgeable {100*sub.contradiction_obs.isna().mean():5.1f}%")
