"""Paper figure: free space carries alignment information without co-visible objects.

Two panels: (a) the contradiction score against the true alignment error, with
the thresholds of the three-way blur policy, and (b) how reliably the score
orders the noise draws within a single frame. Numbers from
runlogs/freespace/contradiction.csv, produced by the measurement script beside it.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 9,
    "axes.titlesize": 9, "legend.fontsize": 7.6, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.3, "lines.markersize": 3.6, "legend.frameon": False,
    "pdf.fonttype": 42,
})

BLUE, ORANGE, AQUA, INK = "#2a78d6", "#eb6834", "#1baf7a", "#1a1a1a"
GREY = "#8a8a8a"

# Thresholds of the deployable policy, set by matching the oracle's own split
# of the samples so that the two assign the same proportions.
T_LOW, T_HIGH = 0.429, 0.600
MIN_JUDGEABLE = 20

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, os.pardir, "runlogs", "freespace",
                              "contradiction.csv"))
noisy = df[df.sigma > 0]
usable = noisy[noisy.partner_cells_observed >= MIN_JUDGEABLE]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.4, 2.5))

# ---- (a) the score tracks the true error -----------------------------------
edges = [0, .1, .2, .3, .5, .75, 1., 1.5, 2., 99]
binned = usable.assign(bin=pd.cut(usable.error_m, edges)).groupby("bin", observed=True)
err = binned.error_m.mean().values
med = binned.contradiction_obs.median().values
q25 = binned.contradiction_obs.quantile(.25).values
q75 = binned.contradiction_obs.quantile(.75).values

ax1.fill_between(err, q25, q75, color=BLUE, alpha=0.16, linewidth=0)
ax1.plot(err, med, "o-", color=BLUE, zorder=3)

for y, label in ((T_LOW, r"$\sigma_{\mathrm{blur}}{=}0$"),
                 (T_HIGH, r"$\sigma_{\mathrm{blur}}{=}1$")):
    ax1.axhline(y, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax1.text(2.9, T_LOW - 0.055, r"$\sigma_{\mathrm{blur}}{=}0$ below", color=GREY,
         fontsize=7, ha="right")
ax1.text(2.9, T_HIGH + 0.022, r"$\sigma_{\mathrm{blur}}{=}2$ above", color=GREY,
         fontsize=7, ha="right")

ax1.set_xscale("log")
ax1.set_xlim(0.05, 3.2)
ax1.set_ylim(0.2, 1.02)
ax1.set_xlabel("true alignment error (m)")
ax1.set_ylabel("contradiction score")
ax1.set_title("(a) score vs. true misalignment")
ax1.set_xticks([0.1, 0.3, 1.0, 3.0])
ax1.set_xticklabels(["0.1", "0.3", "1.0", "3.0"])
ax1.grid(axis="y", color=GREY, lw=0.4, alpha=0.25)
ax1.set_axisbelow(True)

rho_all = stats.spearmanr(usable.contradiction_obs, usable.error_m).statistic
ax1.text(0.058, 0.955, rf"Spearman $\rho={rho_all:.2f}$", fontsize=7.6, color=INK)

# ---- (b) within one frame, does it order the draws? ------------------------
rhos = []
for _, group in usable.groupby(["case", "frame"]):
    if group.contradiction_obs.notna().sum() >= 8:
        r = stats.spearmanr(group.contradiction_obs, group.error_m).statistic
        if np.isfinite(r):
            rhos.append(r)
rhos = np.array(rhos)

ax2.hist(rhos, bins=np.arange(-1, 1.0001, 0.1), color=AQUA, alpha=0.75,
         edgecolor="white", linewidth=0.5)
ax2.axvline(0, color=GREY, lw=0.6)
ax2.axvline(np.median(rhos), color=ORANGE, lw=1.3)
ax2.text(np.median(rhos) - 0.05, ax2.get_ylim()[1] * 0.93,
         f"median {np.median(rhos):.2f}", color=ORANGE, fontsize=7.6, ha="right")
ax2.set_xlim(-1, 1)
ax2.set_xlabel(r"per-frame Spearman $\rho$")
ax2.set_ylabel("frames")
ax2.set_title("(b) ranking within one frame")
ax2.grid(axis="y", color=GREY, lw=0.4, alpha=0.25)
ax2.set_axisbelow(True)
ax2.text(-0.95, ax2.get_ylim()[1] * 0.93,
         f"{100 * (rhos > 0).mean():.0f}% positive\n{len(rhos)} frames",
         fontsize=7.6, color=INK, va="top")

# The limit is evidence, not the statistic: state it on the figure itself.
no_evidence = 100.0 * noisy.contradiction_obs.isna().mean()
fig.text(0.5, -0.02,
         f"Partner lands nowhere the ego can judge in {no_evidence:.0f}% of draws; "
         "those are excluded here and need a fallback policy.",
         ha="center", fontsize=7, color=GREY)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"freespace_uncertainty.{ext}"),
                dpi=400, bbox_inches="tight")
print("wrote freespace_uncertainty.{pdf,png}")
print(f"  pooled rho {rho_all:.3f}   frames {len(rhos)}   "
      f"median per-frame rho {np.median(rhos):.3f}   "
      f"positive {100 * (rhos > 0).mean():.1f}%")
