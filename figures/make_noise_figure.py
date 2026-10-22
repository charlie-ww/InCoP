"""Paper figure: detection accuracy under injected pose noise.

Reads nothing at run time; the numbers are the nine-point sweep in
runlogs/sweep.log and the author's no-fusion baseline, both reproducible with
runlogs/sweep.sh. Writes PDF (for LaTeX) and 300-dpi PNG.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.3,
    "lines.markersize": 4,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})

STD   = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80]
AP30  = [.3987, .3377, .2734, .2215, .1977, .1656, .1518, .1442, .1418]
AP50  = [.2525, .1918, .1425, .1181, .1049, .0927, .0872, .0850, .0852]
DIST  = [.4317, .3624, .2883, .2433, .2145, .1847, .1671, .1599, .1581]
OCCL  = [.3607, .3105, .2588, .2018, .1833, .1527, .1428, .1346, .1327]
NOFUSE_30, NOFUSE_50 = 0.2196, 0.1433
SCALE = 2.15          # measured: mean true misalignment ≈ 2.15 x pos_std

BLUE, ORANGE, AQUA, INK = "#2a78d6", "#eb6834", "#1baf7a", "#1a1a1a"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.85))

# (a) overall accuracy at two IoU thresholds
ax1.axhline(NOFUSE_30, color=INK, ls=(0, (5, 3)), lw=0.9, label=r"No fusion, IoU $0.3$")
ax1.axhline(NOFUSE_50, color=INK, ls=(0, (1.5, 2)), lw=0.9, label=r"No fusion, IoU $0.5$")
ax1.plot(STD, AP30, "o-", color=BLUE,   label=r"Fused, IoU $0.3$")
ax1.plot(STD, AP50, "s-", color=ORANGE, label=r"Fused, IoU $0.5$")
ax1.annotate("fusion falls below\nno fusion", xy=(0.60, 0.148), xytext=(0.44, 0.205),
             fontsize=7.2, color=INK, ha="left",
             arrowprops=dict(arrowstyle="->", lw=0.7, color=INK,
                             connectionstyle="arc3,rad=-0.25"))
ax1.set_ylabel("mAP")
ax1.set_ylim(0.05, 0.42)
ax1.set_title("(a) Overall accuracy", loc="left", pad=20)
h, l = ax1.get_legend_handles_labels()
ax1.legend([h[2], h[3], h[0], h[1]], [l[2], l[3], l[0], l[1]],
           loc="upper right", handlelength=1.9, labelspacing=0.35, borderpad=0.2)

# (b) accuracy by complementarity type, with the error each type tolerates
X_DIST, X_OCCL = 0.191, 0.134          # where each curve meets the no-fusion line
ax2.axvspan(X_OCCL, X_DIST, color=BLUE, alpha=0.07, lw=0)
ax2.axhline(NOFUSE_30, color=INK, ls=(0, (5, 3)), lw=0.9, label="No fusion")
ax2.plot(STD, DIST, "^-", color=AQUA, label="Distance cases")
ax2.plot(STD, OCCL, "v-", color=BLUE, label="Occlusion cases")
for x, c in ((X_OCCL, BLUE), (X_DIST, AQUA)):
    ax2.plot([x, x], [0.10, NOFUSE_30], color=c, ls=":", lw=1.0)
    ax2.plot([x], [NOFUSE_30], "o", color=c, ms=4.5, mec="white", mew=0.8, zorder=5)
ax2.annotate(f"{X_OCCL:.2f}", xy=(X_OCCL - 0.013, 0.112), ha="right", fontsize=7.4, color=BLUE)
ax2.annotate(f"{X_DIST:.2f}", xy=(X_DIST + 0.013, 0.112), ha="left", fontsize=7.4, color=AQUA)
ax2.annotate("tolerable noise", xy=(0.163, 0.127), ha="center", va="bottom",
             fontsize=7.2, color=INK)
ax2.annotate("occlusion complementarity\nstops helping sooner", xy=(0.345, 0.255),
             fontsize=7.2, color=INK, ha="left")
ax2.set_ylabel(r"mAP @ IoU $0.3$")
ax2.set_ylim(0.10, 0.46)
ax2.set_title("(b) By type of complementarity", loc="left", pad=20)
h, l = ax2.get_legend_handles_labels()
ax2.legend([h[1], h[2], h[0]], [l[1], l[2], l[0]],
           loc="upper right", handlelength=1.9, labelspacing=0.35, borderpad=0.2)

for ax in (ax1, ax2):
    ax.set_xlabel(r"Injected pose noise $\sigma_{\mathrm{pos}}$ (m)")
    ax.set_xlim(-0.03, 0.83)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    sec = ax.secondary_xaxis("top", functions=(lambda v: v * SCALE, lambda v: v / SCALE))
    sec.set_xlabel("Mean true misalignment (m)", labelpad=4)
    sec.tick_params(direction="in", width=0.7, labelsize=8)
    sec.set_xticks([0.0, 0.5, 1.0, 1.5])

fig.tight_layout(pad=0.4, w_pad=1.6)
fig.savefig("figures/pose_noise.pdf", bbox_inches="tight")
fig.savefig("figures/pose_noise.png", dpi=300, bbox_inches="tight")
print("wrote figures/pose_noise.pdf and .png")
