"""Paper figure: how overlap between the two robots changes pose-noise sensitivity.

Same model, same noise sweep, same scene set; only the test split differs
(complementary pairs vs high-overlap pairs). Numbers from runlogs/sweep.log and
runlogs/highoverlap/ho_*.log.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 9,
    "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.3, "lines.markersize": 4, "legend.frameon": False,
    "pdf.fonttype": 42,
})

STD  = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80]
LOW  = [.3987, .3377, .2734, .2215, .1977, .1656, .1518, .1442, .1418]
HIGH = [.3095, .2892, .2644, .2429, .2222, .1992, .1906, .1731, .1633]
LOW_R  = [v / LOW[0]  * 100 for v in LOW]
HIGH_R = [v / HIGH[0] * 100 for v in HIGH]
XCROSS = 0.115          # linear interpolation of LOW - HIGH between 0.10 and 0.15
SCALE  = 2.15           # measured mean true misalignment per unit of pos_std

BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#1a1a1a"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

# (a) absolute accuracy
ax1.plot(STD, LOW,  "o-", color=BLUE,   label="Low overlap (complementary)")
ax1.plot(STD, HIGH, "s-", color=ORANGE, label="High overlap")
ax1.axvline(XCROSS, color=INK, ls=":", lw=0.9)
ax1.plot([XCROSS], [0.2667], "o", color=INK, ms=4, zorder=5)
ax1.annotate(f"curves cross\n$\\sigma_{{\\mathrm{{pos}}}}\\approx{XCROSS:.2f}$ m",
             xy=(XCROSS, 0.2667), xytext=(0.26, 0.315), fontsize=7.4, color=INK,
             arrowprops=dict(arrowstyle="->", lw=0.7, color=INK,
                             connectionstyle="arc3,rad=-0.3"))
ax1.annotate("complementary views\nhelp when poses are good",
             xy=(0.015, 0.128), fontsize=7.2, color=BLUE, ha="left")
ax1.set_ylabel(r"mAP @ IoU $0.3$")
ax1.set_ylim(0.11, 0.43)
ax1.set_title("(a) Absolute accuracy", loc="left", pad=20)
ax1.legend(loc="upper right", handlelength=1.9, labelspacing=0.35, borderpad=0.2)

# (b) retention, each curve against its own clean score
ax2.fill_between(STD, LOW_R, HIGH_R, color=ORANGE, alpha=0.10, lw=0)
ax2.plot(STD, LOW_R,  "o-", color=BLUE,   label="Low overlap (complementary)")
ax2.plot(STD, HIGH_R, "s-", color=ORANGE, label="High overlap")
ax2.annotate("$\\approx$22 pt gap", xy=(0.47, 0.5 * (38.1 + 61.6)),
             ha="center", va="center", fontsize=7.6, color=ORANGE)
ax2.set_ylabel("Retained accuracy (\\% of clean)")
ax2.set_ylim(30, 103)
ax2.set_title("(b) Retention, normalised per curve", loc="left", pad=20)
ax2.legend(loc="upper right", handlelength=1.9, labelspacing=0.35, borderpad=0.2)

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
fig.savefig("figures/overlap_vs_noise.pdf", bbox_inches="tight")
fig.savefig("figures/overlap_vs_noise.png", dpi=300, bbox_inches="tight")
print("wrote figures/overlap_vs_noise.pdf and .png")
