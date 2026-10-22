"""Paper figure: overlap sensitivity is robust to the pose-noise model.

Three panels: (a) accuracy under the project's 1:10 translation:rotation noise,
(b) the same under the 1:1 ratio used in the collaborative-perception literature,
(c) why — accuracy split by which robot can see each object.
Numbers from runlogs/{sweep.log, highoverlap/, ratio11/} and the per-run
visibility_subset_metrics in the evaluation YAMLs.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 9,
    "axes.titlesize": 9, "legend.fontsize": 7.6, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.3, "lines.markersize": 3.6, "legend.frameon": False,
    "pdf.fonttype": 42,
})

STD = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80]
LOW10  = [.3987, .3377, .2734, .2215, .1977, .1656, .1518, .1442, .1418]
HIGH10 = [.3095, .2892, .2644, .2429, .2222, .1992, .1906, .1731, .1633]
LOW11  = [.3983, .3692, .3149, .2701, .2356, .1955, .1732, .1541, .1460]
HIGH11 = [.3089, .2936, .2784, .2553, .2395, .2193, .1974, .1842, .1739]
# subset retention, low-overlap split, 1:1 noise
EGO  = [100.0, 96.3, 88.7, 82.3, 77.7, 71.8, 69.1, 66.4, 65.2]
SHAR = [100.0, 88.1, 68.5, 50.1, 38.5, 24.6, 18.1, 12.9, 11.1]
PART = [100.0, 83.7, 58.2, 38.0, 24.4,  8.8,  3.8,  1.2,  0.3]

BLUE, ORANGE, AQUA, INK = "#2a78d6", "#eb6834", "#1baf7a", "#1a1a1a"

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75))
ax1, ax2, ax3 = axes

def gap(lo, hi, p):
    i = STD.index(p)
    return hi[i] / hi[0] * 100 - lo[i] / lo[0] * 100

for ax, lo, hi, title, xc in (
    (ax1, LOW10, HIGH10, r"(a) $\sigma_{\mathrm{rot}}=10\,\sigma_{\mathrm{pos}}$", 0.115),
    (ax2, LOW11, HIGH11, r"(b) $\sigma_{\mathrm{rot}}=\sigma_{\mathrm{pos}}$", 0.19),
):
    ax.plot(STD, lo, "o-", color=BLUE,   label="Low overlap")
    ax.plot(STD, hi, "s-", color=ORANGE, label="High overlap")
    ax.axvline(xc, color=INK, ls=":", lw=0.9)
    ax.text(xc + 0.03, 0.385, f"cross\n{xc:.2f} m", fontsize=7, color=INK, va="top")
    ax.text(0.80, 0.128, f"gap at 0.4 m:\n{gap(lo, hi, 0.40):+.0f} pt",
            fontsize=7.2, color=ORANGE, ha="right")
    ax.set_ylabel(r"mAP @ IoU $0.3$")
    ax.set_ylim(0.11, 0.43)
    ax.set_title(title, loc="left", pad=8)
    ax.legend(loc="upper right", handlelength=1.7, labelspacing=0.3, borderpad=0.2)

ax3.plot(STD, EGO,  "^-", color=AQUA,   label="Ego-visible only")
ax3.plot(STD, SHAR, "s-", color=ORANGE, label="Seen by both")
ax3.plot(STD, PART, "v-", color=BLUE,   label="Partner-visible only")
ax3.annotate("the complementary\ninformation collapses",
             xy=(0.40, 3.8), xytext=(0.30, 30), fontsize=7.2, color=BLUE,
             arrowprops=dict(arrowstyle="->", lw=0.7, color=BLUE,
                             connectionstyle="arc3,rad=-0.25"))
ax3.set_ylabel("Retained accuracy (\\% of clean)")
ax3.set_ylim(-4, 106)
ax3.set_title(r"(c) By visibility, $\sigma_{\mathrm{rot}}=\sigma_{\mathrm{pos}}$", loc="left", pad=8)
ax3.legend(loc="upper right", handlelength=1.7, labelspacing=0.3, borderpad=0.2)

for ax in axes:
    ax.set_xlabel(r"$\sigma_{\mathrm{pos}}$ (m)")
    ax.set_xlim(-0.03, 0.83)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)

fig.tight_layout(pad=0.4, w_pad=1.3)
fig.savefig("figures/overlap_noise_model.pdf", bbox_inches="tight")
fig.savefig("figures/overlap_noise_model.png", dpi=300, bbox_inches="tight")
print("wrote figures/overlap_noise_model.pdf and .png")
