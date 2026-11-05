from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# FIGURE 12 — TIAGO GFI VS CENTRALITY
# Reported manuscript values preserved
# Regular fonts; panel labels below; no Type 3
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8
})

# ------------------------------------------------------------
# Manuscript Table V: TIAGo stratified-random placements
# Ten placements per object
# ------------------------------------------------------------

objects = ["Mug", "Bowl", "Spoon"]

gfi = np.array([
    0.4647,
    0.3350,
    0.6483
])

centrality = np.array([
    0.5826,
    0.9045,
    0.8182
])

# Reported paired correlation differences
delta = np.array([
    -0.1179,
    -0.5695,
    -0.1699
])

# Reported 95% paired bootstrap confidence intervals
ci_low = np.array([
    -0.8872,
    -1.3593,
    -0.7333
])

ci_high = np.array([
    0.9072,
    0.0000,
    0.0000
])

assert np.all(ci_low <= delta)
assert np.all(delta <= ci_high)

xerr = np.vstack([
    delta - ci_low,
    ci_high - delta
])

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"

# ------------------------------------------------------------
# Compact two-panel publication layout
# ------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(7.3, 3.05),
    gridspec_kw={
        "width_ratios": [1.0, 1.0]
    }
)

# ============================================================
# PANEL (a): SPEARMAN CORRELATIONS
# ============================================================

x = np.arange(len(objects))
width = 0.33

bars1 = ax1.bar(
    x - width / 2,
    gfi,
    width,
    color=BLUE,
    label="GFI"
)

bars2 = ax1.bar(
    x + width / 2,
    centrality,
    width,
    color=ORANGE,
    label="Centrality"
)

ax1.set_xticks(x)
ax1.set_xticklabels(objects)

ax1.set_ylabel(
    "Spearman correlation with grasp success"
)

ax1.set_ylim(0, 1.08)
ax1.set_yticks(np.arange(0, 1.01, 0.2))

ax1.grid(axis="y", alpha=0.18)
ax1.set_axisbelow(True)

# Values just above the individual bars
for bars in (bars1, bars2):
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.3f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                h
            ),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="normal"
        )

ax1.legend(
    loc="upper left",
    ncol=1,
    frameon=False,
    fontsize=8
)

# ============================================================
# PANEL (b): PAIRED DIFFERENCES
# ============================================================

y = np.arange(len(objects))[::-1]

ax2.errorbar(
    delta,
    y,
    xerr=xerr,
    fmt="o",
    color=BLUE,
    markersize=5,
    linewidth=1.5,
    elinewidth=1.5,
    capsize=3
)

ax2.axvline(
    0,
    color="#bb5555",
    linestyle="--",
    linewidth=1
)

ax2.set_yticks(y)
ax2.set_yticklabels(objects)

ax2.set_xlim(-1.5, 1.0)
ax2.set_xticks([
    -1.5, -1.0, -0.5, 0.0, 0.5, 1.0
])

ax2.set_xlabel(
    "Correlation difference: GFI - Centrality"
)

ax2.grid(axis="x", alpha=0.18)
ax2.set_axisbelow(True)

# ------------------------------------------------------------
# Shared heading, compact and regular
# ------------------------------------------------------------

fig.suptitle(
    "TIAGo stratified-random placements "
    "(10 placements per object)",
    fontsize=9,
    fontweight="normal",
    y=0.97
)

# ------------------------------------------------------------
# Layout: reserve space for labels BELOW panels
# ------------------------------------------------------------

fig.subplots_adjust(
    left=0.105,
    right=0.975,
    top=0.82,
    bottom=0.31,
    wspace=0.43
)

# ------------------------------------------------------------
# (a) and (b) BELOW the graphs
# ------------------------------------------------------------

for ax, caption in [
    (
        ax1,
        "(a) Physical TIAGo predictive association"
    ),
    (
        ax2,
        "(b) Paired differences (95% bootstrap CI)"
    )
]:
    box = ax.get_position()

    fig.text(
        (box.x0 + box.x1) / 2,
        0.065,
        caption,
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="normal"
    )

# ------------------------------------------------------------
# Export PDF and PNG without overwriting the original
# ------------------------------------------------------------

for ext in ("pdf", "png"):
    output = Path(
        "ICARA2027_TIAGo_GFI_vs_Centrality_VERIFIED_v2."
        + ext
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white"
    )

    print("Saved:", output.resolve())

plt.close(fig)

print("\nFigure 12 successfully regenerated.")
print("Reported GFI and centrality correlations preserved.")
print("Reported paired bootstrap intervals preserved.")
