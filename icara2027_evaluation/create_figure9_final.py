from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============================================================
# FIGURE 9 — ICARA 2027, FINAL READABLE VERSION
# Same data as v2, improved layout, no Type 3 fonts
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

# Original values: unchanged
objects = [
    "Banana", "Power\nDrill", "Mug", "Spoon", "Bowl"
]

gfi = np.array([0.560, 1.000, 0.489, 0.500, 0.517])
panda = np.array([0.792, 0.912, 0.970, 0.455, 0.500])

names = [
    "Banana (21/29)",
    "Power Drill (34/35)",
    "Mug (22/34)",
    "Spoon (22/24)",
    "Bowl (2/32)"
]

delta = np.array([-0.232, 0.088, -0.481, 0.045, 0.017])
ci_low = np.array([-0.595, 0.000, -0.739, -0.409, -0.233])
ci_high = np.array([0.131, 0.206, -0.227, 0.500, 0.267])

assert np.all(ci_low <= delta)
assert np.all(delta <= ci_high)

xerr = np.vstack([delta - ci_low, ci_high - delta])

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"

# Layout designed for a two-column conference paper
fig, axes = plt.subplots(
    1, 2,
    figsize=(10.0, 4.0),
    gridspec_kw={"width_ratios": [1.0, 1.1]}
)

ax1, ax2 = axes

# ------------------------------------------------------------
# Panel (a): AUROC comparison
# ------------------------------------------------------------
x = np.arange(len(objects))
width = 0.34

b1 = ax1.bar(
    x - width / 2,
    gfi,
    width,
    color=BLUE
)

b2 = ax1.bar(
    x + width / 2,
    panda,
    width,
    color=ORANGE
)

ax1.axhline(
    0.5, color="gray",
    linestyle=":", linewidth=1.3
)

ax1.set_ylim(0, 1.16)
ax1.set_ylabel("AUROC", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(objects, fontweight="bold")
ax1.set_title(
    "(a) Grasp-success AUROC",
    fontweight="bold",
    pad=9
)
ax1.grid(axis="y", alpha=0.15)
ax1.set_axisbelow(True)

# Value labels, with enough space above bars
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.3f}",
            (
                bar.get_x() + bar.get_width() / 2,
                h
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

# ------------------------------------------------------------
# Panel (b): Paired differences
# ------------------------------------------------------------
y = np.arange(len(names))[::-1]

ax2.errorbar(
    delta, y,
    xerr=xerr,
    fmt="o",
    color=BLUE,
    markersize=6,
    linewidth=1.7,
    elinewidth=1.7,
    capsize=4
)

ax2.axvline(
    0,
    linestyle="--",
    linewidth=1.2,
    color="gray"
)

ax2.set_yticks(y)
ax2.set_yticklabels(names)
ax2.set_xlim(-0.85, 0.60)
ax2.set_xlabel(
    "AUROC difference (GFI - Panda)",
    fontweight="bold"
)
ax2.set_title(
    "(b) Paired AUROC differences",
    fontweight="bold",
    pad=9
)
ax2.grid(axis="x", alpha=0.20)
ax2.set_axisbelow(True)

# ------------------------------------------------------------
# Shared legend ABOVE the panels
# ------------------------------------------------------------
handles = [
    Patch(color=BLUE, label="Robot-independent GFI"),
    Patch(color=ORANGE, label="Panda-specific baseline"),
    Line2D(
        [0], [0],
        linestyle=":",
        color="gray",
        label="Chance level"
    )
]

fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    ncol=3,
    frameon=False,
    fontsize=10
)

# Balanced margins avoid clipping and overlapping
fig.subplots_adjust(
    left=0.075,
    right=0.975,
    bottom=0.18,
    top=0.78,
    wspace=0.64
)

# No duplicate footer:
# The manuscript caption already explains the figure.

for ext in ("pdf", "png"):
    filename = (
        f"ICARA2027_GFI_vs_Panda_FINAL_v3.{ext}"
    )
    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.18
    )
    print("Saved:", filename)

plt.close(fig)

print("Figure 9 corrected successfully.")
print("Original AUROC and CI values retained.")
