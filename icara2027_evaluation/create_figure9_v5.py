from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============================================================
# FIGURE 9 — CONSISTENT IEEE-STYLE VERSION
# Panel labels below, regular fonts, no Type 3
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

# Verified original data
objects = ["Banana", "Power\nDrill", "Mug", "Spoon", "Bowl"]

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

fig, axes = plt.subplots(
    1, 2,
    figsize=(11.5, 4.7),
    gridspec_kw={"width_ratios": [1.05, 1.0]}
)

ax1, ax2 = axes

# ------------------------------------------------------------
# Panel (a): AUROC comparison
# ------------------------------------------------------------
x = np.arange(len(objects))
width = 0.31

bars1 = ax1.bar(
    x - width / 2, gfi, width,
    color=BLUE,
    label="Robot-independent GFI"
)

bars2 = ax1.bar(
    x + width / 2, panda, width,
    color=ORANGE,
    label="Panda-specific baseline"
)

ax1.axhline(0.5, color="gray", linestyle=":", linewidth=1.2)

ax1.set_ylim(0, 1.15)
ax1.set_ylabel("AUROC")
ax1.set_xticks(x)
ax1.set_xticklabels(objects)
ax1.grid(axis="y", alpha=0.18)
ax1.set_axisbelow(True)

# Numerical values: compact and regular font
# Alternate small vertical offsets to reduce collisions
for bars, offset in [(bars1, 3), (bars2, 9)]:
    for bar in bars:
        value = bar.get_height()
        ax1.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width()/2, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="normal"
        )

# ------------------------------------------------------------
# Panel (b): paired AUROC differences
# ------------------------------------------------------------
y = np.arange(len(names))[::-1]

ax2.errorbar(
    delta,
    y,
    xerr=xerr,
    fmt="o",
    color=BLUE,
    markersize=5,
    linewidth=1.6,
    capsize=4
)

ax2.axvline(
    0,
    color="gray",
    linestyle="--",
    linewidth=1
)

ax2.set_yticks(y)
ax2.set_yticklabels(names)
ax2.set_xlim(-0.85, 0.60)
ax2.set_xlabel("AUROC difference (GFI - Panda)")
ax2.grid(axis="x", alpha=0.18)
ax2.set_axisbelow(True)

# ------------------------------------------------------------
# Shared legend above plots, regular text
# ------------------------------------------------------------
handles = [
    Patch(facecolor=BLUE, label="Robot-independent GFI"),
    Patch(facecolor=ORANGE, label="Panda-specific baseline"),
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

# Reserve room under axes for panel labels
fig.subplots_adjust(
    left=0.07,
    right=0.98,
    top=0.82,
    bottom=0.25,
    wspace=0.61
)

# ------------------------------------------------------------
# Panel descriptions BELOW the individual panels
# ------------------------------------------------------------
for ax, caption in zip(
    axes,
    ["(a) AUROC comparison", "(b) Paired AUROC differences"]
):
    bbox = ax.get_position()
    fig.text(
        (bbox.x0 + bbox.x1) / 2,
        0.065,
        caption,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="normal"
    )

# ------------------------------------------------------------
# Export
# ------------------------------------------------------------
for extension in ("pdf", "png"):
    output = Path(
        f"ICARA2027_GFI_vs_Panda_FINAL_v5.{extension}"
    )
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.12
    )
    print("Saved:", output.resolve())

plt.close(fig)

print("Figure 9 v5 generated successfully.")
