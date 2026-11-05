from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# FIGURE 9 — COMPACT VERSION v7
# Original numerical results preserved
# Vertical values inside bars, regular font, no Type 3

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})

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

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(10.8, 3.65),
    gridspec_kw={"width_ratios": [1.05, 1.0]}
)

# PANEL (a) — AUROC
x = np.arange(len(objects))
width = 0.32

bars1 = ax1.bar(
    x - width/2, gfi, width,
    color=BLUE
)
bars2 = ax1.bar(
    x + width/2, panda, width,
    color=ORANGE
)

ax1.axhline(
    0.5, color="gray",
    linestyle=":", linewidth=1.0
)

ax1.set_ylim(0, 1.12)
ax1.set_ylabel("AUROC")
ax1.set_xticks(x)
ax1.set_xticklabels(objects)
ax1.grid(axis="y", alpha=0.15)
ax1.set_axisbelow(True)

# ALL values placed vertically INSIDE their bars.
# Each value stays inside its own bar without overlap.
for bars in (bars1, bars2):
    for bar in bars:
        value = bar.get_height()
        center = bar.get_x() + bar.get_width() / 2

        ax1.text(
            center,
            0.06,
            f"{value:.3f}",
            rotation=90,
            rotation_mode="anchor",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="normal",
            color="white",
            zorder=5
        )

# PANEL (b) — Paired differences
y = np.arange(len(names))[::-1]

ax2.errorbar(
    delta, y,
    xerr=xerr,
    fmt="o",
    color=BLUE,
    markersize=5,
    linewidth=1.5,
    capsize=3
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

# One shared legend, close to panels
handles = [
    Patch(facecolor=BLUE, label="Robot-independent GFI"),
    Patch(facecolor=ORANGE, label="Panda-specific baseline"),
    Line2D(
        [0], [0],
        color="gray",
        linestyle=":",
        label="Chance level"
    )
]

fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    ncol=3,
    frameon=False,
    fontsize=9
)

fig.subplots_adjust(
    left=0.07,
    right=0.98,
    top=0.89,
    bottom=0.22,
    wspace=0.60
)

# Panel labels below plots, regular font
for ax, caption in [
    (ax1, "(a) AUROC comparison"),
    (ax2, "(b) Paired AUROC differences")
]:
    box = ax.get_position()
    fig.text(
        (box.x0 + box.x1) / 2,
        0.025,
        caption,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="normal"
    )

# Export vector PDF and PNG
for ext in ("pdf", "png"):
    output = Path(
        f"ICARA2027_GFI_vs_Panda_FINAL_v7.{ext}"
    )
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06
    )
    print("Saved:", output.resolve())

plt.close(fig)

print("Figure 9 v7 successfully generated.")
print("Scientific values and confidence intervals unchanged.")
