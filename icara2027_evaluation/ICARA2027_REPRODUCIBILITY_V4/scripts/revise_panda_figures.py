import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("icara2027_evaluation/FINAL_FIGURES")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white"
})

# ==============================================
# FIGURE 9: SEVEN-METHOD PANDA LINE CHART
# ==============================================

K = np.array([1, 3, 5, 10])

series = [
    ("GFI + retrospective admissibility",
     [100, 83.3333, 90, 92.5], "#C62828", "o", "-"),

    ("Centrality-only (tilted poses permitted)",
     [75, 66.6667, 80, 87.5], "#1976D2", "s", "--"),

    ("Height-only",
     [100, 91.6667, 85, 72.5], "#388E3C", "^", "-."),

    ("Uprightness-only",
     [100, 100, 90, 87.5], "#7B1FA2", "D", ":"),

    ("Height + Centrality",
     [100, 91.6667, 85, 87.5], "#F57C00", "v", "--"),

    ("Centrality-only (upright poses permitted)",
     [75, 66.6667, 75, 75], "#00897B", "P", "-."),

    ("Original GFI",
     [75, 66.6667, 75, 75], "#795548", "X", "-")
]

fig, ax = plt.subplots(figsize=(11.5, 6.8))

for label, values, color, marker, style in series:
    ax.plot(
        K, values,
        label=label,
        color=color,
        marker=marker,
        linestyle=style,
        linewidth=2.5,
        markersize=8
    )

ax.set_xlabel("Panda selection budget per object, K")
ax.set_ylabel("Successful selections (%)")
ax.set_xticks(K)
ax.set_ylim(0, 105)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.grid(alpha=0.2)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=2,
    fontsize=13.5,
    frameon=False,
    columnspacing=2.0,
    handlelength=2.5,
    handletextpad=0.7
)

fig.subplots_adjust(
    left=0.10, right=0.98,
    top=0.96, bottom=0.36
)

for ext in ("pdf", "png", "svg"):
    fig.savefig(
        OUT / f"figure_9_panda_seven_methods_LARGE.{ext}",
        dpi=600,
        bbox_inches="tight"
    )

plt.close(fig)

# ==============================================
# FIGURE 10: FOUR-METHOD PANDA BAR CHART
# ==============================================

objects = ["Banana", "Mug", "Power drill", "Spoon"]

methods = [
    ("Centrality-only (tilted poses permitted)",
     [77.8, 77.8, 100, 85.7], "#3478BD"),

    ("Centrality-only (upright poses permitted)",
     [66.7, 40, 100, 85.7], "#E69F49"),

    ("Original GFI",
     [66.7, 40, 100, 85.7], "#398B65"),

    ("GFI + retrospective admissibility",
     [90, 90, 100, 90], "#8A65B8")
]

x = np.arange(len(objects))
width = 0.19

fig, ax = plt.subplots(figsize=(12.5, 6.8))

for i, (label, values, color) in enumerate(methods):
    offset = (i - 1.5) * width

    bars = ax.bar(
        x + offset,
        values,
        width,
        color=color,
        label=label
    )

    # White percentages INSIDE the bars
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 4,
            f"{value:.1f}%",
            ha="center",
            va="top",
            rotation=90,
            fontsize=13.5,
            fontweight="bold",
            color="white"
        )

ax.set_ylabel("Recorded grasp-success rate (%)")
ax.set_xticks(x)
ax.set_xticklabels(objects, fontsize=17)
ax.set_ylim(0, 108)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

# Full-width labels, TWO columns
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    fontsize=13,
    frameon=False,
    columnspacing=1.8,
    handletextpad=0.6
)

fig.subplots_adjust(
    left=0.10, right=0.98,
    top=0.78, bottom=0.12
)

for ext in ("pdf", "png", "svg"):
    fig.savefig(
        OUT / f"Panda_Four_Method_Grasp_Success_LARGE.{ext}",
        dpi=600,
        bbox_inches="tight"
    )

plt.close(fig)

print("\nSUCCESS: Both Panda figures generated.")
print("Output directory:", OUT.resolve())
