import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
INPUT = BASE / "experiment_03_robustness_results.csv"

OBJECTS = [
    "Mug", "Bowl", "Spoon",
    "Power drill", "Plate", "Banana"
]
LEVELS = ["Small", "Moderate", "Large"]

data = {}
with open(INPUT, newline="") as f:
    for row in csv.DictReader(f):
        data[(row["object"], row["level"])] = float(
            row["mean_top10_overlap"]
        )

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.transparent": False,
})

fig, ax = plt.subplots(figsize=(3.45, 2.55))

styles = [
    ("o", "-",  "#1f77b4"),
    ("s", "--", "#ff7f0e"),
    ("^", "-.", "#2ca02c"),
    ("D", ":",  "#d62728"),
    ("v", "-",  "#9467bd"),
    ("P", "--", "#8c564b"),
]

for obj, (marker, linestyle, color) in zip(OBJECTS, styles):
    values = [data[(obj, level)] for level in LEVELS]

    ax.plot(
        [0, 1, 2],
        values,
        marker=marker,
        linestyle=linestyle,
        color=color,
        linewidth=1.3,
        markersize=4.5,
        label=obj
    )

ax.set_xticks([0, 1, 2])
ax.set_xticklabels([
    "2 mm / 1°",
    "5 mm / 3°",
    "10 mm / 5°"
])

ax.set_xlabel("Position / orientation noise (SD)", labelpad=5)
ax.set_ylabel("Mean top-10 retention", labelpad=5)

ax.set_xlim(-0.12, 2.12)
ax.set_ylim(0, 10.5)
ax.set_yticks([0, 2, 4, 6, 8, 10])

ax.grid(
    True,
    linestyle=":",
    linewidth=0.45,
    alpha=0.40
)

# Move legend inside the plotting area.
ax.legend(
    loc="lower left",
    ncol=2,
    frameon=True,
    framealpha=0.90,
    facecolor="white",
    edgecolor="0.8",
    columnspacing=0.8,
    handlelength=1.6,
    borderpad=0.35
)

# Reserve sufficient space for both axis labels.
fig.subplots_adjust(
    left=0.18,
    right=0.97,
    top=0.97,
    bottom=0.20
)

pdf_path = BASE / "E3a_Robustness_IEEE_Compact.pdf"
png_path = BASE / "E3a_Robustness_IEEE_Compact.png"

fig.savefig(
    pdf_path,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.06
)

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.06
)

print("Publication PDF:", pdf_path)
print("Preview PNG:", png_path)

plt.close(fig)
