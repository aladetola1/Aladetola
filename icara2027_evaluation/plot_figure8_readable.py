from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# FIGURE 8 — ROBUSTNESS TO POSE PERTURBATIONS
# Compact, readable, regular fonts, no Type 3
# Scientific data unchanged
# ============================================================

BASE = Path(__file__).resolve().parent
INPUT = BASE / "experiment_03_robustness_results.csv"

OBJECTS = [
    "Mug", "Bowl", "Spoon",
    "Power drill", "Plate", "Banana"
]

LEVELS = ["Small", "Moderate", "Large"]

# ------------------------------------------------------------
# Load original experimental measurements
# ------------------------------------------------------------
data = {}

with INPUT.open(newline="") as f:
    for row in csv.DictReader(f):
        key = (row["object"], row["level"])
        data[key] = float(row["mean_top10_overlap"])

assert all(
    (obj, level) in data
    for obj in OBJECTS
    for level in LEVELS
), "Some object/perturbation measurements are missing."

# ------------------------------------------------------------
# Font settings
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.8,
    "savefig.transparent": False
})

# Slightly larger than the original 3.45 x 2.55 inch figure
fig, ax = plt.subplots(figsize=(4.15, 2.95))

styles = [
    ("o", "-",  "#1f77b4"),
    ("s", "--", "#ff7f0e"),
    ("^", "-.", "#2ca02c"),
    ("D", ":",  "#d62728"),
    ("v", "-",  "#9467bd"),
    ("P", "--", "#8c564b")
]

# ------------------------------------------------------------
# Six object-level curves
# ------------------------------------------------------------
for obj, (marker, linestyle, color) in zip(
    OBJECTS, styles
):
    values = [
        data[(obj, level)]
        for level in LEVELS
    ]

    ax.plot(
        [0, 1, 2],
        values,
        marker=marker,
        linestyle=linestyle,
        color=color,
        linewidth=1.5,
        markersize=5.5,
        label=obj
    )

# ------------------------------------------------------------
# Axes
# ------------------------------------------------------------
ax.set_xticks([0, 1, 2])

ax.set_xticklabels([
    "2 mm / 1°",
    "5 mm / 3°",
    "10 mm / 5°"
])

ax.set_xlabel(
    "Position / orientation noise (SD)",
    labelpad=5
)

ax.set_ylabel(
    "Mean top-10 retention",
    labelpad=5
)

ax.set_xlim(-0.12, 2.12)
ax.set_ylim(0, 10.5)
ax.set_yticks([0, 2, 4, 6, 8, 10])

ax.grid(
    True,
    linestyle=":",
    linewidth=0.5,
    alpha=0.35
)

ax.set_axisbelow(True)

# ------------------------------------------------------------
# Compact legend inside plot
# ------------------------------------------------------------
ax.legend(
    loc="lower left",
    ncol=2,
    frameon=True,
    framealpha=0.93,
    facecolor="white",
    edgecolor="0.8",
    columnspacing=0.8,
    handlelength=1.8,
    borderpad=0.35,
    labelspacing=0.3
)

# ------------------------------------------------------------
# Compact margins
# ------------------------------------------------------------
fig.subplots_adjust(
    left=0.18,
    right=0.975,
    top=0.965,
    bottom=0.21
)

# ------------------------------------------------------------
# Export new files
# ------------------------------------------------------------
for ext in ("pdf", "png"):
    output = BASE / f"E3a_Robustness_IEEE_v2.{ext}"

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06
    )

    print("Saved:", output)

plt.close(fig)

print("Figure 8 generated successfully.")
print("Original experimental measurements preserved.")
print("Number of plotted measurements:", len(OBJECTS) * len(LEVELS))
