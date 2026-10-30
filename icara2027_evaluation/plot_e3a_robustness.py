
import csv
from pathlib import Path
import numpy as np
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

results = {}
with open(INPUT, newline="") as f:
    for row in csv.DictReader(f):
        results[(row["object"], row["level"])] = float(
            row["mean_top10_overlap"]
        )

fig, ax = plt.subplots(figsize=(7.2, 4.2))

markers = ["o", "s", "^", "D", "v", "P"]
styles = ["-", "--", "-.", ":", "-", "--"]

for i, obj in enumerate(OBJECTS):
    y = [results[(obj, level)] for level in LEVELS]
    ax.plot(
        range(3), y,
        marker=markers[i],
        linestyle=styles[i],
        linewidth=1.8,
        markersize=6,
        label=obj
    )

ax.set_xticks(range(3))
ax.set_xticklabels([
    "Small\n2 mm / 1°",
    "Moderate\n5 mm / 3°",
    "Large\n10 mm / 5°"
])

ax.set_ylabel("Mean Top-10 retention (out of 10)")
ax.set_xlabel("Pose perturbation level")
ax.set_ylim(0, 10.5)
ax.grid(True, alpha=0.25)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.23),
    ncol=3,
    frameon=False,
    fontsize=9
)

fig.tight_layout()

png_path = BASE / "E3a_Robustness_ICARA2027.png"
pdf_path = BASE / "E3a_Robustness_ICARA2027.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

print("Saved PNG:", png_path)
print("Saved PDF:", pdf_path)

plt.close(fig)

