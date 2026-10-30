from pathlib import Path
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "E3c_disturbance_all_objects.csv"

with SOURCE.open(newline="") as f:
    rows = list(csv.DictReader(f))

objects = ["mug", "bowl", "spoon", "power_drill", "plate", "banana"]
labels = ["Mug", "Bowl", "Spoon", "Drill", "Plate", "Banana"]
groups = ["GFI Top-10", "Random"]

# Each placement is one independent observational unit.
# Its maximum rotation is taken across the 12 disturbance conditions.
placement_max = defaultdict(float)

for row in rows:
    key = (row["object"], row["group"], row["archive_index"])
    placement_max[key] = max(
        placement_max[key],
        float(row["rotation_deg"])
    )

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw={"wspace": 0.48})

# Panel (a): per-object severe rotation
x = np.arange(len(objects))
width = 0.36

for j, group in enumerate(groups):
    counts = [
        sum(
            value > 1.0
            for (obj, grp, idx), value in placement_max.items()
            if obj == name and grp == group
        )
        for name in objects
    ]

    axes[0].bar(
        x + (j - 0.5) * width,
        counts,
        width,
        label=group,
        color=["#386cb0", "#e69f00"][j]
    )

axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=35, ha="right")
axes[0].set_ylabel("Placements (>1°)")
axes[0].set_ylim(0, 10)
axes[0].set_title("(a) Severe rotation by object")
axes[0].legend(fontsize=7, frameon=False)

# Panel (b): threshold sensitivity
thresholds = [0.1, 0.5, 1, 2, 5, 10, 30, 90]

for j, group in enumerate(groups):
    values = [
        sum(
            value > threshold
            for (obj, grp, idx), value in placement_max.items()
            if grp == group
        )
        for threshold in thresholds
    ]

    axes[1].plot(
        np.arange(len(thresholds)),
        values,
        marker=["o", "s"][j],
        markersize=4,
        linewidth=1.5,
        label=group,
        color=["#386cb0", "#e69f00"][j]
    )

axes[1].set_xticks(np.arange(len(thresholds)))
axes[1].set_xticklabels([str(t) for t in thresholds])
axes[1].set_xlabel("Rotation threshold (degrees)")
axes[1].set_ylabel("Placements")
axes[1].set_ylim(bottom=0)
axes[1].set_title("(b) Threshold sensitivity")
axes[1].legend(fontsize=7, frameon=False)

for ax in axes:
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

for ax in axes:
    ax.tick_params(axis="both", labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.title.set_size(9)

fig.subplots_adjust(left=0.09, right=0.99, bottom=0.27, top=0.84, wspace=0.48)

for extension in ["png", "pdf"]:
    output = ROOT / f"E3c_Disturbance_IEEE.{extension}"
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )
    print("Saved:", output)

plt.close(fig)

print("Distinct placements:", len(placement_max))
