from pathlib import Path
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# FIGURE 7 — PUBLICATION-READABLE VERSION
# Original placement-level calculations preserved
# ============================================================

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "E3c_disturbance_all_objects.csv"

with SOURCE.open(newline="") as f:
    rows = list(csv.DictReader(f))

objects = ["mug", "bowl", "spoon", "power_drill", "plate", "banana"]
labels = ["Mug", "Bowl", "Spoon", "Drill", "Plate", "Banana"]
groups = ["GFI Top-10", "Random"]
colors = ["#386cb0", "#e69f00"]

# Each placement is one independent observational unit.
# Take its maximum rotation across the 12 disturbance conditions.
placement_max = defaultdict(float)

for row in rows:
    key = (row["object"], row["group"], row["archive_index"])
    placement_max[key] = max(
        placement_max[key],
        float(row["rotation_deg"])
    )

# TrueType fonts — no Type 3
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11
})

# Larger canvas, still two panels side by side
fig, axes = plt.subplots(
    1, 2,
    figsize=(11.5, 4.5),
    gridspec_kw={"wspace": 0.32}
)

# ------------------------------------------------------------
# Panel (a): severe rotation by object
# ------------------------------------------------------------
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
        color=colors[j]
    )

axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=25, ha="right")
axes[0].set_ylabel("Placements (>1°)")
axes[0].set_ylim(0, 10)
axes[0].set_title("(a) Severe rotation by object")
axes[0].legend(loc="upper left", frameon=False)

# ------------------------------------------------------------
# Panel (b): threshold sensitivity
# ------------------------------------------------------------
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
        markersize=7,
        linewidth=2.3,
        label=group,
        color=colors[j]
    )

axes[1].set_xticks(np.arange(len(thresholds)))
axes[1].set_xticklabels([str(t) for t in thresholds])
axes[1].set_xlabel("Rotation threshold (degrees)")
axes[1].set_ylabel("Placements")
axes[1].set_ylim(bottom=0)
axes[1].set_title("(b) Threshold sensitivity")
axes[1].legend(loc="upper right", frameon=False)

for ax in axes:
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11)

fig.subplots_adjust(
    left=0.08,
    right=0.98,
    bottom=0.20,
    top=0.88,
    wspace=0.32
)

# Preserve original files; write new versions
for extension in ["png", "pdf"]:
    output = ROOT / f"E3c_Disturbance_IEEE_v2.{extension}"
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )
    print("Saved:", output)

plt.close(fig)

print("Disturbance trial records:", len(rows))
print("Distinct placements:", len(placement_max))
print("Figure 7 successfully regenerated.")
