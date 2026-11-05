from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# READABLE VERSION OF FIGURE 11
# Physical TIAGo placement-level grasp success
# Data preserved from verified CSV files
# ============================================================

# ------------------------------------------------------------
# Load verified data
# ------------------------------------------------------------
gfi = pd.read_csv("step44d_tiago_top10_matched.csv")
rnd = pd.read_csv("step45h_tiago_random_original_verified.csv")

gfi["object"] = gfi["object"].str.lower()
rnd["object"] = rnd["object"].str.lower()

# Keep only the three TIAGo objects
objects = ["mug", "bowl", "spoon"]
pretty = {"mug": "Mug", "bowl": "Bowl", "spoon": "Spoon"}

gfi = gfi[gfi["object"].isin(objects)].copy()
rnd = rnd[rnd["object"].isin(objects)].copy()

# Placement-level grasp success rate (%)
gfi["group"] = "RGMP-ranked"
rnd["group"] = "Random stable"

gfi["success_rate"] = gfi["success_rate_percent"].astype(float)
rnd["success_rate"] = rnd["success_rate_percent"].astype(float)

df = pd.concat([
    gfi[["object", "group", "success_rate"]],
    rnd[["object", "group", "success_rate"]],
], ignore_index=True)

# ------------------------------------------------------------
# Verify means and bootstrap CI from the original placement data
# ------------------------------------------------------------
gfi_mean = gfi["success_rate"].mean()
rnd_mean = rnd["success_rate"].mean()
observed_diff = gfi_mean - rnd_mean

rng = np.random.default_rng(2027)
B = 20000
diffs = np.empty(B)

gfi_arrays = []
rnd_arrays = []

for obj in objects:
    g = gfi.loc[gfi["object"] == obj, "success_rate"].to_numpy()
    r = rnd.loc[rnd["object"] == obj, "success_rate"].to_numpy()
    assert len(g) == 10 and len(r) == 10
    gfi_arrays.append(g)
    rnd_arrays.append(r)

for b in range(B):
    g_means = []
    r_means = []
    for g, r in zip(gfi_arrays, rnd_arrays):
        g_sample = rng.choice(g, size=len(g), replace=True)
        r_sample = rng.choice(r, size=len(r), replace=True)
        g_means.append(g_sample.mean())
        r_means.append(r_sample.mean())
    diffs[b] = np.mean(g_means) - np.mean(r_means)

ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10
})

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 4.3))

positions = [1, 2]
box_data = [
    gfi["success_rate"].to_numpy(),
    rnd["success_rate"].to_numpy()
]

bp = ax.boxplot(
    box_data,
    positions=positions,
    widths=0.55,
    patch_artist=True,
    showfliers=False,
    medianprops={"color": "#E07B39", "linewidth": 2.0},
    whiskerprops={"linewidth": 1.2},
    capprops={"linewidth": 1.2},
    boxprops={"linewidth": 1.2}
)

box_colors = ["#9ecae1", "#e7b6bb"]
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.60)

# ------------------------------------------------------------
# Overlay placement points by object
# ------------------------------------------------------------
marker_map = {"mug": "o", "bowl": "s", "spoon": "^"}
point_color = "#4c4c4c"

rng_jitter = np.random.default_rng(2027)

for x0, group_name in zip(positions, ["RGMP-ranked", "Random stable"]):
    sub = df[df["group"] == group_name]

    for obj in objects:
        vals = sub.loc[sub["object"] == obj, "success_rate"].to_numpy()
        jitter = rng_jitter.uniform(-0.09, 0.09, len(vals))
        ax.scatter(
            np.full(len(vals), x0) + jitter,
            vals,
            s=22,
            marker=marker_map[obj],
            color=point_color,
            alpha=0.70,
            zorder=3
        )

# Means as black diamonds
ax.scatter(
    positions,
    [gfi_mean, rnd_mean],
    marker="D",
    s=38,
    color="black",
    zorder=4
)

# ------------------------------------------------------------
# Axes, labels, annotations
# ------------------------------------------------------------
ax.set_ylabel("Grasp success (%)", labelpad=5)
ax.set_xticks(positions)
ax.set_xticklabels([
    "RGMP-ranked\n(n=30 placements)",
    "Random stable\n(n=30 placements)"
])
ax.set_ylim(-5, 118)
ax.set_yticks(np.arange(0, 101, 20))
ax.grid(axis="y", alpha=0.22)
ax.set_axisbelow(True)

ax.text(1, 108, f"Mean: {gfi_mean:.0f}%", ha="center", va="bottom", fontsize=11)
ax.text(2, 108, f"Mean: {rnd_mean:.0f}%", ha="center", va="bottom", fontsize=11)

# Footer note inside figure, readable
footer = (
    f"Mean difference: {observed_diff:.1f} pp; "
    f"object-stratified placement bootstrap 95% CI: "
    f"[{ci_low:.1f}, {ci_high:.1f}] pp"
)
fig.text(0.5, 0.07, footer, ha="center", va="center", fontsize=9)

# Legend for objects
legend_handles = [
    Line2D([0], [0], marker=marker_map["mug"], color="none",
           markerfacecolor=point_color, markeredgecolor=point_color,
           markersize=5.5, label="Mug"),
    Line2D([0], [0], marker=marker_map["bowl"], color="none",
           markerfacecolor=point_color, markeredgecolor=point_color,
           markersize=5.5, label="Bowl"),
    Line2D([0], [0], marker=marker_map["spoon"], color="none",
           markerfacecolor=point_color, markeredgecolor=point_color,
           markersize=5.5, label="Spoon"),
]
ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
    ncol=3,
    frameon=False,
    borderaxespad=0.0
)

fig.subplots_adjust(left=0.25, right=0.97, top=0.88, bottom=0.30)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
pdf_path = Path("step43gz_tiago_placement_success_boxplot_v4.pdf")
png_path = Path("step43gz_tiago_placement_success_boxplot_v4.png")

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=600, bbox_inches="tight")

print("Saved:", pdf_path.resolve())
print("Saved:", png_path.resolve())
print(f"GFI mean: {gfi_mean:.1f}%")
print(f"Random mean: {rnd_mean:.1f}%")
print(f"Difference: {observed_diff:.1f} pp")
print(f"95% CI: [{ci_low:.1f}, {ci_high:.1f}] pp")

plt.close(fig)
