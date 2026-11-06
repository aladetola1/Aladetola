from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd()
OUT = ROOT / "icara2027_figures"
OUT.mkdir(exist_ok=True)

WINDOWS = Path(
    "/mnt/c/Users/olaol/Downloads/ICARA2027_GFI_FIGURES"
)

# Verified original physical TIAGo results:
# 10 placements/object, 10 attempts/placement/group.
objects = ["Mug", "Bowl", "Spoon", "Overall"]
gfi = np.array([90, 96, 93, 93], dtype=float)
random = np.array([36, 47, 43, 42], dtype=float)

assert np.isclose(gfi[:3].mean(), gfi[3])
assert np.isclose(random[:3].mean(), random[3])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8
})

fig, ax = plt.subplots(figsize=(4.6, 3.0))

x = np.arange(len(objects))
width = 0.34

bars1 = ax.bar(
    x - width/2,
    gfi,
    width,
    label="GFI",
    color="#C62828"
)

bars2 = ax.bar(
    x + width/2,
    random,
    width,
    label="Random stable",
    color="#1565C0"
)

for bars in (bars1, bars2):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            value + 1.5,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8
        )

ax.set_ylabel("Physical grasp success (%)")
ax.set_xticks(x)
ax.set_xticklabels(objects)
ax.set_ylim(0, 110)
ax.set_yticks(np.arange(0, 101, 20))

ax.grid(axis="y", linestyle=":", alpha=0.35)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.axvline(
    2.5,
    color="gray",
    linestyle=":",
    linewidth=0.8
)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    frameon=False
)

fig.subplots_adjust(
    left=0.14,
    right=0.98,
    top=0.97,
    bottom=0.28
)

name = "figure12_tiago_gfi_vs_random"

for ext in ("pdf", "png", "svg"):
    path = OUT / f"{name}.{ext}"

    fig.savefig(
        path,
        dpi=600 if ext == "png" else None,
        bbox_inches="tight",
        facecolor="white"
    )

    print("Saved:", path)

    if WINDOWS.is_dir():
        shutil.copy2(path, WINDOWS / path.name)
        print("Copied to Windows:", path.name)

plt.close(fig)

print("\nFIGURE 12 GENERATED SUCCESSFULLY")
print("Original physical TIAGo measurements preserved.")
print("No existing Figure 12 overwritten.")
