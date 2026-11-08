from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd()
OUT = ROOT / "icara2027_evaluation" / "FINAL_FIGURES"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "gfi": "#278653",
    "admissible": "#B82929",
    "centrality": "#2775BA",
    "random": "#D99544",
    "tilted": "#4779B9",
    "upright": "#D99544",
}

def save(fig, name, data):
    for ext in ("pdf", "png", "svg"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(
            path,
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
            facecolor="white"
        )
        print("Saved:", path)
    data.to_csv(OUT / f"{name}.csv", index=False)
    plt.close(fig)

# ====================================================
# FIG. 10 — PANDA: THREE RANKING/SELECTION POLICIES
# ====================================================

source = (
    ROOT / "FS_RGMP_RETROSPECTIVE"
    / "FOUR_OBJECT_FIGURES_V3"
    / "panda_seven_method_budget_results.csv"
)

df = pd.read_csv(source)

methods = [
    ("Original GFI", "Original GFI", COLORS["gfi"]),
    ("Centrality-only", "Centrality-only", COLORS["centrality"]),
    ("GFI + admissibility",
     "GFI + retrospective admissibility",
     COLORS["admissible"]),
]

fig, ax = plt.subplots(figsize=(6.4, 3.8))

for method, label, color in methods:
    d = df[df["method"] == method].sort_values("K")

    assert d["K"].tolist() == [1, 3, 5, 10], (
        "Unexpected selection budgets: " + method
    )

    ax.plot(
        d["K"], d["success_rate"],
        marker="o",
        linewidth=2,
        markersize=5,
        label=label,
        color=color
    )

# Verify established K=10 values
expected = {
    "Original GFI": 30,
    "Centrality-only": 35,
    "GFI + admissibility": 37,
}

for method, successes in expected.items():
    row = df[
        (df["method"] == method) & (df["K"] == 10)
    ]
    assert len(row) == 1
    assert int(row.iloc[0]["successes"]) == successes

ax.set_xlabel("Panda selection budget per object, K")
ax.set_ylabel("Successful selections (%)")
ax.set_xticks([1, 3, 5, 10])
ax.set_ylim(0, 105)
ax.set_yticks(np.arange(0, 101, 20))
ax.grid(alpha=0.25, linestyle=":")
ax.legend(loc="lower right", frameon=False, fontsize=8)
fig.tight_layout()

save(
    fig,
    "Fig10_Panda_GFI_Centrality_Admissibility",
    df[df["method"].isin([m[0] for m in methods])].copy()
)

# ====================================================
# FIG. 12 — PHYSICAL TIAGO: GFI VERSUS RANDOM
# ====================================================

physical = pd.DataFrame({
    "object": ["Mug", "Bowl", "Spoon", "Overall"],
    "GFI_successes": [90, 96, 93, 279],
    "Random_successes": [36, 47, 43, 126],
    "attempts_per_method": [100, 100, 100, 300],
})

physical["GFI_rate"] = (
    100 * physical["GFI_successes"]
    / physical["attempts_per_method"]
)
physical["Random_rate"] = (
    100 * physical["Random_successes"]
    / physical["attempts_per_method"]
)

fig, ax = plt.subplots(figsize=(6.3, 3.8))

x = np.arange(len(physical))
w = 0.32

for dx, col, label, color in [
    (-w/2, "GFI_rate", "GFI-ranked", COLORS["gfi"]),
    (w/2, "Random_rate", "Random stable", COLORS["random"]),
]:
    bars = ax.bar(
        x + dx, physical[col], w,
        label=label, color=color
    )
    for bar, rate in zip(bars, physical[col]):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1.5,
            f"{rate:.0f}%",
            ha="center",
            fontsize=8
        )

ax.set_xticks(x)
ax.set_xticklabels(physical["object"])
ax.set_ylim(0, 110)
ax.set_ylabel("Physical grasp success (%)")
ax.grid(axis="y", alpha=0.2, linestyle=":")
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=2)
fig.tight_layout()

save(fig, "Fig12_TIAGo_GFI_vs_Random", physical)

# ====================================================
# NEW FIGURE — PHYSICAL TIAGO MUG
# ====================================================

mug = pd.DataFrame({
    "condition": [
        "Original GFI",
        "Central tilted",
        "Central upright"
    ],
    "success_rate": [90.0, 80.0, 70.0],
    "number_of_placements": [10, 1, 1],
    "attempts": [100, 10, 10],
    "successful_attempts": [90, 8, 7],
})

fig, ax = plt.subplots(figsize=(5.6, 3.8))

colors = [
    COLORS["gfi"],
    COLORS["tilted"],
    COLORS["upright"]
]

bars = ax.bar(
    mug["condition"],
    mug["success_rate"],
    color=colors,
    width=0.57
)

for bar, row in zip(bars, mug.itertuples()):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        row.success_rate + 1.5,
        f"{row.success_rate:.0f}%\n"
        f"({row.successful_attempts}/{row.attempts})",
        ha="center",
        fontsize=8
    )

ax.set_ylim(0, 110)
ax.set_ylabel("Physical grasp success (%)")
ax.grid(axis="y", alpha=0.2, linestyle=":")
ax.set_axisbelow(True)
fig.tight_layout()

save(fig, "TIAGo_Mug_Orientation_Validation", mug)

print("\nALL THREE FIGURES GENERATED")
