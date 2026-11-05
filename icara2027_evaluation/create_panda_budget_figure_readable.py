from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# READABLE SINGLE-COLUMN VERSION OF PANDA BUDGET FIGURE
# Original Step 47R data preserved exactly
# ============================================================

output_dir = Path.cwd() / "icara2027_figures"
output_dir.mkdir(parents=True, exist_ok=True)

budgets = np.array([1, 3, 5, 10])
n_objects = 5

# ------------------------------------------------------------
# Original measured successful selections from Step 47R
# ------------------------------------------------------------
results = {
    "GFI (RGMP)": {
        "successes": [3, 8, 15, 30],
        "color": "#C62828",
        "marker": "o",
        "linestyle": "-"
    },
    "Centrality": {
        "successes": [3, 8, 16, 35],
        "color": "#1565C0",
        "marker": "s",
        "linestyle": "--"
    },
    "Height": {
        "successes": [4, 11, 17, 29],
        "color": "#2E7D32",
        "marker": "^",
        "linestyle": "-."
    },
    "Uprightness": {
        "successes": [4, 13, 19, 37],
        "color": "#6A1B9A",
        "marker": "D",
        "linestyle": ":"
    },
    "Height + Centrality": {
        "successes": [4, 11, 17, 35],
        "color": "#EF6C00",
        "marker": "v",
        "linestyle": (0, (5, 2))
    }
}

# ------------------------------------------------------------
# Font setup — avoid Type 3 fonts
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 7.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.8
})

# ------------------------------------------------------------
# Single-column readable layout
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.1, 3.0))

for method, info in results.items():
    successes = np.array(info["successes"], dtype=float)

    # Same formula as original Step 47R
    success_rate = 100 * successes / (n_objects * budgets)

    ax.plot(
        budgets,
        success_rate,
        label=method,
        color=info["color"],
        marker=info["marker"],
        linestyle=info["linestyle"],
        linewidth=1.8 if method == "GFI (RGMP)" else 1.5,
        markersize=5.2 if method == "GFI (RGMP)" else 4.6,
        markerfacecolor="white",
        markeredgewidth=1.0
    )

ax.set_xlabel("Panda evaluation budget per object, K", labelpad=4)
ax.set_ylabel("Successful selections (%)", labelpad=4)

ax.set_xticks(budgets)
ax.set_xlim(0.8, 10.2)
ax.set_ylim(0, 105)
ax.set_yticks(np.arange(0, 101, 20))

ax.grid(
    True,
    linestyle=":",
    linewidth=0.5,
    alpha=0.40
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Compact legend under plot, readable in one-column format
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.24),
    ncol=2,
    frameon=False,
    columnspacing=1.2,
    handlelength=2.4,
    handletextpad=0.6
)

fig.subplots_adjust(
    left=0.16,
    right=0.98,
    top=0.97,
    bottom=0.33
)

name = "icara2027_grasp_success_vs_budget_v2"

for ext in ("pdf", "png", "svg"):
    path = output_dir / f"{name}.{ext}"
    fig.savefig(
        path,
        dpi=600 if ext == "png" else None,
        facecolor="white",
        bbox_inches="tight"
    )
    print("Saved:", path)

plt.close(fig)

print("\nREADABLE PANDA BUDGET FIGURE COMPLETED")
print("Data preserved exactly from original Step 47R")
print("Five objects, matched retrospective selection pool")
print("Evaluation budgets: K = 1, 3, 5, 10")
print("No original experimental results modified")
