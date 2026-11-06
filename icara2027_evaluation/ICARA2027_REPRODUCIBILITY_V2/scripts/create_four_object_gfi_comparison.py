from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd()
SOURCE = ROOT / "step43fr_verified_gfi_grasp_analysis.csv"
OUT = ROOT / "FS_RGMP_RETROSPECTIVE" / "FOUR_OBJECT_FIGURES_V1"
OUT.mkdir(parents=True, exist_ok=True)

objects = ["banana", "mug", "power_drill", "spoon"]
budgets = [1, 3, 5, 10]

df = pd.read_csv(SOURCE)

required = {
    "object", "archive_index", "success", "status",
    "H", "D", "U", "GFI"
}
assert required <= set(df.columns)
assert not df.duplicated(["object", "archive_index"]).any()

# Original scoring equations remain unchanged
df["GFI_score"] = (
    0.5 * df["H"] +
    0.3 * df["D"] +
    0.2 * df["U"]
)

df["Centrality"] = df["D"]
df["Height"] = df["H"]
df["Uprightness"] = df["U"]

# Preserve original E5 definition
df["HeightCentrality"] = (
    0.5 * df["H"] + 0.3 * df["D"]
)

assert np.allclose(
    df["GFI_score"], df["GFI"],
    rtol=1e-7, atol=1e-8
)

valid_statuses = {
    "SUCCESS",
    "APPROACH_FAILED",
    "GRASP_FAILED",
    "NO_GEOMETRIC_CANDIDATE",
    "NO_STATIC_FEASIBLE_CANDIDATE",
}

assert set(df["status"]) <= valid_statuses

# Based on recorded outcomes, not new simulations
df["static_feasible"] = df["status"].isin([
    "SUCCESS",
    "APPROACH_FAILED",
    "GRASP_FAILED",
])

df["success"] = df["success"].astype(int)

df = df[df["object"].isin(objects)].copy()

assert df["object"].nunique() == 4

methods = {
    "GFI + feasibility screen": ("GFI_score", True),
    "Centrality-only": ("Centrality", False),
    "Height-only": ("Height", False),
    "Uprightness-only": ("Uprightness", False),
    "Height + Centrality": ("HeightCentrality", False),
    "Centrality + feasibility screen": ("Centrality", True),
    "Original GFI": ("GFI_score", False),
}

records = []
selections = []

for method, (score, screened) in methods.items():
    for K in budgets:
        successes = 0

        for obj in objects:
            pool = df[df["object"] == obj].copy()

            if screened:
                pool = pool[pool["static_feasible"]]

            ranked = pool.sort_values(
                [score, "archive_index"],
                ascending=[False, True],
                kind="stable"
            )

            assert len(ranked) >= K, (method, obj, K)

            top = ranked.head(K)
            successes += int(top["success"].sum())

            for rank, (_, row) in enumerate(
                top.iterrows(), start=1
            ):
                selections.append({
                    "method": method,
                    "object": obj,
                    "K": K,
                    "rank": rank,
                    "archive_index": int(row["archive_index"]),
                    "score": float(row[score]),
                    "status": row["status"],
                    "success": int(row["success"]),
                })

        total = 4 * K

        records.append({
            "method": method,
            "K": K,
            "successes": successes,
            "total": total,
            "success_rate": 100 * successes / total,
            "screened": screened
        })

results = pd.DataFrame(records)

print("\n===== FOUR-OBJECT RESULTS =====")
print(results.to_string(index=False))

# Validate known results before figure creation
def check(method, K, expected):
    subset = results[
        (results["method"] == method)
        & (results["K"] == K)
    ]
    value = int(subset.iloc[0]["successes"])
    assert value == expected, (
        f"{method}: expected {expected}, got {value}"
    )

check("GFI + feasibility screen", 10, 37)
check("Centrality-only", 10, 35)
check("Centrality + feasibility screen", 10, 38)
check("Original GFI", 10, 30)

results.to_csv(
    OUT / "four_object_all_methods_results.csv",
    index=False
)

pd.DataFrame(selections).to_csv(
    OUT / "four_object_selection_audit.csv",
    index=False
)

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

styles = {
    "GFI + feasibility screen": ("#C62828", "o", "-"),
    "Centrality-only": ("#1565C0", "s", "--"),
    "Height-only": ("#2E7D32", "^", "-."),
    "Uprightness-only": ("#6A1B9A", "D", ":"),
    "Height + Centrality": ("#EF6C00", "v", "--"),
    "Centrality + feasibility screen": (
        "#1565C0", "s", "-"
    ),
}

def create_figure(method_names, filename, note):
    fig, ax = plt.subplots(figsize=(5.1, 3.7))

    for method in method_names:
        data = results[
            results["method"] == method
        ].sort_values("K")

        color, marker, linestyle = styles[method]

        ax.plot(
            data["K"],
            data["success_rate"],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.1 if method.startswith("GFI") else 1.5,
            markersize=5,
            label=method
        )

    ax.set_xlabel("Panda selection budget per object, K")
    ax.set_ylabel("Successful selections (%)")
    ax.set_xticks(budgets)
    ax.set_xlim(0.8, 10.2)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(True, alpha=0.3, linestyle=":")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=2,
        frameon=False,
        fontsize=7.5
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        top=0.96,
        bottom=0.39
    )

    fig.text(
        0.5, 0.02, note,
        ha="center", va="bottom",
        fontsize=7
    )

    for ext in ["pdf", "png", "svg"]:
        path = OUT / f"{filename}.{ext}"
        fig.savefig(
            path,
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
            facecolor="white"
        )
        print("Saved:", path)

    plt.close(fig)

# Main figure: five ranking methods
create_figure(
    [
        "GFI + feasibility screen",
        "Centrality-only",
        "Height-only",
        "Uprightness-only",
        "Height + Centrality"
    ],
    "figure_four_objects_gfi_vs_baselines",
    "Post-hoc four-object comparison; only GFI screened"
)

# Fairness analysis: identical screening
create_figure(
    [
        "GFI + feasibility screen",
        "Centrality + feasibility screen",
        "Centrality-only"
    ],
    "figure_four_objects_common_screening",
    "Retrospective tested pool; identical screening applied"
)

print("\n===== FINAL TOP-10 COMPARISON =====")
print(
    results[results["K"] == 10][
        ["method", "successes", "total", "success_rate"]
    ].to_string(index=False)
)

print("\nCOMPLETED")
print("Original experimental files unchanged.")
print("Bowl excluded post hoc.")
print("No new grasp simulations were performed.")
