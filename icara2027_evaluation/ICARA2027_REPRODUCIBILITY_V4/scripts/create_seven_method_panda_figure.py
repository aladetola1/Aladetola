from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Paths and experimental settings
# ---------------------------------------------------
ROOT = Path.cwd()

SOURCE = ROOT / "icara2027_evaluation" / "step43fr_verified_gfi_grasp_analysis.csv"

OUT = (
    ROOT / "FS_RGMP_RETROSPECTIVE"
    / "FOUR_OBJECT_FIGURES_V3"
)
OUT.mkdir(parents=True, exist_ok=True)

objects = ["banana", "mug", "power_drill", "spoon"]
budgets = [1, 3, 5, 10]

UPRIGHT_THRESHOLD = 0.99

# ---------------------------------------------------
# Load previously recorded Panda evaluation outcomes
# ---------------------------------------------------
df = pd.read_csv(SOURCE)

required = {
    "object", "archive_index", "success", "status",
    "H", "D", "U", "GFI"
}

assert required <= set(df.columns)
assert not df.duplicated(
    ["object", "archive_index"]
).any()

df = df[df["object"].isin(objects)].copy()

assert df["object"].nunique() == 4

# ---------------------------------------------------
# Original geometric scoring definitions
# ---------------------------------------------------
df["GFI_score"] = (
    0.5 * df["H"]
    + 0.3 * df["D"]
    + 0.2 * df["U"]
)

df["Centrality"] = df["D"]
df["Height"] = df["H"]
df["Uprightness"] = df["U"]

# Retain original E5 implementation
df["HeightCentrality"] = (
    0.5 * df["H"] + 0.3 * df["D"]
)

assert np.allclose(
    df["GFI_score"],
    df["GFI"],
    rtol=1e-7,
    atol=1e-8
)

# ---------------------------------------------------
# Retrospective static admissibility
# This is NOT a new prospective Panda test.
# ---------------------------------------------------
valid_statuses = {
    "SUCCESS",
    "APPROACH_FAILED",
    "GRASP_FAILED",
    "NO_GEOMETRIC_CANDIDATE",
    "NO_STATIC_FEASIBLE_CANDIDATE",
}

assert set(df["status"]) <= valid_statuses

df["static_feasible"] = df["status"].isin([
    "SUCCESS",
    "APPROACH_FAILED",
    "GRASP_FAILED",
])

df["upright_eligible"] = (
    df["U"] >= UPRIGHT_THRESHOLD
)

df["success"] = df["success"].astype(int)

# ---------------------------------------------------
# Six displayed ranking methods
# ---------------------------------------------------
methods = {
    "GFI + admissibility": {
        "score": "GFI_score",
        "filter": "static_feasible",
    },
    "Centrality-only": {
        "score": "Centrality",
        "filter": None,
    },
    "Height-only": {
        "score": "Height",
        "filter": None,
    },
    "Uprightness-only": {
        "score": "Uprightness",
        "filter": None,
    },
    "Height + Centrality": {
        "score": "HeightCentrality",
        "filter": None,
    },
    "Centrality + Uprightness": {
        "score": "Centrality",
        "filter": "upright_eligible",
    },
    "Original GFI": {
        "score": "GFI_score",
        "filter": None,
    },
}

records = []
selections = []

# ---------------------------------------------------
# Evaluate selection budgets
# ---------------------------------------------------
for method, config in methods.items():

    score = config["score"]
    filter_column = config["filter"]

    for K in budgets:

        successes = 0
        total = 0

        for obj in objects:

            pool = df[df["object"] == obj].copy()

            if filter_column is not None:
                pool = pool[
                    pool[filter_column]
                ].copy()

            ranked = pool.sort_values(
                [score, "archive_index"],
                ascending=[False, True],
                kind="stable"
            )

            if len(ranked) < K:
                raise ValueError(
                    f"Insufficient candidates: "
                    f"{method}, {obj}, K={K}, "
                    f"available={len(ranked)}"
                )

            chosen = ranked.head(K)

            successes += int(chosen["success"].sum())
            total += len(chosen)

            for rank, (_, row) in enumerate(
                chosen.iterrows(), start=1
            ):
                selections.append({
                    "method": method,
                    "object": obj,
                    "K": K,
                    "rank": rank,
                    "archive_index": int(
                        row["archive_index"]
                    ),
                    "score": float(row[score]),
                    "H": float(row["H"]),
                    "D": float(row["D"]),
                    "U": float(row["U"]),
                    "status": row["status"],
                    "success": int(row["success"]),
                    "static_feasible": bool(
                        row["static_feasible"]
                    ),
                    "upright_eligible": bool(
                        row["upright_eligible"]
                    ),
                })

        records.append({
            "method": method,
            "K": K,
            "successes": successes,
            "total": total,
            "success_rate": 100 * successes / total,
            "selection_filter": (
                filter_column
                if filter_column is not None
                else "none"
            ),
        })

results = pd.DataFrame(records)
selection_df = pd.DataFrame(selections)

# ---------------------------------------------------
# Verify existing published figure values
# ---------------------------------------------------
def verify(method, K, expected):
    value = int(
        results.loc[
            (results["method"] == method)
            & (results["K"] == K),
            "successes"
        ].iloc[0]
    )
    assert value == expected, (
        f"{method}, K={K}: "
        f"expected {expected}, obtained {value}"
    )

verify("GFI + admissibility", 10, 37)
verify("Centrality-only", 10, 35)
verify("Original GFI", 10, 30)

# Check all new upright-filtered selections
assert selection_df.loc[
    selection_df["method"] == "Centrality + Uprightness",
    "upright_eligible"
].all()

# ---------------------------------------------------
# Export complete numerical results
# ---------------------------------------------------
results.to_csv(
    OUT / "panda_seven_method_budget_results.csv",
    index=False
)

selection_df.to_csv(
    OUT / "panda_seven_method_selected_placements.csv",
    index=False
)

print("\n===== SEVEN-METHOD RESULTS =====")
print(results.to_string(index=False))

print("\n===== TOP-10 SUCCESS RATES =====")
print(
    results[results["K"] == 10][[
        "method", "successes", "total", "success_rate"
    ]].to_string(index=False)
)

# ---------------------------------------------------
# Publication-ready figure
# ---------------------------------------------------
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
    "GFI + admissibility": ("#C62828", "o", "-"),
    "Centrality-only": ("#1565C0", "s", "--"),
    "Height-only": ("#2E7D32", "^", "-."),
    "Uprightness-only": ("#6A1B9A", "D", ":"),
    "Height + Centrality": ("#EF6C00", "v", "--"),
    "Centrality + Uprightness": (
        "#008B8B", "P", "-"
    ),
    "Original GFI": (
        "#795548", "X", "--"
    ),
}

fig, ax = plt.subplots(figsize=(7.0, 4.8))

for method in methods:

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
        linewidth=2.0 if method == "GFI + admissibility" else 1.6,
        markersize=5.5,
        label=method,
    )

ax.set_xlabel(
    "Panda selection budget per object, K"
)
ax.set_ylabel("Successful selections (%)")

ax.set_xticks(budgets)
ax.set_xlim(0.8, 10.2)
ax.set_ylim(0, 105)
ax.set_yticks(np.arange(0, 101, 20))

ax.grid(
    True,
    alpha=0.25,
    linestyle=":"
)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    frameon=False,
    fontsize=8
)

fig.subplots_adjust(
    left=0.12,
    right=0.98,
    top=0.96,
    bottom=0.38
)

fig.text(
    0.5,
    0.015,
    "Retrospective Panda outcomes; "
    "GFI uses static-candidate screening. "
    "Centrality + Uprightness: U >= 0.99.",
    ha="center",
    fontsize=7
)

for ext in ["pdf", "png", "svg"]:

    output = (
        OUT / f"figure_9_panda_seven_methods.{ext}"
    )

    fig.savefig(
        output,
        dpi=600 if ext == "png" else None,
        bbox_inches="tight",
        facecolor="white"
    )

    print("Saved:", output)

plt.close(fig)

print("\nCOMPLETED")
print("All original experimental files preserved.")
print("No new Panda simulations executed.")
