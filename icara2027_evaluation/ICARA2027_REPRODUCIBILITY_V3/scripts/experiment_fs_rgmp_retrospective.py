from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# New analysis only. Never overwrite previous experiments.
ROOT = Path.cwd()
OUT = ROOT / "FS_RGMP_RETROSPECTIVE"
OUT.mkdir(exist_ok=True)

source = ROOT / "step43fr_verified_gfi_grasp_analysis.csv"
df = pd.read_csv(source)

required = {
    "object", "archive_index", "status", "success",
    "H", "D", "U", "GFI",
    "selected_GFI", "selected_Centrality-only"
}
assert required <= set(df.columns)
assert not df.duplicated(["object", "archive_index"]).any()

# Original score definitions, unchanged.
df["score_gfi"] = (
    0.5 * df["H"]
    + 0.3 * df["D"]
    + 0.2 * df["U"]
)
df["score_centrality"] = df["D"]

assert np.allclose(
    df["score_gfi"], df["GFI"],
    atol=1e-8, rtol=1e-7
)

# Static feasibility is reconstructed retrospectively
# from previously recorded Panda outcomes.
STATIC_PASS = {
    "SUCCESS",
    "APPROACH_FAILED",
    "GRASP_FAILED",
}
STATIC_FAIL = {
    "NO_GEOMETRIC_CANDIDATE",
    "NO_STATIC_FEASIBLE_CANDIDATE",
}

unexpected = set(df["status"]) - STATIC_PASS - STATIC_FAIL
assert not unexpected, f"Unexpected status: {unexpected}"

df["static_feasible"] = df["status"].isin(STATIC_PASS)
df["success"] = df["success"].astype(int)

objects = sorted(df["object"].unique())

print("\n===== ORIGINAL SELECTION AUDIT =====")
for method, flag, score in [
    ("GFI", "selected_GFI", "score_gfi"),
    ("Centrality", "selected_Centrality-only",
     "score_centrality"),
]:
    original = df[
        pd.to_numeric(df[flag], errors="raise") == 1
    ]
    recomputed = pd.concat([
        g.sort_values(
            [score, "archive_index"],
            ascending=[False, True],
            kind="stable"
        ).head(10)
        for _, g in df.groupby("object")
    ])

    a = set(zip(original["object"], original["archive_index"]))
    b = set(zip(recomputed["object"],
                recomputed["archive_index"]))

    print(f"\n{method}:")
    print("Recorded selections:", len(a))
    print("Recorded successes:", int(original["success"].sum()))
    print("Recomputed successes:",
          int(recomputed["success"].sum()))
    print("Different selected indices:", len(a ^ b))
    print("Exact selections match:", a == b)

print("\n===== COMMON STATIC FEASIBILITY =====")
print(df.groupby("object")["static_feasible"]
      .agg(["sum", "count"]).to_string())

# Evaluate the exact same tested placement pool for
# both methods. This is RETROSPECTIVE, not independent
# pre-execution screening.
records = []
selected_rows = []
budgets = [1, 3, 5, 10]

for screened in [False, True]:
    for method, score in [
        ("GFI", "score_gfi"),
        ("Centrality-only", "score_centrality")
    ]:
        for K in budgets:
            successes = 0
            total = 0
            complete = True

            for obj in objects:
                pool = df[df["object"] == obj].copy()

                if screened:
                    pool = pool[pool["static_feasible"]]

                pool = pool.sort_values(
                    [score, "archive_index"],
                    ascending=[False, True],
                    kind="stable"
                )

                if len(pool) < K:
                    complete = False
                    continue

                chosen = pool.head(K)
                successes += int(chosen["success"].sum())
                total += len(chosen)

                for rank, (_, row) in enumerate(
                    chosen.iterrows(), start=1
                ):
                    selected_rows.append({
                        "screened": screened,
                        "method": method,
                        "K": K,
                        "object": obj,
                        "rank": rank,
                        "archive_index": int(row["archive_index"]),
                        "score": float(row[score]),
                        "status": row["status"],
                        "success": int(row["success"]),
                        "static_feasible": bool(
                            row["static_feasible"]
                        ),
                    })

            records.append({
                "screened": screened,
                "method": method,
                "K": K,
                "successes": successes,
                "evaluated": total,
                "complete_matched_budget": complete,
                "success_rate": (
                    100 * successes / total
                    if complete and total else np.nan
                ),
            })

results = pd.DataFrame(records)
selected = pd.DataFrame(selected_rows)

results.to_csv(
    OUT / "fs_rgmp_budget_results.csv",
    index=False
)
selected.to_csv(
    OUT / "fs_rgmp_selected_placements.csv",
    index=False
)

print("\n===== RETROSPECTIVE COMPARISON =====")
print(results.to_string(index=False))

# Plot only complete matched-budget comparisons.
fig, ax = plt.subplots(figsize=(4.8, 3.5))

styles = [
    (False, "GFI", "GFI original", "--", "o"),
    (False, "Centrality-only",
     "Centrality original", "--", "s"),
    (True, "GFI", "GFI + filter", "-", "o"),
    (True, "Centrality-only",
     "Centrality + filter", "-", "s"),
]

for screened, method, label, line, marker in styles:
    subset = results[
        (results["screened"] == screened)
        & (results["method"] == method)
        & results["complete_matched_budget"]
    ].sort_values("K")

    if not subset.empty:
        ax.plot(
            subset["K"],
            subset["success_rate"],
            linestyle=line,
            marker=marker,
            linewidth=1.7,
            label=label
        )

ax.set_xlabel("Selection budget per object, K")
ax.set_ylabel("Successful selections (%)")
ax.set_xticks(budgets)
ax.set_ylim(0, 105)
ax.grid(alpha=0.3, linestyle=":")
ax.legend(fontsize=8)
ax.set_title(
    "Retrospective feasibility-screened comparison",
    fontsize=10
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

fig.tight_layout()

for ext in ["pdf", "png"]:
    path = OUT / f"fs_rgmp_retrospective.{ext}"
    fig.savefig(path, dpi=400, bbox_inches="tight")
    print("Saved:", path)

plt.close(fig)

print("\nANALYSIS COMPLETED")
print("Source outcomes unchanged.")
print("No new Panda simulations performed.")
print("Retrospective tested-pool results only.")
