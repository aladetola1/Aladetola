from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
EVAL = HERE.parents[2]
PACKAGE = HERE.parents[1]
RESULTS = PACKAGE / "results"

SOURCE = EVAL / "step43fr_verified_gfi_grasp_analysis.csv"
EXPECTED = RESULTS / "step48_gfi_du_ablation_summary.csv"

df = pd.read_csv(SOURCE)

required = [
    "object", "archive_index", "success",
    "H", "D", "U", "GFI"
]

assert len(df) == 154
assert all(c in df.columns for c in required)
assert not df[required].isna().any().any()
assert not df.duplicated(
    ["object", "archive_index"]
).any()

assert set(df["success"].unique()).issubset({0, 1})

df["GFI_DU"] = 0.3 * df["D"] + 0.2 * df["U"]
df["GFI_check"] = 0.5 * df["H"] + df["GFI_DU"]

max_error = np.max(
    np.abs(df["GFI"] - df["GFI_check"])
)
assert max_error < 1e-10

methods = {
    "Full GFI": "GFI",
    "Centrality+Uprightness": "GFI_DU",
    "Centrality": "D",
}

rows = []

for obj, group in df.groupby("object"):
    ranked = {
        name: group.sort_values(
            [column, "archive_index"],
            ascending=[False, True],
            kind="mergesort"
        )
        for name, column in methods.items()
    }

    full_top10 = ranked["Full GFI"].head(10)[
        "archive_index"
    ].tolist()

    du_top10 = ranked[
        "Centrality+Uprightness"
    ].head(10)["archive_index"].tolist()

    assert full_top10 == du_top10, (
        f"Top-10 ordering differs for {obj}"
    )

    for k in (1, 3, 5, 10):
        full = set(
            ranked["Full GFI"].head(k)["archive_index"]
        )

        for method, selected in ranked.items():
            top = selected.head(k)

            rows.append({
                "method": method,
                "K": k,
                "successes": int(top["success"].sum()),
                "evaluations": k,
                "overlap": len(
                    full.intersection(top["archive_index"])
                ),
            })

actual = (
    pd.DataFrame(rows)
    .groupby(["method", "K"], as_index=False)
    .agg(
        successes=("successes", "sum"),
        evaluations=("evaluations", "sum"),
        overlap=("overlap", "sum"),
    )
)

expected = pd.read_csv(EXPECTED)

columns = [
    "method", "K", "successes",
    "evaluations", "overlap"
]

actual = actual[columns].sort_values(
    ["method", "K"]
).reset_index(drop=True)

expected = expected[columns].sort_values(
    ["method", "K"]
).reset_index(drop=True)

pd.testing.assert_frame_equal(
    actual, expected, check_dtype=False
)

print("=" * 65)
print("STEP 48 — GFI HEIGHT-REMOVAL ABLATION VERIFIED")
print("=" * 65)
print("Panda placements:", len(df))
print("Maximum score reconstruction error:", max_error)
print("Full GFI and GFI_DU top-10 ordering identical")
print("for all five Panda objects.")
print()
print(actual.to_string(index=False))
print()
print("Saved Step 48 summary matches recomputed results.")
print("Original experimental files remain unchanged.")
