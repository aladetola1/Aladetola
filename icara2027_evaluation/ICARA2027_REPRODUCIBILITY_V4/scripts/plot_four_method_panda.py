from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
})

ROOT = Path("icara2027_evaluation")
OUT = ROOT / "FINAL_FIGURES"
OUT.mkdir(parents=True, exist_ok=True)

panda = pd.read_csv(
    ROOT / "ICARA2027_REPRODUCIBILITY_V2"
    / "data/panda/panda_154_placement_records.csv"
)

selected = pd.read_csv(
    ROOT / "experiment_05_selected_placements.csv"
)

admissible = pd.read_csv(
    ROOT / "FS_RGMP_RETROSPECTIVE"
    / "fs_rgmp_selected_placements.csv"
)

objects = ["banana", "mug", "power_drill", "spoon"]

methods = [
    "Centrality-only",
    "Upright-filtered centrality",
    "GFI",
    "GFI + admissibility",
]

keys = ["object", "archive_index"]

if panda.duplicated(keys).any():
    raise ValueError("Duplicate Panda placement records")

rows = []

for obj in objects:
    for method in methods:

        if method == "GFI + admissibility":
            group = admissible[
                (admissible["object"] == obj)
                & (admissible["method"] == "GFI")
                & (admissible["K"] == 10)
                & (admissible["screened"] == True)
            ]

            n = len(group)
            successes = int(group["success"].sum())

            if n != 10:
                raise ValueError(
                    f"Expected 10 admissible selections: {obj}"
                )

            source = "Robot-specific assessment"

        else:
            source_method = (
                "GFI"
                if method == "Upright-filtered centrality"
                else method
            )

            group = selected[
                (selected["object"] == obj)
                & (selected["method"] == source_method)
            ][keys]

            if len(group) != 10:
                raise ValueError(
                    f"Expected 10 placements: {obj}"
                )

            matched = group.merge(
                panda[keys + ["success"]],
                on=keys,
                how="left",
                validate="one_to_one"
            )

            observed = matched.dropna(subset=["success"])

            n = len(observed)
            successes = int(observed["success"].sum())

            source = (
                "Matched GFI outcomes"
                if method == "Upright-filtered centrality"
                else "Retrospective Panda outcomes"
            )

        rows.append({
            "object": obj,
            "method": method,
            "successes": successes,
            "evaluated": n,
            "missing": 10 - n,
            "success_rate": (
                100 * successes / n if n else np.nan
            ),
            "source": source,
        })

df = pd.DataFrame(rows)

colors = [
    "#3478BD",
    "#E69F49",
    "#398B65",
    "#8A65B8",
]

fig, ax = plt.subplots(figsize=(10.5, 5.5))

x = np.arange(len(objects))
width = 0.19

for j, method in enumerate(methods):

    group = (
        df[df.method == method]
        .set_index("object")
        .loc[objects]
    )

    positions = x + (j - 1.5) * width
    values = group["success_rate"].to_numpy(float)

    bars = ax.bar(
        positions,
        values,
        width=width,
        color=colors[j],
        edgecolor="white",
        linewidth=0.5,
        label=method
    )

    for bar, value in zip(bars, values):
        if np.isnan(value):
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8
        )

ax.set_xticks(x)
ax.set_xticklabels([
    "Banana", "Mug", "Power drill", "Spoon"
])

ax.set_ylabel("Recorded grasp-success rate (%)")
ax.set_ylim(0, 115)
ax.set_yticks([0, 25, 50, 75, 100])

ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.13),
    ncol=4,
    frameon=False
)

fig.subplots_adjust(
    left=0.08,
    right=0.99,
    bottom=0.23,
    top=0.82
)

fig.text(
    0.5, 0.075,
    "GFI + admissibility uses a separate "
    "robot-specific candidate-selection protocol.",
    ha="center",
    fontsize=8
)

fig.text(
    0.5, 0.045,
    "Upright-filtered centrality reuses GFI outcomes; "
    "retrospective denominators may be below 10.",
    ha="center",
    fontsize=8
)

pdf = OUT / "Panda_Four_Method_Grasp_Success.pdf"
png = OUT / "Panda_Four_Method_Grasp_Success.png"
csv = OUT / "Panda_Four_Method_Grasp_Success.csv"

fig.savefig(pdf, format="pdf", facecolor="white")
fig.savefig(png, dpi=600, facecolor="white")
df.to_csv(csv, index=False)

plt.close(fig)

print("\nFOUR-METHOD FIGURE GENERATED")
print(pdf.resolve())
print(png.resolve())
print(csv.resolve())
print("\nRESULTS")
print(df.to_string(index=False))
