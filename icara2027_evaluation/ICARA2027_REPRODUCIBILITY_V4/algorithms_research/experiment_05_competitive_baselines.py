from pathlib import Path
import pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent

OBJECTS = [
    "mug", "bowl", "spoon",
    "power_drill", "plate", "banana"
]

METHODS = [
    "GFI",
    "Centrality-only",
    "Height-only",
    "Uprightness-only",
    "Height+Centrality",
]

all_records = []
summary_records = []

for obj in OBJECTS:

    archive_path = ROOT / "paper_runs" / obj / "data_export.pkl"

    with open(archive_path, "rb") as f:
        data = pickle.load(f)

    candidates = []

    for original_index, pose in enumerate(data["final_6dof_pos"]):

        xyz = np.asarray(pose["xyz"], dtype=float)
        rpy = np.asarray(pose["euler_rpy"], dtype=float)

        if not (
            np.all(np.isfinite(xyz))
            and np.all(np.isfinite(rpy))
            and xyz[2] > 0
        ):
            continue

        x, y, z = xyz
        roll, pitch, yaw = rpy

        H = z
        D = np.exp(-np.hypot(x, y) / 0.1)
        U = np.exp(-np.hypot(roll, pitch))

        GFI = 0.5 * H + 0.3 * D + 0.2 * U

        candidates.append({
            "object": obj,
            "archive_index": original_index,
            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "H": H,
            "D": D,
            "U": U,
            "GFI": GFI,
            "Centrality-only": D,
            "Height-only": H,
            "Uprightness-only": U,
            "Height+Centrality": 0.5 * H + 0.3 * D,
        })

    df = pd.DataFrame(candidates)

    assert len(df) >= 10

    print(f"\n{obj.upper()} — {len(df)} candidates")
    print("-" * 65)

    gfi_top10 = set(
        df.sort_values(
            "GFI", ascending=False, kind="stable"
        ).head(10)["archive_index"]
    )

    for method in METHODS:

        selected = df.sort_values(
            method, ascending=False, kind="stable"
        ).head(10).copy()

        selected["method"] = method
        selected["rank"] = np.arange(1, 11)

        all_records.extend(selected.to_dict("records"))

        overlap = len(
            set(selected["archive_index"]) & gfi_top10
        )

        summary_records.append({
            "object": obj,
            "method": method,
            "candidate_count": len(df),
            "overlap_with_gfi_top10": overlap,
            "mean_height_m": selected["H"].mean(),
            "mean_centrality": selected["D"].mean(),
            "mean_uprightness": selected["U"].mean(),
        })

        print(
            f"{method:20s} "
            f"Overlap with GFI: {overlap:2d}/10"
        )

records_path = OUTPUT / "experiment_05_selected_placements.csv"
summary_path = OUTPUT / "experiment_05_baseline_summary.csv"

pd.DataFrame(all_records).to_csv(records_path, index=False)
pd.DataFrame(summary_records).to_csv(summary_path, index=False)

print("\nSaved:")
print(records_path)
print(summary_path)
