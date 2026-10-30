
from pathlib import Path
import pickle
import numpy as np

# Project root: run this script from any directory.
ROOT = Path(__file__).resolve().parent.parent

OBJECTS = {
    "Mug": "mug",
    "Bowl": "bowl",
    "Spoon": "spoon",
    "Power drill": "power_drill",
    "Plate": "plate",
    "Banana": "banana",
}

EXPECTED_COUNTS = {
    "Mug": 481,
    "Bowl": 217,
    "Spoon": 389,
    "Power drill": 549,
    "Plate": 242,
    "Banana": 213,
}


def load_stable_poses(object_folder):
    archive = (
        ROOT / "paper_runs" / object_folder / "data_export.pkl"
    )

    with open(archive, "rb") as f:
        data = pickle.load(f)

    return [
        pose
        for pose in data["final_6dof_pos"]
        if float(pose["xyz"][2]) > 0
    ]


def calculate_components(pose):
    xyz = np.asarray(pose["xyz"], dtype=float)
    roll, pitch, yaw = np.asarray(
        pose["euler_rpy"], dtype=float
    )

    H = xyz[2]
    D = np.exp(-np.linalg.norm(xyz[:2]) / 0.1)
    U = np.exp(-np.sqrt(roll**2 + pitch**2))

    GFI = 0.5 * H + 0.3 * D + 0.2 * U

    return H, D, U, GFI


print("\nICARA 2027 — EXPERIMENT E1")
print("Original RGMP data verification\n")

total = 0

for object_name, folder in OBJECTS.items():
    poses = load_stable_poses(folder)

    components = np.array([
        calculate_components(pose)
        for pose in poses
    ])

    count = len(poses)
    total += count

    assert count == EXPECTED_COUNTS[object_name], (
        f"Unexpected stable-placement count for {object_name}: "
        f"{count}"
    )

    print(
        f"{object_name:12s} | "
        f"Stable poses: {count:4d} | "
        f"Max GFI: {components[:, 3].max():.6f}"
    )

print(f"\nTOTAL STABLE PLACEMENTS: {total}")
print("Verification completed.")


# ------------------------------------------------------------
# Verify reconstructed Top-10 GFI against original CSV results
# ------------------------------------------------------------

import csv

print("\nTOP-10 GFI REPRODUCIBILITY CHECK")
print("-" * 65)

for object_name, folder in OBJECTS.items():

    poses = load_stable_poses(folder)

    reconstructed = np.array([
        calculate_components(pose)[3]
        for pose in poses
    ])

    reconstructed_top10 = np.sort(reconstructed)[::-1][:10]

    csv_path = (
        ROOT / "results" / "top10_gfi_tables"
        / f"{folder}_top10_detailed.csv"
    )

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    original_top10 = np.array([
        float(row["gfi"])
        for row in rows
    ])

    original_top10 = np.sort(original_top10)[::-1][:10]

    assert len(original_top10) == 10, (
        f"Expected 10 original GFI scores for {object_name}"
    )

    max_error = np.max(
        np.abs(reconstructed_top10 - original_top10)
    )

    print(
        f"{object_name:12s} | "
        f"Maximum absolute error: {max_error:.12e}"
    )

    assert max_error < 1e-8, (
        f"GFI mismatch detected for {object_name}"
    )

print("-" * 65)
print("All six objects passed Top-10 GFI verification.")



# ============================================================
# EXPERIMENT E1: GEOMETRIC COMPONENT ABLATION
# ============================================================

from scipy.stats import spearmanr

print("\nEXPERIMENT E1 — ABLATION STUDY")
print("=" * 75)

results = []

for object_name, folder in OBJECTS.items():

    poses = load_stable_poses(folder)

    components = np.array([
        calculate_components(pose)
        for pose in poses
    ])

    H = components[:, 0]
    D = components[:, 1]
    U = components[:, 2]
    GFI = components[:, 3]

    scores = {
        "H-only": H,
        "D-only": D,
        "U-only": U,
    }

    # Stable sorting gives deterministic handling of ties.
    rgmp_top10 = set(
        np.argsort(-GFI, kind="stable")[:10]
    )

    print(f"\n{object_name} ({len(poses)} stable placements)")
    print(f"{'Method':12s} {'Overlap':>12s} {'Spearman':>12s}")
    print("-" * 40)

    for method, values in scores.items():

        method_top10 = set(
            np.argsort(-values, kind="stable")[:10]
        )

        overlap = len(rgmp_top10 & method_top10)

        correlation = spearmanr(values, GFI).statistic

        print(
            f"{method:12s} "
            f"{overlap:>9d}/10 "
            f"{correlation:>12.4f}"
        )

        results.append({
            "object": object_name,
            "method": method,
            "stable_placements": len(poses),
            "top10_overlap": overlap,
            "spearman_rho": correlation,
        })

# Save the ablation results separately.
import csv

output_path = (
    ROOT / "icara2027_evaluation"
    / "experiment_01_ablation_results.csv"
)

with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "object",
            "method",
            "stable_placements",
            "top10_overlap",
            "spearman_rho",
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to:")
print(output_path)
