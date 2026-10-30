
from pathlib import Path
import pickle
import csv
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent

OBJECTS = {
    "Mug": "mug",
    "Bowl": "bowl",
    "Spoon": "spoon",
    "Power drill": "power_drill",
    "Plate": "plate",
    "Banana": "banana",
}

# Position standard deviation in metres;
# roll/pitch standard deviation in degrees.
LEVELS = {
    "Small": (0.002, 1.0),
    "Moderate": (0.005, 3.0),
    "Large": (0.010, 5.0),
}

N_REPETITIONS = 100
RNG = np.random.default_rng(2027)


def load_poses(folder):
    path = ROOT / "paper_runs" / folder / "data_export.pkl"

    with open(path, "rb") as f:
        data = pickle.load(f)

    poses = [
        p for p in data["final_6dof_pos"]
        if float(p["xyz"][2]) > 0
    ]

    xyz = np.array(
        [p["xyz"] for p in poses], dtype=float
    )

    rpy = np.array(
        [p["euler_rpy"] for p in poses], dtype=float
    )

    return xyz, rpy


def calculate_gfi(xyz, rpy):
    H = xyz[:, 2]

    D = np.exp(
        -np.linalg.norm(xyz[:, :2], axis=1) / 0.1
    )

    U = np.exp(
        -np.linalg.norm(rpy[:, :2], axis=1)
    )

    return 0.5 * H + 0.3 * D + 0.2 * U


results = []

print("\nICARA 2027 — EXPERIMENT E3a")
print("GFI ranking robustness under pose perturbations")
print("=" * 75)

for object_name, folder in OBJECTS.items():

    xyz, rpy = load_poses(folder)

    original_scores = calculate_gfi(xyz, rpy)

    original_top10 = set(
        np.argsort(-original_scores, kind="stable")[:10]
    )

    print(f"\n{object_name} ({len(xyz)} placements)")
    print(
        f"{'Level':12s} {'Overlap':>12s} "
        f"{'Spearman':>12s} {'Mean |dGFI|':>14s}"
    )
    print("-" * 55)

    for level, (sigma_m, sigma_deg) in LEVELS.items():

        overlaps = []
        correlations = []
        score_changes = []

        sigma_rad = np.deg2rad(sigma_deg)

        for _ in range(N_REPETITIONS):

            xyz_noise = RNG.normal(
                0, sigma_m, size=xyz.shape
            )

            rpy_noise = np.zeros_like(rpy)

            rpy_noise[:, :2] = RNG.normal(
                0, sigma_rad,
                size=(len(rpy), 2)
            )

            perturbed_xyz = xyz + xyz_noise
            perturbed_rpy = rpy + rpy_noise

            perturbed_scores = calculate_gfi(
                perturbed_xyz, perturbed_rpy
            )

            perturbed_top10 = set(
                np.argsort(
                    -perturbed_scores, kind="stable"
                )[:10]
            )

            overlaps.append(
                len(original_top10 & perturbed_top10)
            )

            correlations.append(
                spearmanr(
                    original_scores,
                    perturbed_scores
                ).statistic
            )

            score_changes.append(
                np.mean(
                    np.abs(
                        original_scores - perturbed_scores
                    )
                )
            )

        mean_overlap = float(np.mean(overlaps))
        mean_rho = float(np.mean(correlations))
        mean_change = float(np.mean(score_changes))

        print(
            f"{level:12s} "
            f"{mean_overlap:>10.2f}/10 "
            f"{mean_rho:>12.4f} "
            f"{mean_change:>14.6f}"
        )

        results.append({
            "object": object_name,
            "level": level,
            "position_sigma_m": sigma_m,
            "angle_sigma_deg": sigma_deg,
            "repetitions": N_REPETITIONS,
            "mean_top10_overlap": mean_overlap,
            "mean_spearman_rho": mean_rho,
            "mean_absolute_gfi_change": mean_change,
        })


output_file = OUTPUT / "experiment_03_robustness_results.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to:")
print(output_file)
