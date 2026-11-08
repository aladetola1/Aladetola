
from pathlib import Path
import csv
import pickle
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

WEIGHTS = {
    "Original RGMP": (0.50, 0.30, 0.20),
    "Height emphasis": (0.60, 0.25, 0.15),
    "Centrality emphasis": (0.40, 0.40, 0.20),
    "Uprightness emphasis": (0.40, 0.25, 0.35),
    "Equal weights": (1/3, 1/3, 1/3),
}


def load_components(folder):
    path = ROOT / "paper_runs" / folder / "data_export.pkl"

    with open(path, "rb") as f:
        data = pickle.load(f)

    components = []

    for pose in data["final_6dof_pos"]:
        xyz = np.asarray(pose["xyz"], dtype=float)

        if xyz[2] <= 0:
            continue

        roll, pitch, _ = np.asarray(
            pose["euler_rpy"], dtype=float
        )

        H = xyz[2]
        D = np.exp(-np.linalg.norm(xyz[:2]) / 0.1)
        U = np.exp(-np.hypot(roll, pitch))

        components.append([H, D, U])

    return np.asarray(components, dtype=float)


results = []

print("\nICARA 2027 — EXPERIMENT E2")
print("Parameter sensitivity of RGMP rankings")
print("=" * 75)

for object_name, folder in OBJECTS.items():
    components = load_components(folder)

    original = components @ np.array(
        WEIGHTS["Original RGMP"]
    )

    original_top10 = set(
        np.argsort(-original, kind="stable")[:10]
    )

    print(f"\n{object_name} ({len(components)} placements)")
    print(f"{'Configuration':23s} {'Overlap':>10s} {'Spearman':>12s}")
    print("-" * 49)

    for configuration, weights in WEIGHTS.items():
        scores = components @ np.array(weights)

        top10 = set(
            np.argsort(-scores, kind="stable")[:10]
        )

        overlap = len(original_top10 & top10)
        rho = float(spearmanr(original, scores).statistic)

        print(
            f"{configuration:23s} "
            f"{overlap:>7d}/10 "
            f"{rho:>12.4f}"
        )

        results.append({
            "object": object_name,
            "configuration": configuration,
            "weight_H": weights[0],
            "weight_D": weights[1],
            "weight_U": weights[2],
            "top10_overlap": overlap,
            "spearman_rho": rho,
        })

output_file = OUTPUT / "experiment_02_sensitivity_results.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to:")
print(output_file)

