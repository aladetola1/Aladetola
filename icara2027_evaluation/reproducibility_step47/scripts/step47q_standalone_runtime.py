from pathlib import Path
from time import perf_counter_ns
import csv
import pickle
import numpy as np

HERE = Path(__file__).resolve()
EVAL = HERE.parents[2]
PROJECT = EVAL.parent
OUTPUT = EVAL / "reproducibility_step47" / "results"

OBJECTS = {
    "Mug": "mug",
    "Bowl": "bowl",
    "Spoon": "spoon",
    "Power Drill": "power_drill",
    "Plate": "plate",
    "Banana": "banana",
}

METHODS = [
    "GFI",
    "Centrality",
    "Height",
    "Uprightness",
    "Height+Centrality",
]

def rank(poses, method):
    scores = []

    for pose in poses:
        x, y, z = map(float, pose["xyz"])

        if method == "Height":
            score = z

        elif method == "Centrality":
            score = np.exp(-np.hypot(x, y) / 0.1)

        elif method == "Uprightness":
            roll, pitch = map(
                float, pose["euler_rpy"][:2]
            )
            score = np.exp(-np.hypot(roll, pitch))

        elif method == "Height+Centrality":
            D = np.exp(-np.hypot(x, y) / 0.1)
            score = 0.5*z + 0.3*D

        elif method == "GFI":
            roll, pitch = map(
                float, pose["euler_rpy"][:2]
            )
            D = np.exp(-np.hypot(x, y) / 0.1)
            U = np.exp(-np.hypot(roll, pitch))
            score = 0.5*z + 0.3*D + 0.2*U

        else:
            raise ValueError(method)

        scores.append(score)

    return np.argsort(
        -np.asarray(scores),
        kind="stable"
    )[:10]

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    rows = []

    print("\nSTEP 47Q — REPRODUCIBLE STANDALONE RUNTIME")
    print("=" * 75)

    for name, folder in OBJECTS.items():
        archive_path = (
            PROJECT / "paper_runs" /
            folder / "data_export.pkl"
        )

        with archive_path.open("rb") as f:
            archive = pickle.load(f)

        poses = [
            p for p in archive["final_6dof_pos"]
            if float(p["xyz"][2]) > 0
        ]

        print(f"\n{name}: {len(poses)} placements")

        for method in METHODS:

            for _ in range(30):
                rank(poses, method)

            timings = []

            for _ in range(200):
                start = perf_counter_ns()
                rank(poses, method)
                timings.append(
                    (perf_counter_ns() - start) / 1e6
                )

            median = float(np.median(timings))

            rows.append({
                "object": name,
                "candidate_count": len(poses),
                "method": method,
                "repetitions": 200,
                "warmups": 30,
                "median_runtime_ms": median,
                "mean_runtime_ms": float(
                    np.mean(timings)
                ),
                "std_runtime_ms": float(
                    np.std(timings, ddof=1)
                )
            })

            print(f"  {method:20s}{median:.4f} ms")

    output_file = OUTPUT / "step47q_standalone_runtime.csv"

    if output_file.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output_file}"
        )

    with output_file.open(
        "x", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", output_file)
    print("Original experimental records unchanged.")
    print("STEP 47U COMPLETED")

if __name__ == "__main__":
    main()
