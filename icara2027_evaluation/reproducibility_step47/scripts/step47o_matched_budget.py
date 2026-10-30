from pathlib import Path
import csv
import pickle
import numpy as np

# Find project paths independently of the working directory.
HERE = Path(__file__).resolve()
EVAL = HERE.parents[2]
PROJECT = EVAL.parent
OUTPUT = EVAL / "reproducibility_step47" / "results"

OBJECTS = {
    "Spoon": ("spoon", "step43bs", 24),
    "Banana": ("banana", "step43cf", 29),
    "Power Drill": ("power_drill", "step43cn", 35),
    "Mug": ("mug", "step43cu", 34),
    "Bowl": ("bowl", "step43ea", 32),
}

METHODS = [
    "GFI",
    "Centrality",
    "Height",
    "Uprightness",
    "Height+Centrality",
]

BUDGETS = [1, 3, 5, 10]

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    aggregate = {
        (method, k): 0
        for method in METHODS
        for k in BUDGETS
    }

    for name, (folder, prefix, expected) in OBJECTS.items():
        result_file = (
            EVAL /
            f"{prefix}_independent_{folder}_results.csv"
        )
        archive_file = (
            PROJECT / "paper_runs" /
            folder / "data_export.pkl"
        )

        with result_file.open(
            newline="", encoding="utf-8-sig"
        ) as f:
            records = list(csv.DictReader(f))

        assert len(records) == expected

        with archive_file.open("rb") as f:
            archive = pickle.load(f)

        placements = []

        for record in records:
            index = int(record["archive_index"])
            outcome = record["success"].strip().lower()
            assert outcome in ("true", "false", "1", "0")

            pose = archive["final_6dof_pos"][index]
            xyz = np.asarray(pose["xyz"], dtype=float)
            rpy = np.asarray(pose["euler_rpy"], dtype=float)

            assert np.all(np.isfinite(xyz))
            assert np.all(np.isfinite(rpy))
            assert xyz[2] > 0

            x, y, z = xyz
            roll, pitch = rpy[:2]

            H = float(z)
            D = float(np.exp(-np.hypot(x, y) / 0.1))
            U = float(np.exp(-np.hypot(roll, pitch)))

            placements.append({
                "index": index,
                "success": int(outcome in ("true", "1")),
                "GFI": 0.5*H + 0.3*D + 0.2*U,
                "Centrality": D,
                "Height": H,
                "Uprightness": U,
                "Height+Centrality": 0.5*H + 0.3*D
            })

        assert len({
            r["index"] for r in placements
        }) == expected

        placements.sort(key=lambda r: r["index"])

        for method in METHODS:
            ranked = sorted(
                placements,
                key=lambda r: -r[method]
            )

            for k in BUDGETS:
                successes = sum(
                    r["success"] for r in ranked[:k]
                )

                aggregate[(method, k)] += successes

                rows.append({
                    "object": name,
                    "method": method,
                    "budget_per_object": k,
                    "evaluations": k,
                    "successful_selections": successes,
                    "success_rate": successes / k
                })

    output_file = OUTPUT / "step47o_matched_budget.csv"

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

    print("\nSTEP 47O — REPRODUCIBLE RESULTS")
    print("=" * 66)

    for method in METHODS:
        text = " | ".join(
            f"K={k}: {aggregate[(method, k)]}/{5*k}"
            for k in BUDGETS
        )
        print(f"{method:19s} {text}")

    print("\nSaved:", output_file)
    print("Original independent experiments unchanged.")
    print("STEP 47T COMPLETED")

if __name__ == "__main__":
    main()
