import numpy as np
import os
import csv

RUNS = {
    "mug": "runs/run22",
    "bowl": "runs/run23",
    "banana": "runs/run24",
    "drill": "runs/run25",
    "plate": "runs/run26",
    "spoon": "runs/run27",
}

N_PERTURB = 100
SIGMA = 0.01


def load_best(run_path):
    data_path = os.path.join(run_path, "success_archives/individuals_0.npz")
    d = np.load(data_path, allow_pickle=True)

    fitness = d["fitnesses"]
    inds = d["inds"]

    best_idx = np.argmax(fitness)

    return inds[best_idx], fitness[best_idx]


def mock_evaluate(x, base_fitness):
    """
    ⚠️ REPLACE THIS WITH REAL PYBULLET SIMULATION LATER
    """
    noise = np.random.normal(0, 0.02)
    return base_fitness - abs(noise)


results = []

for obj, path in RUNS.items():
    print(f"\n=== Processing {obj} ===")

    base, base_fit = load_best(path)

    fitness_drops = []
    successes = 0

    for i in range(N_PERTURB):
        noise = np.random.normal(0, SIGMA, size=6)
        x = base + noise

        f = mock_evaluate(x, base_fit)
        fitness_drops.append(base_fit - f)

        # proxy success definition
        if f > base_fit - 0.05:
            successes += 1

    robustness = successes / N_PERTURB

    results.append([
        obj,
        base_fit,
        robustness,
        np.mean(fitness_drops),
        np.std(fitness_drops),
    ])

    print("best fitness:", base_fit)
    print("robustness:", robustness)


# Save CSV
out_file = "robustness_report.csv"

with open(out_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "object",
        "best_fitness",
        "robustness",
        "mean_fitness_drop",
        "std_fitness_drop"
    ])
    writer.writerows(results)

print("\nSaved:", out_file)
