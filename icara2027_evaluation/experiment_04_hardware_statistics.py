from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

RGMP_FILE = (
    ROOT / "results" / "experimental_validation"
    / "experimental_grasp_success_data.csv"
)

RANDOM_FILE = (
    ROOT / "results" / "experimental_validation"
    / "random_baseline"
    / "random_grasp_success_data.csv"
)

SEED = 2027
N_BOOTSTRAP = 20000

rng = np.random.default_rng(SEED)

rgmp = pd.read_csv(RGMP_FILE)
random = pd.read_csv(RANDOM_FILE)

COLUMN = "grasp_success_rate_percent"

objects = ["Mug", "Bowl", "Spoon"]

print("\nICARA 2027 — HARDWARE STATISTICAL ANALYSIS")
print("=" * 65)

# Each observation is one placement, not one grasp attempt.
# We resample placements separately within each object
# and experimental strategy.

rgmp_groups = {}
random_groups = {}

for obj in objects:
    rgmp_groups[obj] = rgmp.loc[
        rgmp["object"] == obj, COLUMN
    ].to_numpy(dtype=float)

    random_groups[obj] = random.loc[
        random["object"] == obj, COLUMN
    ].to_numpy(dtype=float)

    assert len(rgmp_groups[obj]) == 10
    assert len(random_groups[obj]) == 10

observed_rgmp = rgmp[COLUMN].mean()
observed_random = random[COLUMN].mean()
observed_difference = observed_rgmp - observed_random

bootstrap_differences = np.empty(N_BOOTSTRAP)

for b in range(N_BOOTSTRAP):
    rgmp_samples = []
    random_samples = []

    for obj in objects:
        a = rgmp_groups[obj]
        c = random_groups[obj]

        rgmp_samples.extend(
            rng.choice(a, size=len(a), replace=True)
        )

        random_samples.extend(
            rng.choice(c, size=len(c), replace=True)
        )

    bootstrap_differences[b] = (
        np.mean(rgmp_samples)
        - np.mean(random_samples)
    )

ci_lower, ci_upper = np.percentile(
    bootstrap_differences, [2.5, 97.5]
)

print(f"RGMP mean:           {observed_rgmp:.2f}%")
print(f"Random mean:         {observed_random:.2f}%")
print(f"Observed difference: {observed_difference:.2f} pp")

print(
    "95% stratified placement-bootstrap CI: "
    f"[{ci_lower:.2f}, {ci_upper:.2f}] pp"
)

print("\nObject-level differences:")
for obj in objects:
    difference = (
        rgmp_groups[obj].mean()
        - random_groups[obj].mean()
    )
    print(f"  {obj:8s}: {difference:+.2f} pp")

output = (
    Path(__file__).resolve().parent
    / "experiment_04_hardware_statistics_results.csv"
)

pd.DataFrame([{
    "rgmp_mean_percent": observed_rgmp,
    "random_mean_percent": observed_random,
    "difference_percentage_points": observed_difference,
    "ci_95_lower_pp": ci_lower,
    "ci_95_upper_pp": ci_upper,
    "bootstrap_replicates": N_BOOTSTRAP,
    "random_seed": SEED,
    "resampling_unit": "placement",
}]).to_csv(output, index=False)

print(f"\nSaved results to: {output}")
