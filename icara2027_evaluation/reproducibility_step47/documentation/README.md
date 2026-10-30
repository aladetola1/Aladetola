# ICARA 2027 — Step 47 Reproducibility Record

## Purpose

Evaluate computational overhead and retrospective grasp-selection
performance of the RGMP Grasp-Friendliness Index (GFI) relative to
simpler geometric ranking baselines.

These experiments address practical computational efficiency, but
do not establish a novel optimization algorithm or superiority of GFI.

## Original data

- Six YCB object placement archives: 2,091 placements.
- Five objects independently evaluated using Panda/PyBullet.
- 154 independently evaluated placements, 101 successful.
- Original frozen protocols and independent CSV results remain
  in the parent evaluation directory.

## Reproducible experiments

### Step 47O — Matched-budget grasp selection

Script:
scripts/step47o_matched_budget.py

Output:
results/step47o_matched_budget.csv

Reconstructs GFI and four geometric baseline rankings over the same
independently evaluated candidate pool for each object.

Budgets: K = 1, 3, 5, 10 evaluations per object.

At K=10:
- GFI: 30/50 successful selections
- Centrality: 35/50
- Height: 29/50
- Uprightness: 37/50
- Height+Centrality: 35/50

This is a retrospective analysis of a selected candidate population.
The 154 evaluated placements were chosen through competing ranking
strategies, not sampled randomly from the complete archives.

Results cannot be generalized to unevaluated archive placements.

### Step 47Q — Standalone ranking computation

Script:
scripts/step47q_standalone_runtime.py

Output:
results/step47q_standalone_runtime.csv

Measures score computation and stable top-10 sorting for each method.

Each method computes only its required geometric descriptors.

Settings:
- 30 warmup repetitions.
- 200 measured repetitions.
- Python time.perf_counter_ns().
- Archive loading excluded.
- Python/NumPy implementations, not optimized native kernels.

The numerical runtime results vary with hardware and system load.

### Panda/PyBullet pilot timing

Development-placement-only timing was performed interactively.

Bowl, development index 130:
- 10 repeated evaluations.
- Median: 1.7780 seconds.
- Mean: 1.8373 seconds.
- Standard deviation: 0.1431 seconds.
- Outcome: GRASP_FAILED in all repetitions.

Additional preliminary development benchmarks:
- Spoon, index 2: median 1.0549 s, 2 runs.
- Banana, index 572: median 1.0654 s, 2 runs.
- Power Drill, index 3: median 0.9298 s, 2 runs.
- Mug, index 19: median 1.0003 s, 2 runs.

These Panda times include per-placement simulation setup,
grasp evaluation, and cleanup, but exclude module import and archive
loading. They are not population-level runtime estimates.

Pilot Panda timing scripts and raw timing samples have not yet been
archived as standalone executable benchmark files.

## Interpretation

Geometric ranking is computationally inexpensive.

However, GFI does not outperform simpler geometric baselines in
the retrospective Panda matched-budget analysis.

In particular, uprightness-only has higher successful-selection
counts at every tested budget.

GFI should not be described as offering a superior computational
efficiency versus grasp-success trade-off relative to these baselines.

Comparisons between geometric ranking time and full Panda simulation
time must not be presented as direct algorithmic speedup factors.

## Figures

figures/icara2027_grasp_success_vs_budget.pdf
figures/icara2027_grasp_success_vs_budget.png
figures/icara2027_grasp_success_vs_budget.svg

## Reproduction

Run from the original project environment:

python reproducibility_step47/scripts/step47o_matched_budget.py
python reproducibility_step47/scripts/step47q_standalone_runtime.py

Both scripts refuse to overwrite their existing result CSV files.
To reproduce into fresh CSV files, archive or rename the existing
outputs first.

Do not modify frozen protocols or original independent results.
