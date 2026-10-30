
# ICARA 2027 — RGMP Experimental Research Log

Date: 25 October 2026

## Original method
GFI = 0.5H + 0.3D + 0.2U

H = z
D = exp(-sqrt(x²+y²)/0.1)
U = exp(-sqrt(roll²+pitch²))

Original method remains unchanged.

## Dataset
Six YCB objects, 2,091 stable placements:
Mug 481; Bowl 217; Spoon 389;
Power drill 549; Plate 242; Banana 213.

## Reproducibility
All 60 original Top-10 GFI values reproduced
with maximum absolute error = 0.

## E1 — Component ablation
Mean Top-10 overlap with full RGMP:
H-only: 0%
D-only: 65%
U-only: 5%

## E2 — Weight sensitivity
Four alternative weight configurations.
24 object-configuration comparisons.
Top-10 membership retention: 100%.
Lowest Spearman rho: 0.9730.

## E3a — Pose-perturbation robustness
100 Monte Carlo repetitions per object and level.
Random seed: 2027.

Average Top-10 retention:
Small (2 mm, 1 degree): 9.03/10
Moderate (5 mm, 3 degrees): 7.43/10
Large (10 mm, 5 degrees): 5.62/10

These are synthetic ranking-robustness experiments,
not physical placement-stability experiments.

## Top-10 selection margins
Mug: 0.00078984
Bowl: 0.00250644
Spoon: 0.00346461
Power drill: 0.00084147
Plate: 0.00371430
Banana: 0.01118986

A single selection margin does not fully explain
the observed robustness differences.

## Next experiments
E3b: Physical robustness in PyBullet, if feasible.
E4: Cross-object consistency and consolidated analysis.

## Manuscript constraints
ICARA 2027: maximum six pages.
Preserve the original RGMP scoring methodology.
Do not claim improved grasp success from ranking-only tests.
