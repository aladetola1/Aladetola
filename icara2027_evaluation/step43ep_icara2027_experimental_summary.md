# ICARA 2027 — Independent Grasp Evaluation Summary

## Experimental scope

This report summarizes five completed object-specific independent grasp evaluations.

Plate development was concluded without a held-out grasp evaluation. No plate success rate is estimated.

The reported success rates are object-specific and should not be pooled without harmonizing the evaluation protocols.

## Independent object-level results

| Object | Successes | Trials | Rate |
|---|---:|---:|---:|
| Spoon | 22 | 24 | 91.67% |
| Banana | 21 | 29 | 72.41% |
| Power Drill | 34 | 35 | 97.14% |
| Mug | 22 | 34 | 64.71% |
| Bowl | 2 | 32 | 6.25% |

## Five-method comparison

Each method is evaluated on its own selected subset of held-out placements. These subsets may overlap.

| Object | GFI | Centrality-only | Height-only | Uprightness-only | Height+Centrality |
|---|---:|---:|---:|---:|---:|
| Spoon | 6/7 (85.71%) | 6/7 (85.71%) | 7/8 (87.50%) | 9/9 (100.00%) | 6/7 (85.71%) |
| Banana | 6/9 (66.67%) | 7/9 (77.78%) | 6/10 (60.00%) | 8/10 (80.00%) | 7/9 (77.78%) |
| Power Drill | 10/10 (100.00%) | 10/10 (100.00%) | 8/9 (88.89%) | 10/10 (100.00%) | 10/10 (100.00%) |
| Mug | 4/10 (40.00%) | 7/9 (77.78%) | 6/10 (60.00%) | 7/10 (70.00%) | 7/9 (77.78%) |
| Bowl | 0/10 (0.00%) | 0/10 (0.00%) | 0/10 (0.00%) | 2/9 (22.22%) | 0/10 (0.00%) |

## Failure distributions

### Spoon

- GRASP_FAILED: 2
- SUCCESS: 22

### Banana

- APPROACH_FAILED: 1
- GRASP_FAILED: 3
- NO_STATIC_FEASIBLE_CANDIDATE: 4
- SUCCESS: 21

### Power Drill

- APPROACH_FAILED: 1
- SUCCESS: 34

### Mug

- APPROACH_FAILED: 4
- GRASP_FAILED: 1
- NO_STATIC_FEASIBLE_CANDIDATE: 7
- SUCCESS: 22

### Bowl

- GRASP_FAILED: 17
- NO_STATIC_FEASIBLE_CANDIDATE: 13
- SUCCESS: 2

## Methodological qualifications

- All five objects used independently recorded, object-specific frozen evaluation protocols.
- Development placements were excluded from independent evaluation.
- The spoon experiment used a different contact and success-checking protocol from later evaluations.
- The banana, power-drill and mug evaluators measure relative slip during retention from the end of lifting.
- The bowl evaluator measures cumulative relative translation and rotation from the end of finger closure through lifting and retention.
- The bowl's two successful independent grasps occurred at archive indices 930 and 1103, both selected by Uprightness-only.
- The plate investigation identified limitations of the tested parallel-jaw grasp-planning configurations; its physical graspability was not determined.
- These experiments are simulated robot executions, not physical robot experiments.
- Results do not demonstrate consistent superiority of GFI over the baseline methods.

## Experimental provenance

- spoon: SHA-256 `4c17ff5c3c2c52b8b96173da5a79334491b9c692210eddbf2bd717dc8953c3cb`
- banana: SHA-256 `ace702e7e1a01a950085853d8e006db7f367b49e17b946f7d2c6d75761a8c579`
- power_drill: SHA-256 `79e6d7a6276e1a59611aecce30a07e4faf00f71f44577fa878b80a3eb644bbc0`
- mug: SHA-256 `3814e2fa30ae64d52d3525a1dd12397ae6cf8312022c3503afeb322c5026b0a0`
- bowl: SHA-256 `11fc03d899cc81747132c1519195488284fc0ef76da76f93c6401e8fcc2f4844`
- Plate development record: SHA-256 `a42450dd29784f7177a41488689be378140aeb737a579579bfcd52b169e4e149`

## Interpretation

The frozen independent evaluations demonstrate substantial differences in simulated grasp success across object geometries and candidate-selection methods. Performance is particularly strong for power drill and spoon placements but weak for the bowl under its restricted radial rim-grasp planner. GFI does not consistently outperform the four ranking baselines in these experiments. These observations should be interpreted alongside the object-specific grasp planners and differing success criteria rather than as a pooled measure of universal graspability.
