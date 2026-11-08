# ICARA 2027 — RGMP Reproducibility V4

## Manuscript

**Robot-Model-Independent Geometric Descriptor Characterization
for Grasp-Oriented Placement Prioritization**

This repository supports the ICARA 2027 investigation of
geometric descriptor behavior in robotic placement prioritization.

## Research objective

The objective is not to demonstrate that a composite geometric
score is universally superior.

Instead, this study systematically investigates the strengths,
limitations, and interactions of:

- Reference-frame height (H)
- Workspace centrality (D)
- Roll-pitch uprightness (U)
- Their combination through the Grasp Friendliness Index (GFI)

RGMP operates after PLACE IT optimization and preserves the
original quality-diversity search, stability criteria, and archive.

The geometric score is:

GFI = 0.5H + 0.3D + 0.2U

The weights are fixed heuristics, not parameters optimized
against robot grasp-success data.

## Experimental investigations

### 1. Geometric characterization

Six YCB objects are used to examine:

- Descriptor ablation
- Weight and centrality-scale sensitivity
- Pose perturbation robustness
- Settling and mechanical disturbance response

These analyses characterize placement geometry and stability,
not confirmed robot-specific graspability.

### 2. Independent Franka Panda evaluation

The original Panda dataset contains 154 unique evaluated
placements across five objects:

- Spoon: 24
- Banana: 29
- Power Drill: 35
- Mug: 34
- Bowl: 32

The original evaluation union was constructed from five
geometric ranking policies.

Subsequent comparisons include upright-filtered centrality
and retrospective static-admissibility screening.

The four-object ranking analyses exclude Bowl.

The original archive-based top-ten lists and retrospective
re-ranking of evaluated candidates are different analyses.

Some archive-based top-ten placements lack recorded outcomes.
Missing outcomes must not be treated as observed failures.

### 3. Geometric ranking findings

Centrality-only ranking permits tilted placements and can
outperform the composite GFI for some evaluated objects.

Restricting centrality to upright configurations does not
necessarily improve robot-specific grasp success.

These results characterize descriptor-dependent behavior,
not universal superiority of any ranking policy.

### 4. Retrospective robot-specific admissibility

Recorded Panda statuses are used to identify placements
with static grasp-candidate availability.

Successful static admission does not guarantee approach,
grasp execution, or retention success.

At K=10 across four objects:

Original GFI: 30/40 (75%)
GFI with retrospective admissibility: 37/40 (92.5%)

This is a retrospective selected-shortlist result,
not a prospectively validated feasibility predictor.

### 5. Physical TIAGo experiments

Physical evaluation uses Mug, Bowl, and Spoon.

GFI-ranked placements:
279/300 successful attempts (93%).

Stratified-random stable placements:
126/300 successful attempts (42%).

Observed difference:
51 percentage points.

The placement-level object-stratified bootstrap uses
20,000 repetitions and seed 2027.

The random baseline selects two placements without
replacement from each of five planar-distance quantile bands.

The original physical random-selection seed has not been
established from the retained records.

### 6. Additional Mug orientation experiments

One near-central tilted configuration:
8/10 successful grasps (80%).

One near-central upright configuration:
7/10 successful grasps (70%).

These are individual-pose investigations, not complete
ranking-policy comparisons.

They do not establish that tilted orientations are
generally more graspable.

## Reproducibility materials

The package retains the available experimental records,
analysis scripts, configuration files, and figures
from the previous reproducibility release.

The V4 package additionally collects the final manuscript
and revised Panda figures where available.

The available pose-level data include Cartesian coordinates
and roll-pitch-yaw orientations for evaluated placements.
Top-ten ranked placement records should be checked against
their original archive indices before claiming complete
coverage of every ranking policy.

## Limitations

- Heuristically selected GFI weights
- Restricted object diversity
- One physical robot-gripper platform
- Retrospective Panda admissibility analysis
- Incomplete original archive-based top-ten outcome coverage
- Unmeasured TIAGo manual pose-reconstruction errors
- Unavailable original TIAGo attempt-level logs
- No prospective validation of static-admissibility prediction

## Scientific contribution

The central contribution is the systematic characterization
of how geometric descriptors affect placement prioritization,
where geometric preferences become insufficient, and why
robot-specific feasibility assessment remains necessary.

RGMP complements PLACE IT and downstream grasp planning.
It does not replace either framework.

## Source-code provenance and manuscript version

The `placeit_source/` directory preserves the modified
PLACE IT implementation used for RGMP development.

Its original README may contain earlier manuscript titles
and development descriptions.

The authoritative ICARA 2027 manuscript is:

Robot-Model-Independent Geometric Descriptor
Characterization for Grasp-Oriented Placement
Prioritization.

The current paper characterizes geometric descriptors,
their limitations, and their interaction with downstream
robot-specific grasp evaluation.

The original PLACE IT optimization fitness is preserved.
GFI is calculated separately as an additional geometric
placement score.

The `algorithms_research/` directory contains the
experimental implementations supporting this
descriptor-characterization study.
