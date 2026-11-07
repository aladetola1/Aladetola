# ICARA 2027 — RGMP Reproducibility V3

## Research contribution

Robot-model-independent Geometric Manipulation Proxy (RGMP)
extends PLACE IT by post-ranking stable archived placements.

The paper investigates when geometric descriptors help,
when they fail, and how robot-specific feasibility can
refine placement selection.

RGMP preserves the original CMA-MAE optimizer,
fitness function, and quality-diversity archive.

## Geometric scores

GFI = 0.5 H + 0.3 D + 0.2 U

- H = z / (1 m)
- D = exp(-sqrt(x² + y²) / (0.1 m))
- U = exp(-sqrt(roll² + pitch²) / (1 rad))

Angles are in radians. GFI is a post-optimization
geometric ranking heuristic, not a grasp predictor.

## Panda experiments

The source dataset contains 154 evaluated placements
across five objects. The main retrospective comparison
uses four objects: Banana, Mug, Power Drill, and Spoon.

At selection budget K=10:

| Selection policy | Success |
|---|---:|
| Original GFI | 30/40 (75%) |
| Centrality-only | 35/40 (87.5%) |
| Height-only | 29/40 (72.5%) |
| Uprightness-only | 35/40 (87.5%) |
| Height + Centrality | 35/40 (87.5%) |
| Centrality + Uprightness (U >= 0.99) | 30/40 (75%) |
| GFI + retrospective admissibility | 37/40 (92.5%) |

Equal retrospective static screening applied to
centrality-only yielded 38/40 (95%).

Admissibility was reconstructed from recorded
evaluation outcomes and was not an independently
validated prospective feasibility predictor.

The selection comparisons use different eligibility
conditions and do not establish GFI superiority
under matched robot-specific screening.

## Physical TIAGo experiments

Original placement-ranking study:

- GFI-ranked: 279/300 successful grasp attempts (93%).
- Random stable placements: 126/300 (42%).
- Objects: Mug, Bowl, Spoon.
- Ten placements per object and policy.
- Ten physical attempts per placement.

Object-specific GFI / random success:

- Mug: 90/100 versus 36/100
- Bowl: 96/100 versus 47/100
- Spoon: 93/100 versus 43/100

Additional physical Mug experiment:

- Central tilted, archive index 125: 8/10 (80%).
- Central upright, archive index 901: 7/10 (70%).
- Original GFI Mug top-ten study: 90/100 (90%).

The additional conditions are individual physical
placements, not complete centrality-only top-ten
ranking-policy evaluations.

The original GFI result aggregates ten placements
and is not a matched single-pose comparison.

## Directory structure

- `data/panda/`: Panda outcomes and ranking results
- `data/tiago/`: physical grasp-success summaries
- `config/`: frozen Panda protocol configurations
- `scripts/`: evaluation and plotting scripts
- `figures/`: final manuscript PDF and PNG figures

## Reproducibility limitations

The Panda comparisons are retrospective analyses
of previously evaluated candidate outcomes.

Some plotting scripts depend on the original
project directory structure; this directory is
not a standalone executable software release.

Physical TIAGo per-attempt logs, independently
measured placement reconstruction errors, and
complete robot execution configurations are not
provided in this package.

The physical experiments cannot be independently
reproduced from aggregate success summaries alone.

Historical datasets and analyses remain preserved
in earlier releases. AUROC is not part of the
current manuscript's reported results.

## Manuscript

From Stable Placement to Grasp-Aware Placement:
A Robot-Independent Geometric Manipulation Proxy
for Quality-Diversity Robotic Placement.

ICARA 2027 manuscript in preparation.
