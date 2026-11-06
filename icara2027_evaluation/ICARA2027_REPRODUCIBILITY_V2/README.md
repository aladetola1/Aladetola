# ICARA 2027 — Reproducibility V2

## Physical TIAGo
- 30 GFI-ranked placements: 279/300 successful attempts.
- 30 stratified-random stable placements: 126/300.
- Three objects: Mug, Bowl, Spoon.
- Ten placements per object per method; ten attempts each.
- Random baseline sampled two placements from each of
  five planar-distance quantile bands, without replacement.

## Source provenance
The GFI dataset combines reconstructed placement-level
outcomes with archive-matched geometric poses.
The random dataset agrees 30/30 with the available
original experimental success-rate CSV.
An earlier manuscript reconstruction interchanged
Bowl samples 6 and 7 (70% and 0%). This package uses
the original experimentally recorded ordering.

The original per-attempt physical success/failure logs
and measured pose-reconstruction errors have not been
independently verified and are not supplied.
Do not treat aggregate counts as individual attempt logs.

## Panda
The source CSV contains 154 previously evaluated placements
across five objects, with geometric scores, poses and outcomes.
The four-object analysis is retrospective and excludes Bowl.
GFI top-K selection uses static-feasibility screening before
ranking; the plotted geometric ranking baselines do not apply
the same screening. Equally screened centrality achieved
38/40 (95%) versus GFI's 37/40 (92.5%).
Consequently, the plotted performance comparison does not
establish GFI superiority under equal screening conditions.

The AUROC comparison evaluates GFI and centrality scores
within the same statically feasible candidate subset.

## Configuration and replication limits
Frozen Panda grasp protocols are included. Some
candidate-generation settings differ by object.
The original TIAGo random-selection seed and complete
physical grasp-planner configuration are not established
by these records. Statistical bootstrap procedures and
seeds must be checked against the supplied scripts before
claiming fully independent reproduction.

## Contents
- data/tiago: physical placement-level summaries and source
- data/panda: evaluated poses, outcomes and statistics
- config: frozen Panda protocol specifications
- scripts: source and plotting scripts
- figures: updated manuscript figures

Original experimental files remain unchanged.
