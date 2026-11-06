# ICARA 2027 — Top-ten placement traceability

## Verified GFI reference placements

all_six_objects_gfi_top10_verified.csv:
60 GFI-ranked placements, ten per object, including
archive indices, final simulated 6-DoF poses, and
geometric descriptor scores.

All poses were verified against the original PLACE IT
paper_runs/<object>/data_export.pkl final_6dof_pos
records using object name and zero-based array index.

## Physical TIAGo placement summaries

tiago_gfi_top10_poses_and_outcomes.csv:
30 archive-matched GFI reference poses and reconstructed
placement-level physical success counts. Success totals
are 279/300.

tiago_random_original_success_rates.csv:
30 available original sample-level success percentages.
The random archive-index mapping and full 6-DoF poses
could not be independently verified. Therefore this
release does not present reconstructed random indices
as verified physical placement identities.

Original per-attempt experimental logs are unavailable
in the inspected records. Archive reference poses are
not measured physical execution poses.

GFI is a post-optimization geometric ranking heuristic.
The weights are (0.5, 0.3, 0.2), with height scale 1 m,
centrality scale 0.1 m, and angular scale 1 rad.

This supplement does not replace the V2 release.
