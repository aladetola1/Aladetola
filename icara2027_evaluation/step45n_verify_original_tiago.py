#!/usr/bin/env python3
"""Verify ICARA 2027 TIAGo statistics without modifying experimental data."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def verify(package_root):
    root = Path(package_root)

    top = pd.read_csv(
        root / "original_tiago/experimental_grasp_success_data.csv"
    )
    random = pd.read_csv(
        root / "original_tiago/random_grasp_success_data.csv"
    )
    matched = pd.read_csv(
        root / "evaluation/step45h_tiago_random_original_verified.csv"
    )

    for frame in (top, random, matched):
        frame["object"] = frame["object"].str.lower().str.strip()

    expected_objects = {"mug", "bowl", "spoon"}

    for name, frame in [("Top", top), ("Random", random)]:
        assert len(frame) == 30, f"{name}: expected 30 placements"
        assert set(frame["object"]) == expected_objects

        for obj, part in frame.groupby("object"):
            assert len(part) == 10, (name, obj)
            rates = part["grasp_success_rate_percent"]
            assert rates.between(0, 100).all()
            assert np.allclose(rates / 10, np.round(rates / 10))

    assert not top.duplicated(["object", "gfi_rank"]).any()
    assert not random.duplicated(["object", "sample"]).any()

    top_means = top.groupby("object")[
        "grasp_success_rate_percent"
    ].mean()

    random_means = random.groupby("object")[
        "grasp_success_rate_percent"
    ].mean()

    expected_top = {"mug": 90, "bowl": 96, "spoon": 93}
    expected_random = {"mug": 36, "bowl": 47, "spoon": 43}

    for obj in sorted(expected_objects):
        assert np.isclose(top_means[obj], expected_top[obj])
        assert np.isclose(random_means[obj], expected_random[obj])

    assert len(matched) == 30
    assert not matched.duplicated(["object", "sample_number"]).any()

    joined = random.merge(
        matched,
        left_on=["object", "sample"],
        right_on=["object", "sample_number"],
        suffixes=("_original", "_matched"),
        validate="one_to_one"
    )

    assert len(joined) == 30
    assert np.allclose(
        joined["grasp_success_rate_percent"],
        joined["success_rate_percent"]
    )

    print("ICARA 2027 — ORIGINAL TIAGO VERIFICATION")
    print("=" * 66)

    print("\nObject-wise grasp success (%)")
    print("Object     RGMP-ranked    Stratified random")

    for obj in ["mug", "bowl", "spoon"]:
        print(
            f"{obj.title():10s} "
            f"{top_means[obj]:10.1f} "
            f"{random_means[obj]:18.1f}"
        )

    top_overall = top_means.mean()
    random_overall = random_means.mean()

    print(f"\nRGMP overall:   {top_overall:.1f}%")
    print(f"Random overall: {random_overall:.1f}%")
    print(f"Difference:     {top_overall-random_overall:.1f} pp")
    print("Placements:     60")
    print("Attempts:       600 (10 per placement)")

    assert np.isclose(top_overall, 93.0)
    assert np.isclose(random_overall, 42.0)

    print("\nOriginal-baseline score correlations")

    expected = {
        "mug": (0.4647, 0.5826),
        "bowl": (0.3350, 0.9045),
        "spoon": (0.6483, 0.8182)
    }

    for obj in ["mug", "bowl", "spoon"]:
        part = matched[matched["object"] == obj]

        rho_gfi = spearmanr(
            part["GFI"], part["success_rate_percent"]
        ).statistic

        rho_d = spearmanr(
            part["D"], part["success_rate_percent"]
        ).statistic

        assert np.isclose(rho_gfi, expected[obj][0], atol=0.0001)
        assert np.isclose(rho_d, expected[obj][1], atol=0.0001)

        print(
            f"{obj.title():10s} "
            f"GFI={rho_gfi:.4f}, "
            f"Centrality={rho_d:.4f}, "
            f"Difference={rho_gfi-rho_d:+.4f}"
        )

    print("\nALL VERIFIED NUMERICAL CHECKS PASSED")
    print("No experimental files modified.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "package_root",
        help="Directory containing original_tiago/ and evaluation/"
    )
    arguments = parser.parse_args()
    verify(arguments.package_root)
