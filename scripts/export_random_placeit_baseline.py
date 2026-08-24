from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

OBJECTS = {
    "Mug": Path("results/mug/run87"),
    "Bowl": Path("results/bowl/run89"),
    "Spoon": Path("results/spoon/run94"),
}

OUTPUT_DIR = Path("results/random_placeit_baseline")

N_RANDOM = 10

# Fixed seed makes the random selection exactly reproducible.
RANDOM_SEED = 2026


# ============================================================
# HELPERS
# ============================================================

def convert_info(entry: Any) -> dict[str, Any]:
    """Convert NumPy object entry to a normal Python dictionary."""

    if isinstance(entry, dict):
        return entry

    if hasattr(entry, "item"):
        converted = entry.item()

        if isinstance(converted, dict):
            return converted

    raise TypeError(
        "Archive info entry is not a dictionary. "
        f"Received type: {type(entry)}"
    )


def extract_gfi(info: dict[str, Any]) -> float:
    """Extract GFI if available."""

    possible_keys = (
        "grasp_proxy",
        "gfi",
        "GFI",
        "grasp_fitness",
        "grasp_score",
    )

    for key in possible_keys:
        if key in info:
            return float(info[key])

    return float("nan")


def extract_final_pose(
    info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract final Cartesian position and RPY orientation.

    Returns
    -------
    xyz : [x, y, z] in metres
    rpy : [roll, pitch, yaw] in radians
    """

    possible_pose_keys = (
        "final_pose",
        "final_6dof_pose",
        "final_pose_6dof",
    )

    pose = None

    for key in possible_pose_keys:
        if key in info:
            pose = info[key]
            break

    if pose is None:
        raise KeyError(
            "No final pose field found. "
            f"Available keys: {sorted(info.keys())}"
        )

    # --------------------------------------------------------
    # Pose stored as dictionary
    # --------------------------------------------------------

    if isinstance(pose, dict):

        xyz = None
        rpy = None

        for key in ("xyz", "position", "final_position"):
            if key in pose:
                xyz = np.asarray(
                    pose[key],
                    dtype=float
                ).reshape(-1)

                break

        for key in ("euler_rpy", "rpy", "orientation"):
            if key in pose:
                rpy = np.asarray(
                    pose[key],
                    dtype=float
                ).reshape(-1)

                break

        if xyz is None or rpy is None:
            raise KeyError(
                "Final pose dictionary does not contain "
                "both XYZ and RPY information. "
                f"Pose keys: {sorted(pose.keys())}"
            )

        if xyz.size < 3 or rpy.size < 3:
            raise ValueError(
                f"Invalid final pose: xyz={xyz}, rpy={rpy}"
            )

        return xyz[:3], rpy[:3]

    # --------------------------------------------------------
    # Pose stored as six-element array
    # --------------------------------------------------------

    pose_array = np.asarray(
        pose,
        dtype=float
    ).reshape(-1)

    if pose_array.size < 6:
        raise ValueError(
            "Final pose must contain at least six values: "
            "x, y, z, roll, pitch, yaw."
        )

    return pose_array[:3], pose_array[3:6]


# ============================================================
# LOAD COMPLETE STABLE ARCHIVE
# ============================================================

def load_stable_archive(
    object_name: str,
    run_directory: Path,
) -> tuple[np.ndarray, np.ndarray]:

    archive_path = (
        run_directory
        / "success_archives"
        / "individuals_0.npz"
    )

    if not archive_path.exists():
        raise FileNotFoundError(
            f"{object_name}: archive not found:\n"
            f"{archive_path}"
        )

    archive = np.load(
        archive_path,
        allow_pickle=True
    )

    print(f"\n{object_name}")
    print("-" * 60)
    print("Archive keys:", archive.files)

    if "infos" not in archive.files:
        raise KeyError(
            f"{object_name}: archive does not contain 'infos'."
        )

    infos = archive["infos"]

    if "fitnesses" in archive.files:
        fitnesses = np.asarray(
            archive["fitnesses"],
            dtype=float
        )
    else:
        fitnesses = np.full(
            len(infos),
            np.nan
        )

    if len(infos) < N_RANDOM:
        raise RuntimeError(
            f"{object_name}: only {len(infos)} stable placements "
            f"available; cannot sample {N_RANDOM}."
        )

    print(
        f"Stable placements available: {len(infos)}"
    )

    return infos, fitnesses


# ============================================================
# RANDOM SELECTION
# ============================================================

def select_random_placements(
    object_name: str,
    run_directory: Path,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:

    infos, fitnesses = load_stable_archive(
        object_name,
        run_directory,
    )

    # Random selection WITHOUT replacement.
    selected_indices = rng.choice(
        len(infos),
        size=N_RANDOM,
        replace=False,
    )

    rows = []

    for sample_number, archive_index in enumerate(
        selected_indices,
        start=1,
    ):

        info = convert_info(
            infos[archive_index]
        )

        xyz, rpy = extract_final_pose(info)

        rpy_degrees = np.degrees(rpy)

        if archive_index < len(fitnesses):
            fitness = float(
                fitnesses[archive_index]
            )
        else:
            fitness = float("nan")

        row = {
            "object": object_name,
            "random_sample": sample_number,
            "archive_index": int(archive_index),
            "placeit_fitness": fitness,
            "gfi": extract_gfi(info),

            "x_m": float(xyz[0]),
            "y_m": float(xyz[1]),
            "z_m": float(xyz[2]),

            "roll_rad": float(rpy[0]),
            "pitch_rad": float(rpy[1]),
            "yaw_rad": float(rpy[2]),

            "roll_deg": float(rpy_degrees[0]),
            "pitch_deg": float(rpy_degrees[1]),
            "yaw_deg": float(rpy_degrees[2]),
        }

        rows.append(row)

    return rows


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:

    fieldnames = [
        "object",
        "random_sample",
        "archive_index",
        "placeit_fitness",
        "gfi",
        "x_m",
        "y_m",
        "z_m",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# PRINT HARDWARE TABLE
# ============================================================

def print_hardware_table(
    object_name: str,
    rows: list[dict[str, Any]],
) -> None:

    print("\n")
    print("=" * 112)
    print(
        f"{object_name.upper()} — "
        f"10 RANDOM PLACEIT STABLE PLACEMENTS"
    )
    print("=" * 112)

    print(
        f"{'No.':>3} "
        f"{'Index':>7} "
        f"{'x (m)':>11} "
        f"{'y (m)':>11} "
        f"{'z (m)':>11} "
        f"{'Roll (deg)':>12} "
        f"{'Pitch (deg)':>12} "
        f"{'Yaw (deg)':>12}"
    )

    print("-" * 112)

    for row in rows:

        print(
            f"{row['random_sample']:>3d} "
            f"{row['archive_index']:>7d} "
            f"{row['x_m']:>11.6f} "
            f"{row['y_m']:>11.6f} "
            f"{row['z_m']:>11.6f} "
            f"{row['roll_deg']:>12.3f} "
            f"{row['pitch_deg']:>12.3f} "
            f"{row['yaw_deg']:>12.3f}"
        )

    print("=" * 112)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # One deterministic random generator.
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    all_rows = []

    print("\n")
    print("=" * 70)
    print("RANDOM PLACEIT BASELINE")
    print("=" * 70)

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print(
        f"Placements per object: {N_RANDOM}"
    )

    for object_name, run_directory in OBJECTS.items():

        rows = select_random_placements(
            object_name,
            run_directory,
            rng,
        )

        all_rows.extend(rows)

        print_hardware_table(
            object_name,
            rows,
        )

        safe_name = (
            object_name
            .lower()
            .replace(" ", "_")
        )

        individual_path = (
            OUTPUT_DIR
            / f"{safe_name}_random10_hardware_poses.csv"
        )

        save_csv(
            rows,
            individual_path,
        )

        print(
            f"\nSaved: {individual_path}"
        )

    combined_path = (
        OUTPUT_DIR
        / "random10_all_hardware_poses.csv"
    )

    save_csv(
        all_rows,
        combined_path,
    )

    print("\n" + "=" * 70)

    print(
        "Random baseline generation complete."
    )

    print(
        f"Combined file: {combined_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
