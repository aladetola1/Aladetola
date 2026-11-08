from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


OBJECTS = {
    "Mug": Path("results/mug/run87"),
    "Bowl": Path("results/bowl/run89"),
    "Spoon": Path("results/spoon/run94"),
}

OUTPUT_DIR = Path("results/stratified_placeit_baseline")

RANDOM_SEED = 2026
N_BINS = 5
SAMPLES_PER_BIN = 2


def convert_info(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        return entry

    if hasattr(entry, "item"):
        converted = entry.item()
        if isinstance(converted, dict):
            return converted

    raise TypeError(
        f"Archive info entry is not a dictionary: {type(entry)}"
    )


def extract_gfi(info: dict[str, Any]) -> float:
    for key in (
        "grasp_proxy",
        "gfi",
        "GFI",
        "grasp_fitness",
        "grasp_score",
    ):
        if key in info:
            return float(info[key])

    return float("nan")


def extract_final_pose(
    info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:

    pose = None

    for key in (
        "final_pose",
        "final_6dof_pose",
        "final_pose_6dof",
    ):
        if key in info:
            pose = info[key]
            break

    if pose is None:
        raise KeyError(
            f"No final pose found. Keys: {sorted(info.keys())}"
        )

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
                f"Pose dictionary incomplete: {sorted(pose.keys())}"
            )

        return xyz[:3], rpy[:3]

    pose_array = np.asarray(
        pose,
        dtype=float
    ).reshape(-1)

    if pose_array.size < 6:
        raise ValueError(
            "Final pose must contain x, y, z, roll, pitch, yaw."
        )

    return pose_array[:3], pose_array[3:6]


def load_archive(
    object_name: str,
    run_directory: Path,
):

    archive_path = (
        run_directory
        / "success_archives"
        / "individuals_0.npz"
    )

    archive = np.load(
        archive_path,
        allow_pickle=True
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

    print(
        f"{object_name}: "
        f"{len(infos)} stable placements"
    )

    return infos, fitnesses


def build_records(
    infos,
    fitnesses,
):

    records = []

    for archive_index, raw_info in enumerate(infos):

        info = convert_info(raw_info)

        xyz, rpy = extract_final_pose(info)

        distance = float(
            np.linalg.norm(xyz[:2])
        )

        records.append(
            {
                "archive_index": archive_index,
                "info": info,
                "xyz": xyz,
                "rpy": rpy,
                "distance": distance,
                "fitness": float(
                    fitnesses[archive_index]
                ),
            }
        )

    return records


def stratified_sample(
    object_name: str,
    records: list[dict[str, Any]],
    rng: np.random.Generator,
):

    distances = np.array(
        [r["distance"] for r in records],
        dtype=float
    )

    quantiles = np.quantile(
        distances,
        np.linspace(0.0, 1.0, N_BINS + 1)
    )

    selected = []

    print("\n" + "=" * 80)
    print(f"{object_name.upper()} DISTANCE BANDS")
    print("=" * 80)

    for bin_index in range(N_BINS):

        low = quantiles[bin_index]
        high = quantiles[bin_index + 1]

        if bin_index == N_BINS - 1:
            candidates = [
                r for r in records
                if low <= r["distance"] <= high
            ]
        else:
            candidates = [
                r for r in records
                if low <= r["distance"] < high
            ]

        if len(candidates) < SAMPLES_PER_BIN:
            raise RuntimeError(
                f"{object_name}: bin {bin_index + 1} "
                f"has only {len(candidates)} candidates."
            )

        chosen_ids = rng.choice(
            len(candidates),
            size=SAMPLES_PER_BIN,
            replace=False,
        )

        print(
            f"Bin {bin_index + 1}: "
            f"{low:.4f} m to {high:.4f} m "
            f"({len(candidates)} placements)"
        )

        for chosen_id in chosen_ids:
            record = candidates[int(chosen_id)].copy()
            record["distance_bin"] = bin_index + 1
            selected.append(record)

    return selected


def record_to_row(
    object_name: str,
    sample_number: int,
    record: dict[str, Any],
):

    xyz = record["xyz"]
    rpy = record["rpy"]
    rpy_deg = np.degrees(rpy)

    return {
        "object": object_name,
        "sample": sample_number,
        "distance_bin": record["distance_bin"],
        "archive_index": record["archive_index"],
        "distance_from_center_m": record["distance"],
        "placeit_fitness": record["fitness"],
        "gfi": extract_gfi(record["info"]),
        "x_m": xyz[0],
        "y_m": xyz[1],
        "z_m": xyz[2],
        "roll_deg": rpy_deg[0],
        "pitch_deg": rpy_deg[1],
        "yaw_deg": rpy_deg[2],
    }


def save_csv(
    rows,
    path: Path,
):

    fieldnames = [
        "object",
        "sample",
        "distance_bin",
        "archive_index",
        "distance_from_center_m",
        "placeit_fitness",
        "gfi",
        "x_m",
        "y_m",
        "z_m",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_table(
    object_name: str,
    rows,
):

    print("\n" + "=" * 150)
    print(
        f"{object_name.upper()} — "
        "STRATIFIED RANDOM PLACEIT BASELINE"
    )
    print("=" * 150)

    print(
        f"{'No.':>3} "
        f"{'Bin':>4} "
        f"{'Index':>7} "
        f"{'d (m)':>9} "
        f"{'x (m)':>10} "
        f"{'y (m)':>10} "
        f"{'z (m)':>10} "
        f"{'Roll':>10} "
        f"{'Pitch':>10} "
        f"{'Yaw':>10} "
        f"{'GFI':>9}"
    )

    print("-" * 150)

    for row in rows:

        print(
            f"{row['sample']:>3d} "
            f"{row['distance_bin']:>4d} "
            f"{row['archive_index']:>7d} "
            f"{row['distance_from_center_m']:>9.4f} "
            f"{row['x_m']:>10.4f} "
            f"{row['y_m']:>10.4f} "
            f"{row['z_m']:>10.4f} "
            f"{row['roll_deg']:>10.2f} "
            f"{row['pitch_deg']:>10.2f} "
            f"{row['yaw_deg']:>10.2f} "
            f"{row['gfi']:>9.4f}"
        )

    print("=" * 150)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    all_rows = []

    for object_name, run_directory in OBJECTS.items():

        infos, fitnesses = load_archive(
            object_name,
            run_directory,
        )

        records = build_records(
            infos,
            fitnesses,
        )

        selected = stratified_sample(
            object_name,
            records,
            rng,
        )

        rows = [
            record_to_row(
                object_name,
                i,
                record,
            )
            for i, record in enumerate(
                selected,
                start=1
            )
        ]

        all_rows.extend(rows)

        print_table(
            object_name,
            rows,
        )

        safe_name = object_name.lower().replace(
            " ",
            "_"
        )

        save_csv(
            rows,
            OUTPUT_DIR
            / f"{safe_name}_stratified10.csv"
        )

    save_csv(
        all_rows,
        OUTPUT_DIR
        / "stratified10_all_objects.csv"
    )

    print(
        "\nSaved to:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
