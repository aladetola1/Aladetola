
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


OBJECTS = {
    "Mug": Path("results/mug/run87"),
    "Bowl": Path("results/bowl/run89"),
    "Spoon": Path("results/spoon/run94"),
    "Plate": Path("results/plate/run100"),
    "Banana": Path("results/banana/run103"),
    "Power Drill": Path("results/power_drill/run97"),
}

OUTPUT_DIR = Path("results/top10_gfi_tables")
TOP_K = 10


def convert_candidate(candidate: Any) -> dict[str, Any]:
    """Convert a NumPy object entry into a standard dictionary."""

    if isinstance(candidate, dict):
        return candidate

    if hasattr(candidate, "item"):
        converted = candidate.item()
        if isinstance(converted, dict):
            return converted

    raise TypeError(
        "Each candidate must be a dictionary or a NumPy object "
        f"containing a dictionary. Received: {type(candidate)}"
    )


def extract_gfi(candidate: dict[str, Any]) -> float:
    """Extract the stored GFI using the known possible field names."""

    possible_keys = (
        "grasp_proxy",
        "gfi",
        "GFI",
        "grasp_fitness",
        "grasp_score",
    )

    for key in possible_keys:
        if key in candidate:
            return float(candidate[key])

    raise KeyError(
        "No stored GFI field was found. "
        f"Available keys: {sorted(candidate.keys())}"
    )


def extract_component(
    candidate: dict[str, Any],
    possible_keys: tuple[str, ...],
) -> float:
    """Extract a GFI component when it is stored; otherwise return NaN."""

    for key in possible_keys:
        if key in candidate:
            return float(candidate[key])

    return float("nan")


def extract_final_pose(
    candidate: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract final xyz position and roll-pitch-yaw orientation."""

    possible_pose_keys = (
        "final_pose",
        "final_6dof_pose",
        "final_pose_6dof",
    )

    pose = None

    for key in possible_pose_keys:
        if key in candidate:
            pose = candidate[key]
            break

    if pose is None:
        raise KeyError(
            "No final-pose field was found. "
            f"Available keys: {sorted(candidate.keys())}"
        )

    if isinstance(pose, dict):
        xyz_keys = ("xyz", "position", "final_position")
        rpy_keys = ("euler_rpy", "rpy", "orientation")

        xyz = None
        rpy = None

        for key in xyz_keys:
            if key in pose:
                xyz = np.asarray(pose[key], dtype=float).reshape(-1)
                break

        for key in rpy_keys:
            if key in pose:
                rpy = np.asarray(pose[key], dtype=float).reshape(-1)
                break

        if xyz is None or rpy is None:
            raise KeyError(
                "The final-pose dictionary must contain xyz and RPY data. "
                f"Available pose keys: {sorted(pose.keys())}"
            )

        if xyz.size < 3 or rpy.size < 3:
            raise ValueError(
                f"Invalid final pose: xyz={xyz}, rpy={rpy}"
            )

        return xyz[:3], rpy[:3]

    pose_array = np.asarray(pose, dtype=float).reshape(-1)

    if pose_array.size < 6:
        raise ValueError(
            "The final pose must contain at least six values: "
            "x, y, z, roll, pitch, yaw."
        )

    return pose_array[:3], pose_array[3:6]


def load_top10(
    object_name: str,
    run_directory: Path,
) -> list[dict[str, Any]]:
    """Load and validate one object's Top-10 candidate file."""

    file_path = run_directory / "top10_grasp_candidates.npy"

    if not file_path.exists():
        raise FileNotFoundError(
            f"{object_name}: candidate file not found: {file_path}"
        )

    raw_data = np.load(file_path, allow_pickle=True)

    if len(raw_data) < TOP_K:
        raise RuntimeError(
            f"{object_name}: expected at least {TOP_K} candidates, "
            f"but found {len(raw_data)}."
        )

    candidates = [
        convert_candidate(entry)
        for entry in raw_data[:TOP_K]
    ]

    gfi_values = [extract_gfi(candidate) for candidate in candidates]

    for index in range(len(gfi_values) - 1):
        if gfi_values[index] < gfi_values[index + 1]:
            print(
                f"[WARNING] {object_name}: candidates are not in "
                "strictly descending GFI order. They will be sorted."
            )

            candidates = sorted(
                candidates,
                key=extract_gfi,
                reverse=True,
            )
            break

    return candidates[:TOP_K]


def write_compact_csv(
    all_candidates: dict[str, list[dict[str, Any]]],
) -> Path:
    """Write the 10 x 6 compact GFI table for the main paper."""

    output_path = OUTPUT_DIR / "top10_gfi_all_objects.csv"

    object_names = list(all_candidates.keys())

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(["Rank", *object_names])

        for rank in range(1, TOP_K + 1):
            row = [rank]

            for object_name in object_names:
                gfi = extract_gfi(
                    all_candidates[object_name][rank - 1]
                )
                row.append(f"{gfi:.6f}")

            writer.writerow(row)

    return output_path


def write_markdown_table(
    all_candidates: dict[str, list[dict[str, Any]]],
) -> Path:
    """Write a Markdown table that can be copied into Word."""

    output_path = OUTPUT_DIR / "top10_gfi_all_objects.md"
    object_names = list(all_candidates.keys())

    lines = [
        "# Top-10 RGMP-Ranked Placements",
        "",
        "| Rank | " + " | ".join(object_names) + " |",
        "|---:|" + "|".join(["---:" for _ in object_names]) + "|",
    ]

    for rank in range(1, TOP_K + 1):
        values = []

        for object_name in object_names:
            gfi = extract_gfi(
                all_candidates[object_name][rank - 1]
            )
            values.append(f"{gfi:.4f}")

        lines.append(
            f"| {rank} | " + " | ".join(values) + " |"
        )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return output_path


def write_latex_table(
    all_candidates: dict[str, list[dict[str, Any]]],
) -> Path:
    """Write an IEEE-compatible LaTeX table."""

    output_path = OUTPUT_DIR / "top10_gfi_all_objects.tex"
    object_names = list(all_candidates.keys())

    short_names = {
        "Mug": "Mug",
        "Bowl": "Bowl",
        "Spoon": "Spoon",
        "Plate": "Plate",
        "Banana": "Banana",
        "Power Drill": "Power drill",
    }

    column_specification = "c" + ("c" * len(object_names))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Top-10 RGMP-ranked placements for the six benchmark objects.}",
        r"\label{tab:top10-gfi}",
        r"\renewcommand{\arraystretch}{1.1}",
        rf"\begin{{tabular}}{{{column_specification}}}",
        r"\hline",
        "Rank & "
        + " & ".join(short_names[name] for name in object_names)
        + r" \\",
        r"\hline",
    ]

    for rank in range(1, TOP_K + 1):
        values = []

        for object_name in object_names:
            gfi = extract_gfi(
                all_candidates[object_name][rank - 1]
            )
            values.append(f"{gfi:.4f}")

        lines.append(
            f"{rank} & " + " & ".join(values) + r" \\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return output_path


def candidate_to_row(
    object_name: str,
    rank: int,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Convert one candidate into a complete export row."""

    xyz, rpy = extract_final_pose(candidate)
    rpy_degrees = np.degrees(rpy)

    height_score = extract_component(
        candidate,
        (
            "height_score",
            "normalized_height",
            "height_component",
            "H",
        ),
    )

    centrality_score = extract_component(
        candidate,
        (
            "centrality_score",
            "workspace_centrality",
            "centrality_component",
            "C",
        ),
    )

    uprightness_score = extract_component(
        candidate,
        (
            "uprightness_score",
            "upright_score",
            "uprightness_component",
            "U",
        ),
    )

    return {
        "object": object_name,
        "rank": rank,
        "gfi": extract_gfi(candidate),
        "height_score": height_score,
        "centrality_score": centrality_score,
        "uprightness_score": uprightness_score,
        "x_m": xyz[0],
        "y_m": xyz[1],
        "z_m": xyz[2],
        "roll_rad": rpy[0],
        "pitch_rad": rpy[1],
        "yaw_rad": rpy[2],
        "roll_deg": rpy_degrees[0],
        "pitch_deg": rpy_degrees[1],
        "yaw_deg": rpy_degrees[2],
    }


def write_detailed_csv_files(
    all_candidates: dict[str, list[dict[str, Any]]],
) -> tuple[Path, list[Path]]:
    """Write one detailed CSV per object and one combined CSV."""

    fieldnames = [
        "object",
        "rank",
        "gfi",
        "height_score",
        "centrality_score",
        "uprightness_score",
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

    combined_rows: list[dict[str, Any]] = []
    individual_paths: list[Path] = []

    for object_name, candidates in all_candidates.items():
        rows = [
            candidate_to_row(object_name, rank, candidate)
            for rank, candidate in enumerate(candidates, start=1)
        ]

        combined_rows.extend(rows)

        safe_name = object_name.lower().replace(" ", "_")
        output_path = OUTPUT_DIR / f"{safe_name}_top10_detailed.csv"

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

        individual_paths.append(output_path)

    combined_path = OUTPUT_DIR / "top10_all_objects_detailed.csv"

    with combined_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(combined_rows)

    return combined_path, individual_paths


def print_compact_table(
    all_candidates: dict[str, list[dict[str, Any]]],
) -> None:
    """Print the compact GFI table directly in the terminal."""

    object_names = list(all_candidates.keys())

    column_width = 13
    header = f"{'Rank':>5}"

    for object_name in object_names:
        header += f"{object_name:>{column_width}}"

    print("\n" + "=" * len(header))
    print("TOP-10 RGMP-RANKED GFI VALUES")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for rank in range(1, TOP_K + 1):
        row = f"{rank:>5}"

        for object_name in object_names:
            gfi = extract_gfi(
                all_candidates[object_name][rank - 1]
            )
            row += f"{gfi:>{column_width}.6f}"

        print(row)

    print("=" * len(header))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_candidates: dict[str, list[dict[str, Any]]] = {}

    for object_name, run_directory in OBJECTS.items():
        candidates = load_top10(
            object_name=object_name,
            run_directory=run_directory,
        )

        all_candidates[object_name] = candidates

        print(
            f"[OK] {object_name}: loaded "
            f"{len(candidates)} ranked placements."
        )

    print_compact_table(all_candidates)

    compact_csv = write_compact_csv(all_candidates)
    markdown_file = write_markdown_table(all_candidates)
    latex_file = write_latex_table(all_candidates)

    combined_csv, individual_csv_files = write_detailed_csv_files(
        all_candidates
    )

    print("\nFiles generated:")
    print(f"  Compact CSV:    {compact_csv}")
    print(f"  Markdown table: {markdown_file}")
    print(f"  LaTeX table:    {latex_file}")
    print(f"  Detailed CSV:   {combined_csv}")

    for path in individual_csv_files:
        print(f"  Object CSV:     {path}")


if __name__ == "__main__":
    main()

