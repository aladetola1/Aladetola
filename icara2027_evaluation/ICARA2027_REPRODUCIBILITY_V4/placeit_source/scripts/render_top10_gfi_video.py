
from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


@dataclass
class PlacementCandidate:
    """One archived placement candidate."""

    original_index: int
    individual: np.ndarray
    final_pose: np.ndarray
    gfi: float
    fitness: float | None
    is_valid: bool
    is_success: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and render the Top-K PlaceIt placements ranked by GFI."
    )

    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Path to the NPZ archive containing individuals and GFI information.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of highest-GFI placements to retain. Default: 10.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("top10_gfi_video"),
        help="Directory for exported placements, frames, CSV, and MP4.",
    )

    parser.add_argument(
        "--object",
        type=str,
        default="object",
        help="Object name displayed in the video, for example: mug.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate. Default: 30.",
    )

    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.5,
        help="Time to hold the final stable pose. Default: 1.5 seconds.",
    )

    parser.add_argument(
        "--renderer-module",
        type=str,
        default=None,
        help=(
            "Python module containing a replay function. Example: "
            "scripts.render_selected_poses"
        ),
    )

    parser.add_argument(
        "--renderer-function",
        type=str,
        default="render_placement_frames",
        help=(
            "Function inside renderer-module. It must accept "
            "(individual, output_dir, rank, gfi) and return frame paths."
        ),
    )

    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Only export the Top-K candidates; do not render a video.",
    )

    return parser.parse_args()


def first_existing_key(
    archive: np.lib.npyio.NpzFile,
    possible_keys: list[str],
) -> str | None:
    """Return the first archive key that exists."""

    for key in possible_keys:
        if key in archive.files:
            return key

    return None


def normalize_boolean_array(
    values: np.ndarray | None,
    length: int,
    default: bool,
) -> np.ndarray:
    """Convert optional archive values into a Boolean array."""

    if values is None:
        return np.full(length, default, dtype=bool)

    values = np.asarray(values).reshape(-1)

    if len(values) != length:
        raise ValueError(
            f"Boolean field contains {len(values)} entries, "
            f"but {length} individuals were found."
        )

    return values.astype(bool)


def normalize_numeric_array(
    values: np.ndarray | None,
    length: int,
    default: float = np.nan,
) -> np.ndarray:
    """Convert optional archive values into a numeric array."""

    if values is None:
        return np.full(length, default, dtype=float)

    values = np.asarray(values).reshape(-1)

    if len(values) != length:
        raise ValueError(
            f"Numeric field contains {len(values)} entries, "
            f"but {length} individuals were found."
        )

    return values.astype(float)


def load_candidates(archive_path: Path) -> list[PlacementCandidate]:
    """
    Load archive candidates using commonly encountered PlaceIt key names.

    Supported key alternatives include:
        individuals / inds / solutions / genomes
        grasp_proxy / gfi / grasp_fitness
        final_6dof_pose / final_pose / final_poses
        fitness / fitnesses / objective
        is_valid / valid
        is_success / success / is_stable
    """

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    archive = np.load(archive_path, allow_pickle=True)

    print("\nAvailable archive keys:")
    for key in archive.files:
        value = np.asarray(archive[key])
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    individual_key = first_existing_key(
        archive,
        [
            "individuals",
            "inds",
            "solutions",
            "genomes",
            "individual",
        ],
    )

    if individual_key is None:
        raise KeyError(
            "Could not find individuals. Expected one of: "
            "individuals, inds, solutions, genomes."
        )

    individuals = np.asarray(archive[individual_key])

    if individuals.ndim == 1:
        individuals = np.asarray(list(individuals))

    number_of_candidates = len(individuals)

    gfi_key = first_existing_key(
        archive,
        [
            "grasp_proxy",
            "gfi",
            "gfi_scores",
            "grasp_fitness",
            "grasp_accessibility",
        ],
    )

    if gfi_key is None:
        raise KeyError(
            "Could not find GFI values. Expected one of: "
            "grasp_proxy, gfi, gfi_scores, grasp_fitness, "
            "grasp_accessibility."
        )

    final_pose_key = first_existing_key(
        archive,
        [
            "final_6dof_pose",
            "final_pose",
            "final_poses",
            "obj_6dof_final_pose",
        ],
    )

    fitness_key = first_existing_key(
        archive,
        [
            "fitness",
            "fitnesses",
            "objective",
            "objectives",
        ],
    )

    valid_key = first_existing_key(
        archive,
        [
            "is_valid",
            "valid",
            "validity",
        ],
    )

    success_key = first_existing_key(
        archive,
        [
            "is_success",
            "success",
            "is_stable",
            "stable",
        ],
    )

    gfi_values = normalize_numeric_array(
        np.asarray(archive[gfi_key]),
        number_of_candidates,
    )

    fitness_values = normalize_numeric_array(
        np.asarray(archive[fitness_key]) if fitness_key else None,
        number_of_candidates,
    )

    valid_values = normalize_boolean_array(
        np.asarray(archive[valid_key]) if valid_key else None,
        number_of_candidates,
        default=True,
    )

    success_values = normalize_boolean_array(
        np.asarray(archive[success_key]) if success_key else None,
        number_of_candidates,
        default=True,
    )

    if final_pose_key:
        final_poses = np.asarray(archive[final_pose_key])

        if len(final_poses) != number_of_candidates:
            raise ValueError(
                f"Final pose field contains {len(final_poses)} entries, "
                f"but {number_of_candidates} individuals were found."
            )
    else:
        final_poses = np.full(
            (number_of_candidates, 6),
            np.nan,
            dtype=float,
        )

    candidates: list[PlacementCandidate] = []

    for index in range(number_of_candidates):
        fitness = float(fitness_values[index])

        if not math.isfinite(fitness):
            fitness_value: float | None = None
        else:
            fitness_value = fitness

        candidates.append(
            PlacementCandidate(
                original_index=index,
                individual=np.asarray(individuals[index], dtype=float),
                final_pose=np.asarray(final_poses[index], dtype=float),
                gfi=float(gfi_values[index]),
                fitness=fitness_value,
                is_valid=bool(valid_values[index]),
                is_success=bool(success_values[index]),
            )
        )

    return candidates


def select_top_k(
    candidates: list[PlacementCandidate],
    top_k: int,
) -> list[PlacementCandidate]:
    """Filter invalid candidates and return the highest-GFI candidates."""

    if top_k <= 0:
        raise ValueError("--top-k must be greater than zero.")

    eligible = [
        candidate
        for candidate in candidates
        if candidate.is_valid
        and candidate.is_success
        and math.isfinite(candidate.gfi)
    ]

    if not eligible:
        raise RuntimeError(
            "No valid, stable placements with finite GFI values were found."
        )

    eligible.sort(key=lambda candidate: candidate.gfi, reverse=True)

    selected = eligible[: min(top_k, len(eligible))]

    print(
        f"\nSelected {len(selected)} of {len(eligible)} eligible placements."
    )

    for rank, candidate in enumerate(selected, start=1):
        fitness_text = (
            f"{candidate.fitness:.6f}"
            if candidate.fitness is not None
            else "N/A"
        )

        print(
            f"  Rank {rank:02d}: "
            f"index={candidate.original_index}, "
            f"GFI={candidate.gfi:.6f}, "
            f"fitness={fitness_text}"
        )

    return selected


def export_selected_candidates(
    selected: list[PlacementCandidate],
    output_dir: Path,
    object_name: str,
) -> tuple[Path, Path]:
    """Export selected placements to NPZ and CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"{object_name}_top{len(selected)}_gfi.npz"
    csv_path = output_dir / f"{object_name}_top{len(selected)}_gfi.csv"

    np.savez_compressed(
        npz_path,
        individuals=np.asarray(
            [candidate.individual for candidate in selected]
        ),
        final_6dof_pose=np.asarray(
            [candidate.final_pose for candidate in selected]
        ),
        grasp_proxy=np.asarray(
            [candidate.gfi for candidate in selected],
            dtype=float,
        ),
        fitness=np.asarray(
            [
                np.nan
                if candidate.fitness is None
                else candidate.fitness
                for candidate in selected
            ],
            dtype=float,
        ),
        original_index=np.asarray(
            [candidate.original_index for candidate in selected],
            dtype=int,
        ),
        is_valid=np.ones(len(selected), dtype=bool),
        is_success=np.ones(len(selected), dtype=bool),
    )

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "rank",
                "original_index",
                "gfi",
                "fitness",
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
            ]
        )

        for rank, candidate in enumerate(selected, start=1):
            pose = np.asarray(candidate.final_pose).reshape(-1)

            pose_values = [
                float(pose[index]) if index < len(pose) else np.nan
                for index in range(6)
            ]

            writer.writerow(
                [
                    rank,
                    candidate.original_index,
                    candidate.gfi,
                    (
                        ""
                        if candidate.fitness is None
                        else candidate.fitness
                    ),
                    *pose_values,
                ]
            )

    print(f"\nTop-K NPZ saved to: {npz_path}")
    print(f"Top-K CSV saved to: {csv_path}")

    return npz_path, csv_path


def load_renderer(
    module_name: str,
    function_name: str,
) -> Callable[..., list[str] | list[Path]]:
    """Import the project-specific rendering function."""

    module = importlib.import_module(module_name)

    if not hasattr(module, function_name):
        raise AttributeError(
            f"Module '{module_name}' does not contain "
            f"function '{function_name}'."
        )

    renderer = getattr(module, function_name)

    if not callable(renderer):
        raise TypeError(
            f"'{module_name}.{function_name}' is not callable."
        )

    return renderer


def add_video_overlay(
    frame: np.ndarray,
    rank: int,
    total: int,
    object_name: str,
    gfi: float,
    fitness: float | None,
) -> np.ndarray:
    """Add rank, object name, GFI, and stability information to a frame."""

    output = frame.copy()

    overlay = output.copy()
    cv2.rectangle(
        overlay,
        (20, 20),
        (570, 155),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.addWeighted(
        overlay,
        0.62,
        output,
        0.38,
        0,
        output,
    )

    fitness_text = (
        "N/A"
        if fitness is None
        else f"{fitness:.4f}"
    )

    lines = [
        f"Top Grasp-Friendly Placement {rank}/{total}",
        f"Object: {object_name}",
        f"GFI: {gfi:.4f}",
        f"Stability fitness: {fitness_text}",
    ]

    for line_index, text in enumerate(lines):
        cv2.putText(
            output,
            text,
            (40, 52 + line_index * 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def create_video(
    selected: list[PlacementCandidate],
    renderer: Callable[..., list[str] | list[Path]],
    output_dir: Path,
    object_name: str,
    fps: int,
    hold_seconds: float,
) -> Path:
    """
    Render each placement and combine the resulting frames into an MP4.

    Required renderer signature
    ---------------------------
    render_placement_frames(
        individual=<numpy array>,
        output_dir=<Path>,
        rank=<int>,
        gfi=<float>,
    ) -> list[path]
    """

    if fps <= 0:
        raise ValueError("--fps must be greater than zero.")

    frames_root = output_dir / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / (
        f"{object_name}_top{len(selected)}_gfi_placements.mp4"
    )

    writer: cv2.VideoWriter | None = None
    expected_frame_size: tuple[int, int] | None = None

    try:
        for rank, candidate in enumerate(selected, start=1):
            placement_frame_dir = frames_root / f"rank_{rank:02d}"
            placement_frame_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"\nRendering rank {rank}/{len(selected)} "
                f"(GFI={candidate.gfi:.6f})..."
            )

            frame_paths = renderer(
                individual=candidate.individual.copy(),
                output_dir=placement_frame_dir,
                rank=rank,
                gfi=candidate.gfi,
            )

            frame_paths = [Path(path) for path in frame_paths]

            if not frame_paths:
                raise RuntimeError(
                    f"Renderer returned no frames for rank {rank}."
                )

            rendered_frames: list[np.ndarray] = []

            for frame_path in frame_paths:
                frame = cv2.imread(str(frame_path))

                if frame is None:
                    raise RuntimeError(
                        f"Could not read rendered frame: {frame_path}"
                    )

                frame = add_video_overlay(
                    frame=frame,
                    rank=rank,
                    total=len(selected),
                    object_name=object_name,
                    gfi=candidate.gfi,
                    fitness=candidate.fitness,
                )

                height, width = frame.shape[:2]

                if writer is None:
                    expected_frame_size = (width, height)

                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        expected_frame_size,
                    )

                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not create video: {video_path}"
                        )

                if expected_frame_size != (width, height):
                    frame = cv2.resize(frame, expected_frame_size)

                rendered_frames.append(frame)
                writer.write(frame)

            hold_frame_count = max(0, int(round(hold_seconds * fps)))

            if rendered_frames and hold_frame_count:
                final_frame = rendered_frames[-1]

                for _ in range(hold_frame_count):
                    writer.write(final_frame)

    finally:
        if writer is not None:
            writer.release()

    if not video_path.exists():
        raise RuntimeError("Video generation did not produce an output file.")

    print(f"\nVideo saved to: {video_path}")

    return video_path


def main() -> int:
    args = parse_args()

    try:
        candidates = load_candidates(args.archive)

        selected = select_top_k(
            candidates=candidates,
            top_k=args.top_k,
        )

        export_selected_candidates(
            selected=selected,
            output_dir=args.output_dir,
            object_name=args.object,
        )

        if args.selection_only:
            print(
                "\nSelection completed. Video rendering was skipped "
                "because --selection-only was provided."
            )
            return 0

        if not args.renderer_module:
            print(
                "\nTop-K selection was completed successfully.\n"
                "Video rendering was not started because "
                "--renderer-module was not provided.\n\n"
                "After adding render_placement_frames() to your existing "
                "renderer, use:\n\n"
                "python scripts/render_top10_gfi_video.py "
                "--archive YOUR_ARCHIVE.npz "
                "--object mug "
                "--renderer-module scripts.render_selected_poses"
            )

            return 0

        renderer = load_renderer(
            module_name=args.renderer_module,
            function_name=args.renderer_function,
        )

        create_video(
            selected=selected,
            renderer=renderer,
            output_dir=args.output_dir,
            object_name=args.object,
            fps=args.fps,
            hold_seconds=args.hold_seconds,
        )

        return 0

    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
