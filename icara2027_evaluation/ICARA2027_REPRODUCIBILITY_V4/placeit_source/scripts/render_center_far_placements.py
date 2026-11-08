from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIRECTORY = Path(__file__).resolve().parent

for path in (REPOSITORY_ROOT, SCRIPTS_DIRECTORY):
    path_string = str(path)

    if path_string not in sys.path:
        sys.path.insert(0, path_string)


from render_top10_poses_video import (  # noqa: E402
    OBJECTS,
    create_scene,
    render_scene,
)


# ---------------------------------------------------------------------
# Object and figure configuration
# ---------------------------------------------------------------------
OBJECT_ORDER = [
    "mug",
    "bowl",
]

OUTPUT_ORDER = [
    "far_a",
    "centre",
    "far_b",
]

COLUMN_TITLES = {
    "far_a": "Peripheral Placement A",
    "centre": "Central Placement",
    "far_b": "Peripheral Placement B",
}

OBJECT_DISPLAY_NAMES = {    "mug": "Mug",
    "bowl": "Bowl",
    "banana": "Banana",
    "plate": "Plate",
    "power_drill": "Power\nDrill",
    "spoon": "Spoon",
}


# ---------------------------------------------------------------------
# Manuscript presentation settings
# ---------------------------------------------------------------------

# High-resolution rendering before cropping.
DEFAULT_RENDER_WIDTH = 2200
DEFAULT_RENDER_HEIGHT = 1600

# Final size of each panel in the combined figure.
CELL_WIDTH = 700
CELL_HEIGHT = 500

# A value above 1.0 crops the outer background and makes the object
# and workspace platform appear larger.
#
# This does not alter the simulated object scale, selected pose,
# platform dimensions, or physical scene.
CELL_ZOOM_FACTOR = 1.75

# Larger labels for readability after the full figure is reduced in
# the two-column manuscript.
COLUMN_TITLE_FONT_SIZE = 48
OBJECT_LABEL_FONT_SIZE = 52

LEFT_MARGIN = 310
TOP_MARGIN = 145
BOTTOM_MARGIN = 45
COLUMN_GAP = 12
ROW_GAP = 12


def parse_arguments() -> argparse.Namespace:
    """Read optional command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Select and render one central placement and two spatially "
            "distinct peripheral placements for each object."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/center_far_placements"),
        help=(
            "Output directory. "
            "Default: results/center_far_placements"
        ),
    )

    parser.add_argument(
        "--angular-separation-deg",
        type=float,
        default=90.0,
        help=(
            "Preferred minimum angular separation between the two "
            "peripheral placements. Default: 90 degrees."
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_RENDER_WIDTH,
        help=(
            f"Width of each original rendered frame. "
            f"Default: {DEFAULT_RENDER_WIDTH}."
        ),
    )

    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_RENDER_HEIGHT,
        help=(
            f"Height of each original rendered frame. "
            f"Default: {DEFAULT_RENDER_HEIGHT}."
        ),
    )

    parser.add_argument(
        "--zoom-factor",
        type=float,
        default=CELL_ZOOM_FACTOR,
        help=(
            "Presentation-only crop factor used to make the objects "
            f"larger in the combined figure. Default: {CELL_ZOOM_FACTOR}."
        ),
    )

    return parser.parse_args()


def numbered_archive_key(path: Path) -> int:
    """Extract the integer suffix from individuals_<number>.npz."""

    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


def find_latest_success_archive(run_directory: Path) -> Path:
    """Locate the most recent success archive for one run."""

    archive_directory = run_directory / "success_archives"

    if not archive_directory.exists():
        raise FileNotFoundError(
            f"Success archive directory not found: {archive_directory}"
        )

    archive_files = list(
        archive_directory.glob("individuals_*.npz")
    )

    if not archive_files:
        raise FileNotFoundError(
            f"No individuals_*.npz file found in {archive_directory}"
        )

    return max(
        archive_files,
        key=numbered_archive_key,
    )


def as_candidate_dictionary(
    raw_info: Any,
) -> dict[str, Any]:
    """Convert one stored archive-information entry to a dictionary."""

    if isinstance(raw_info, dict):
        return raw_info

    if hasattr(raw_info, "item"):
        converted = raw_info.item()

        if isinstance(converted, dict):
            return converted

    raise TypeError(
        "Archive information entry could not be converted "
        "to a dictionary."
    )


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the interval [-pi, pi]."""

    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


def angular_distance(
    angle_a: float,
    angle_b: float,
) -> float:
    """Return the smallest absolute angular difference in radians."""

    return abs(
        wrap_angle(angle_a - angle_b)
    )


def extract_pose(
    info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the final xyz and Euler roll-pitch-yaw pose."""

    if "final_6dof_pose" not in info:
        raise KeyError(
            "Candidate does not contain 'final_6dof_pose'. "
            f"Available keys: {sorted(info.keys())}"
        )

    pose = info["final_6dof_pose"]

    xyz = np.asarray(
        pose["xyz"],
        dtype=float,
    )

    rpy = np.asarray(
        pose["euler_rpy"],
        dtype=float,
    )

    if xyz.shape != (3,):
        raise ValueError(
            f"Expected three xyz values, received shape {xyz.shape}."
        )

    if rpy.shape != (3,):
        raise ValueError(
            f"Expected three Euler values, received shape {rpy.shape}."
        )

    return xyz, rpy


def candidate_is_usable(
    info: dict[str, Any],
) -> bool:
    """
    Keep only valid, successful, and finite archive candidates.
    """

    is_valid = float(
        info.get("is_valid", 1.0)
    )

    is_success = float(
        info.get("is_success", 1.0)
    )

    if is_valid < 0.5 or is_success < 0.5:
        return False

    try:
        xyz, rpy = extract_pose(info)
    except (KeyError, TypeError, ValueError):
        return False

    return bool(
        np.all(np.isfinite(xyz))
        and np.all(np.isfinite(rpy))
    )


def build_candidate_records(
    infos: np.ndarray,
    fitnesses: np.ndarray,
) -> list[dict[str, Any]]:
    """Build normalized records from the success archive."""

    records: list[dict[str, Any]] = []

    for index, raw_info in enumerate(infos):
        info = as_candidate_dictionary(raw_info)

        if not candidate_is_usable(info):
            continue

        xyz, rpy = extract_pose(info)

        center_distance = float(
            np.linalg.norm(xyz[:2])
        )

        planar_angle = float(
            math.atan2(
                xyz[1],
                xyz[0],
            )
        )

        grasp_proxy = float(
            info.get(
                "grasp_proxy",
                float("nan"),
            )
        )

        records.append(
            {
                "archive_index": int(index),
                "fitness": float(fitnesses[index]),
                "grasp_proxy": grasp_proxy,
                "final_pose": info["final_6dof_pose"],
                "xyz": xyz,
                "rpy": rpy,
                "center_distance": center_distance,
                "planar_angle": planar_angle,
            }
        )

    if len(records) < 3:
        raise RuntimeError(
            "Fewer than three usable stable placements were found."
        )

    return records


def select_centre_and_peripheral(
    records: list[dict[str, Any]],
    minimum_angular_separation: float,
) -> dict[str, dict[str, Any]]:
    """
    Select:
      - centre: minimum planar distance;
      - far_a: maximum planar distance;
      - far_b: distant placement in a different workspace direction.

    If no candidate satisfies the preferred angular separation, Far B
    is selected using a combined distance-and-angle score.
    """

    sorted_by_distance = sorted(
        records,
        key=lambda record: record["center_distance"],
    )

    centre = sorted_by_distance[0]
    far_a = sorted_by_distance[-1]

    remaining = [
        record
        for record in records
        if record["archive_index"] != far_a["archive_index"]
        and record["archive_index"] != centre["archive_index"]
    ]

    if not remaining:
        raise RuntimeError(
            "No remaining candidates are available for Peripheral B."
        )

    angularly_distinct = [
        record
        for record in remaining
        if angular_distance(
            record["planar_angle"],
            far_a["planar_angle"],
        )
        >= minimum_angular_separation
    ]

    if angularly_distinct:
        far_b = max(
            angularly_distinct,
            key=lambda record: record["center_distance"],
        )
    else:
        maximum_distance = max(
            record["center_distance"]
            for record in remaining
        )

        def fallback_score(
            record: dict[str, Any],
        ) -> float:
            normalized_distance = (
                record["center_distance"] / maximum_distance
                if maximum_distance > 0.0
                else 0.0
            )

            normalized_angle = (
                angular_distance(
                    record["planar_angle"],
                    far_a["planar_angle"],
                )
                / math.pi
            )

            return (
                0.7 * normalized_distance
                + 0.3 * normalized_angle
            )

        far_b = max(
            remaining,
            key=fallback_score,
        )

    selected = {
        "far_a": far_a,
        "centre": centre,
        "far_b": far_b,
    }

    for selection_type, record in selected.items():
        record["selection_type"] = selection_type

    return selected


def ensure_pil_image(
    rendered: Any,
) -> Image.Image:
    """Convert one rendering result to a three-channel PIL image."""

    if isinstance(rendered, Image.Image):
        return rendered.convert("RGB")

    array = np.asarray(rendered)

    if array.ndim != 3:
        raise TypeError(
            "The rendering output could not be converted into an image."
        )

    if array.shape[2] == 4:
        array = array[:, :, :3]

    return Image.fromarray(
        array.astype(np.uint8)
    ).convert("RGB")


def load_font(
    size: int,
    bold: bool = False,
) -> ImageFont.ImageFont:
    """Load an available TrueType font."""

    if bold:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]

    for candidate in candidates:
        candidate_path = Path(candidate)

        if candidate_path.exists():
            return ImageFont.truetype(
                str(candidate_path),
                size=size,
            )

    return ImageFont.load_default()


def render_selected_pose(
    object_name: str,
    object_urdf: str,
    record: dict[str, Any],
    width: int,
    height: int,
) -> Image.Image:
    """Render one selected placement at high resolution."""

    create_scene(
        object_name=object_name,
        object_urdf=object_urdf,
        xyz=record["xyz"],
        rpy=record["rpy"],
    )

    rendered = render_scene(
        width=width,
        height=height,
    )

    return ensure_pil_image(rendered)


def centre_crop_for_zoom(
    image: Image.Image,
    zoom_factor: float,
) -> Image.Image:
    """
    Crop around the centre to enlarge the object and workspace region.

    Only presentation framing is modified. The selected pose, object
    mesh, physical dimensions, and simulated scene remain unchanged.
    """

    if zoom_factor < 1.0:
        raise ValueError(
            "zoom_factor must be greater than or equal to 1.0."
        )

    result = image.convert("RGB").copy()

    if math.isclose(
        zoom_factor,
        1.0,
    ):
        return result

    crop_width = max(
        1,
        int(round(result.width / zoom_factor)),
    )

    crop_height = max(
        1,
        int(round(result.height / zoom_factor)),
    )

    left = max(
        0,
        (result.width - crop_width) // 2,
    )

    top = max(
        0,
        (result.height - crop_height) // 2,
    )

    right = min(
        result.width,
        left + crop_width,
    )

    bottom = min(
        result.height,
        top + crop_height,
    )

    return result.crop(
        (
            left,
            top,
            right,
            bottom,
        )
    )


def prepare_cell_image(
    image: Image.Image,
    target_width: int,
    target_height: int,
    zoom_factor: float,
) -> Image.Image:
    """
    Crop and resize one original rendering for the combined figure.
    """

    result = centre_crop_for_zoom(
        image=image,
        zoom_factor=zoom_factor,
    )

    result.thumbnail(
        (
            target_width,
            target_height,
        ),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (
            target_width,
            target_height,
        ),
        "white",
    )

    x_offset = (
        target_width - result.width
    ) // 2

    y_offset = (
        target_height - result.height
    ) // 2

    canvas.paste(
        result,
        (
            x_offset,
            y_offset,
        ),
    )

    return canvas


def save_selection_metadata(
    selected: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Save the selected center and peripheral placement metadata."""

    export_records = []

    for selection_type in OUTPUT_ORDER:
        record = selected[selection_type]

        export_records.append(
            {
                "selection_type": selection_type,
                "archive_index": record["archive_index"],
                "fitness": record["fitness"],
                "grasp_proxy": record["grasp_proxy"],
                "center_distance": record["center_distance"],
                "planar_angle_rad": record["planar_angle"],
                "planar_angle_deg": math.degrees(
                    record["planar_angle"]
                ),
                "final_pose": record["final_pose"],
            }
        )

    np.save(
        output_path,
        export_records,
        allow_pickle=True,
    )


def print_selection(
    object_name: str,
    archive_path: Path,
    selected: dict[str, dict[str, Any]],
) -> None:
    """Print the selected central and peripheral placements."""

    print("\n" + "=" * 78)
    print(f"Object: {object_name}")
    print(f"Archive: {archive_path}")

    for selection_type in OUTPUT_ORDER:
        record = selected[selection_type]

        print(
            f"{selection_type:>8}: "
            f"index={record['archive_index']:>5d}, "
            f"d={record['center_distance']:.6f} m, "
            f"angle={math.degrees(record['planar_angle']):.2f} deg, "
            f"xyz={record['xyz']}, "
            f"GFI={record['grasp_proxy']:.6f}"
        )


def create_combined_figure(
    rendered_images: dict[str, dict[str, Image.Image]],
    output_directory: Path,
    zoom_factor: float,
) -> None:
    """Create the labelled six-object centrality comparison figure."""

    cell_width = CELL_WIDTH
    cell_height = CELL_HEIGHT

    canvas_width = (
        LEFT_MARGIN
        + len(OUTPUT_ORDER) * cell_width
        + (len(OUTPUT_ORDER) - 1) * COLUMN_GAP
    )

    canvas_height = (
        TOP_MARGIN
        + len(OBJECT_ORDER) * cell_height
        + (len(OBJECT_ORDER) - 1) * ROW_GAP
        + BOTTOM_MARGIN
    )

    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            canvas_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    title_font = load_font(
        COLUMN_TITLE_FONT_SIZE,
        bold=True,
    )

    row_font = load_font(
        OBJECT_LABEL_FONT_SIZE,
        bold=True,
    )

    # Draw column headings.
    for column_index, selection_type in enumerate(OUTPUT_ORDER):
        title = COLUMN_TITLES[selection_type]

        column_x = (
            LEFT_MARGIN
            + column_index * (
                cell_width + COLUMN_GAP
            )
        )

        bounding_box = draw.textbbox(
            (0, 0),
            title,
            font=title_font,
        )

        text_width = (
            bounding_box[2]
            - bounding_box[0]
        )

        draw.text(
            (
                column_x
                + (cell_width - text_width) / 2,
                45,
            ),
            title,
            font=title_font,
            fill="black",
        )

    # Draw object labels and panels.
    for row_index, object_name in enumerate(OBJECT_ORDER):
        row_y = (
            TOP_MARGIN
            + row_index * (
                cell_height + ROW_GAP
            )
        )

        display_name = OBJECT_DISPLAY_NAMES[
            object_name
        ]

        bounding_box = draw.multiline_textbbox(
            (0, 0),
            display_name,
            font=row_font,
            spacing=4,
            align="center",
        )

        text_width = (
            bounding_box[2]
            - bounding_box[0]
        )

        text_height = (
            bounding_box[3]
            - bounding_box[1]
        )

        label_x = max(
            12,
            int(
                (
                    LEFT_MARGIN
                    - text_width
                )
                / 2
            ),
        )

        label_y = int(
            row_y
            + (cell_height - text_height) / 2
            - bounding_box[1]
        )

        draw.multiline_text(
            (
                label_x,
                label_y,
            ),
            display_name,
            font=row_font,
            fill="black",
            spacing=4,
            align="center",
        )

        for column_index, selection_type in enumerate(OUTPUT_ORDER):
            column_x = (
                LEFT_MARGIN
                + column_index * (
                    cell_width + COLUMN_GAP
                )
            )

            cell_image = prepare_cell_image(
                image=rendered_images[
                    object_name
                ][selection_type],
                target_width=cell_width,
                target_height=cell_height,
                zoom_factor=zoom_factor,
            )

            canvas.paste(
                cell_image,
                (
                    column_x,
                    row_y,
                ),
            )

    labelled_path = (
        output_directory
        / "all_objects_center_vs_far.png"
    )

    canvas.save(
        labelled_path,
        dpi=(600, 600),
    )

    print(
        "\nCombined labelled figure saved to:"
    )
    print(labelled_path)


def create_plain_combined_figure(
    rendered_images: dict[str, dict[str, Image.Image]],
    output_directory: Path,
    zoom_factor: float,
) -> None:
    """Create the same centrality comparison without text labels."""

    cell_width = CELL_WIDTH
    cell_height = CELL_HEIGHT

    canvas_width = (
        len(OUTPUT_ORDER) * cell_width
        + (len(OUTPUT_ORDER) - 1) * COLUMN_GAP
    )

    canvas_height = (
        len(OBJECT_ORDER) * cell_height
        + (len(OBJECT_ORDER) - 1) * ROW_GAP
    )

    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            canvas_height,
        ),
        "white",
    )

    for row_index, object_name in enumerate(OBJECT_ORDER):
        row_y = row_index * (
            cell_height + ROW_GAP
        )

        for column_index, selection_type in enumerate(OUTPUT_ORDER):
            column_x = column_index * (
                cell_width + COLUMN_GAP
            )

            cell_image = prepare_cell_image(
                image=rendered_images[
                    object_name
                ][selection_type],
                target_width=cell_width,
                target_height=cell_height,
                zoom_factor=zoom_factor,
            )

            canvas.paste(
                cell_image,
                (
                    column_x,
                    row_y,
                ),
            )

    plain_path = (
        output_directory
        / "all_objects_center_vs_far_plain.png"
    )

    canvas.save(
        plain_path,
        dpi=(600, 600),
    )

    print(
        "Combined plain figure saved to:"
    )
    print(plain_path)


def main() -> None:
    """Render and export the central and peripheral comparisons."""

    arguments = parse_arguments()

    if arguments.width <= 0:
        raise ValueError(
            "--width must be greater than zero."
        )

    if arguments.height <= 0:
        raise ValueError(
            "--height must be greater than zero."
        )

    if arguments.zoom_factor < 1.0:
        raise ValueError(
            "--zoom-factor must be at least 1.0."
        )

    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    minimum_angular_separation = math.radians(
        arguments.angular_separation_deg
    )

    rendered_images: dict[
        str,
        dict[str, Image.Image],
    ] = {}

    connection_id = p.connect(p.DIRECT)

    if connection_id < 0:
        raise RuntimeError(
            "Could not connect to PyBullet in DIRECT mode."
        )

    try:
        for object_name in OBJECT_ORDER:
            if object_name not in OBJECTS:
                raise KeyError(
                    f"Object configuration missing for: {object_name}"
                )

            configuration = OBJECTS[object_name]

            run_directory = Path(
                configuration["run"]
            )

            object_urdf = str(
                configuration["urdf"]
            )

            archive_path = find_latest_success_archive(
                run_directory
            )

            with np.load(
                archive_path,
                allow_pickle=True,
            ) as archive:
                required_keys = {
                    "infos",
                    "fitnesses",
                }

                missing_keys = required_keys.difference(
                    archive.files
                )

                if missing_keys:
                    raise KeyError(
                        f"Archive {archive_path} is missing keys: "
                        f"{sorted(missing_keys)}"
                    )

                records = build_candidate_records(
                    infos=archive["infos"],
                    fitnesses=archive["fitnesses"],
                )

            selected = select_centre_and_peripheral(
                records=records,
                minimum_angular_separation=(
                    minimum_angular_separation
                ),
            )

            print_selection(
                object_name=OBJECT_DISPLAY_NAMES[object_name],
                archive_path=archive_path,
                selected=selected,
            )

            object_output_directory = (
                arguments.output_dir
                / object_name
            )

            object_output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            save_selection_metadata(
                selected=selected,
                output_path=(
                    object_output_directory
                    / "selected_placements.npy"
                ),
            )

            rendered_images[object_name] = {}

            for selection_type in OUTPUT_ORDER:
                image = render_selected_pose(
                    object_name=object_name,
                    object_urdf=object_urdf,
                    record=selected[selection_type],
                    width=arguments.width,
                    height=arguments.height,
                )

                image_path = (
                    object_output_directory
                    / f"{selection_type}.png"
                )

                # Save the uncropped original PyBullet rendering.
                image.save(
                    image_path,
                    dpi=(600, 600),
                )

                rendered_images[object_name][
                    selection_type
                ] = image

                print(f"Saved: {image_path}")

        create_combined_figure(
            rendered_images=rendered_images,
            output_directory=arguments.output_dir,
            zoom_factor=arguments.zoom_factor,
        )

        create_plain_combined_figure(
            rendered_images=rendered_images,
            output_directory=arguments.output_dir,
            zoom_factor=arguments.zoom_factor,
        )

    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
