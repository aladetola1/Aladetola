from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIRECTORY = Path(__file__).resolve().parent

for directory in (REPOSITORY_ROOT, SCRIPTS_DIRECTORY):
    directory_string = str(directory)

    if directory_string not in sys.path:
        sys.path.insert(0, directory_string)


from render_top10_poses_video import (  # noqa: E402
    OBJECTS,
    create_scene,
    render_scene,
)


OUTPUT_ROOT = Path("results/gfi_component_figures")

OBJECT_ORDER = [
    "mug",
    "bowl",
    "banana",
    "plate",
    "power_drill",
    "spoon",
]

OBJECT_DISPLAY_NAMES = {
    "mug": "Mug",
    "bowl": "Bowl",
    "banana": "Banana",
    "plate": "Plate",
    "power_drill": "Power\nDrill",
    "spoon": "Spoon",
}

HEIGHT_ORDER = [
    "low",
    "intermediate",
    "high",
]

HEIGHT_TITLES = {
    "low": "Low Stable Height",
    "intermediate": "Intermediate Stable Height",
    "high": "High Stable Height",
}

UPRIGHTNESS_ORDER = [
    "upright",
    "intermediate",
    "tilted",
]

UPRIGHTNESS_TITLES = {
    "upright": "Most Upright Placement",
    "intermediate": "Intermediate Tilt",
    "tilted": "Most Tilted Placement",
}


# ---------------------------------------------------------------------
# Rendering and combined-figure presentation settings
# ---------------------------------------------------------------------

RENDER_WIDTH = 2200
RENDER_HEIGHT = 1600

CELL_WIDTH = 700
CELL_HEIGHT = 500

# Presentation-only crop factor. This enlarges the object/platform
# in the final panel without changing the simulated object or pose.
CELL_ZOOM_FACTOR = 1.75

COLUMN_TITLE_FONT_SIZE = 48
OBJECT_LABEL_FONT_SIZE = 52
OBJECT_LABEL_LINE_SPACING = 4

LEFT_MARGIN = 310
TOP_MARGIN = 145
BOTTOM_MARGIN = 45
COLUMN_GAP = 12
ROW_GAP = 12


def archive_number(path: Path) -> int:
    """Return the numeric suffix in individuals_<number>.npz."""

    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


def find_latest_success_archive(run_directory: Path) -> Path:
    """Locate the most recent success archive for one run."""

    archive_directory = run_directory / "success_archives"

    if not archive_directory.exists():
        raise FileNotFoundError(
            f"Missing success archive directory: {archive_directory}"
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
        key=archive_number,
    )


def convert_info(raw_info: Any) -> dict[str, Any]:
    """Convert one stored archive information entry to a dictionary."""

    if isinstance(raw_info, dict):
        return raw_info

    if hasattr(raw_info, "item"):
        converted = raw_info.item()

        if isinstance(converted, dict):
            return converted

    raise TypeError(
        "The archive information entry is not a dictionary."
    )


def wrap_to_pi(angle: float) -> float:
    """Wrap one angle to the interval [-pi, pi]."""

    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


def extract_final_pose(
    info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract final xyz position and Euler roll-pitch-yaw pose."""

    if "final_6dof_pose" not in info:
        raise KeyError(
            "The archive entry has no 'final_6dof_pose'."
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
            f"Invalid xyz shape: {xyz.shape}"
        )

    if rpy.shape != (3,):
        raise ValueError(
            f"Invalid euler_rpy shape: {rpy.shape}"
        )

    return xyz, rpy


def compute_roll_pitch_tilt(rpy: np.ndarray) -> float:
    """
    Compute a yaw-independent roll-pitch tilt magnitude.

    Yaw is intentionally excluded because yaw does not describe
    deviation from vertical uprightness.
    """

    roll = wrap_to_pi(float(rpy[0]))
    pitch = wrap_to_pi(float(rpy[1]))

    return float(
        math.sqrt(
            roll**2 + pitch**2
        )
    )


def candidate_is_usable(info: dict[str, Any]) -> bool:
    """Return True when an archive entry is valid, successful, and finite."""

    is_valid = float(
        info.get("is_valid", 1.0)
    )

    is_success = float(
        info.get("is_success", 1.0)
    )

    if is_valid < 0.5 or is_success < 0.5:
        return False

    try:
        xyz, rpy = extract_final_pose(info)
    except (KeyError, TypeError, ValueError):
        return False

    return bool(
        np.all(np.isfinite(xyz))
        and np.all(np.isfinite(rpy))
    )


def build_records(
    infos: np.ndarray,
    fitnesses: np.ndarray,
) -> list[dict[str, Any]]:
    """Build normalized records from the success archive."""

    records: list[dict[str, Any]] = []

    for archive_index, raw_info in enumerate(infos):
        info = convert_info(raw_info)

        if not candidate_is_usable(info):
            continue

        xyz, rpy = extract_final_pose(info)

        roll = wrap_to_pi(float(rpy[0]))
        pitch = wrap_to_pi(float(rpy[1]))
        yaw = wrap_to_pi(float(rpy[2]))

        tilt = compute_roll_pitch_tilt(rpy)

        grasp_proxy = float(
            info.get(
                "grasp_proxy",
                float("nan"),
            )
        )

        records.append(
            {
                "archive_index": int(archive_index),
                "fitness": float(fitnesses[archive_index]),
                "grasp_proxy": grasp_proxy,
                "final_pose": info["final_6dof_pose"],
                "xyz": xyz,
                "rpy": rpy,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "height": float(xyz[2]),
                "tilt_rad": tilt,
                "tilt_deg": math.degrees(tilt),
                "center_distance": float(
                    np.linalg.norm(xyz[:2])
                ),
            }
        )

    if len(records) < 3:
        raise RuntimeError(
            "Fewer than three usable stable placements were found."
        )

    return records


def select_three_by_sorted_value(
    records: list[dict[str, Any]],
    key: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """
    Return the minimum, median, and maximum records for one quantity.
    """

    ordered = sorted(
        records,
        key=lambda record: record[key],
    )

    lowest = ordered[0]
    intermediate = ordered[len(ordered) // 2]
    highest = ordered[-1]

    return (
        lowest,
        intermediate,
        highest,
    )


def select_height_examples(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Select low, intermediate, and high stable-height examples."""

    low, intermediate, high = select_three_by_sorted_value(
        records=records,
        key="height",
    )

    return {
        "low": low,
        "intermediate": intermediate,
        "high": high,
    }


def select_uprightness_examples(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Select upright, intermediate, and most-tilted examples."""

    upright, intermediate, tilted = (
        select_three_by_sorted_value(
            records=records,
            key="tilt_rad",
        )
    )

    return {
        "upright": upright,
        "intermediate": intermediate,
        "tilted": tilted,
    }


def render_record(
    object_name: str,
    object_urdf: str,
    record: dict[str, Any],
) -> Image.Image:
    """Render one archived stable placement at high resolution."""

    create_scene(
        object_name=object_name,
        object_urdf=object_urdf,
        xyz=record["xyz"],
        rpy=record["rpy"],
    )

    rendered = render_scene(
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
    )

    if not isinstance(rendered, Image.Image):
        rendered = Image.fromarray(
            np.asarray(rendered).astype(np.uint8)
        )

    return rendered.convert("RGB")


def load_font(
    size: int,
    bold: bool = False,
) -> ImageFont.ImageFont:
    """Load an available TrueType font."""

    if bold:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
    else:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]

    for font_path in font_paths:
        path = Path(font_path)

        if path.exists():
            return ImageFont.truetype(
                str(path),
                size=size,
            )

    return ImageFont.load_default()


def centre_crop_for_zoom(
    image: Image.Image,
    zoom_factor: float,
) -> Image.Image:
    """
    Apply a centred crop to enlarge the visible object and platform.

    This operation changes only the image framing. It does not change
    the simulated object mesh, object pose, platform size, or scene.
    """

    if zoom_factor < 1.0:
        raise ValueError(
            "zoom_factor must be greater than or equal to 1.0."
        )

    result = image.convert("RGB").copy()

    if math.isclose(zoom_factor, 1.0):
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
    target_width: int = CELL_WIDTH,
    target_height: int = CELL_HEIGHT,
    zoom_factor: float = CELL_ZOOM_FACTOR,
) -> Image.Image:
    """
    Crop and resize one rendering consistently for the journal figure.
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


def draw_object_label(
    draw: ImageDraw.ImageDraw,
    display_name: str,
    row_y: int,
    cell_height: int,
    font: ImageFont.ImageFont,
) -> None:
    """
    Draw a centered row label.

    Multiline labels are supported, allowing Power Drill to appear as:
    Power
    Drill
    """

    bounding_box = draw.multiline_textbbox(
        (0, 0),
        display_name,
        font=font,
        spacing=OBJECT_LABEL_LINE_SPACING,
        align="center",
    )

    text_width = bounding_box[2] - bounding_box[0]
    text_height = bounding_box[3] - bounding_box[1]

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
        font=font,
        fill="black",
        spacing=OBJECT_LABEL_LINE_SPACING,
        align="center",
    )


def create_combined_figure(
    rendered_images: dict[str, dict[str, Image.Image]],
    component_order: list[str],
    column_titles: dict[str, str],
    output_path: Path,
) -> None:
    """Create the labelled six-object, three-category comparison figure."""

    cell_width = CELL_WIDTH
    cell_height = CELL_HEIGHT

    canvas_width = (
        LEFT_MARGIN
        + 3 * cell_width
        + 2 * COLUMN_GAP
    )

    canvas_height = (
        TOP_MARGIN
        + 6 * cell_height
        + 5 * ROW_GAP
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

    for column_index, category in enumerate(component_order):
        title = column_titles[category]

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

        text_width = bounding_box[2] - bounding_box[0]

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

        draw_object_label(
            draw=draw,
            display_name=display_name,
            row_y=row_y,
            cell_height=cell_height,
            font=row_font,
        )

        for column_index, category in enumerate(component_order):
            column_x = (
                LEFT_MARGIN
                + column_index * (
                    cell_width + COLUMN_GAP
                )
            )

            cell_image = prepare_cell_image(
                rendered_images[object_name][category],
                target_width=cell_width,
                target_height=cell_height,
                zoom_factor=CELL_ZOOM_FACTOR,
            )

            canvas.paste(
                cell_image,
                (
                    column_x,
                    row_y,
                ),
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        output_path,
        dpi=(600, 600),
    )

    print(
        f"Combined figure saved: {output_path}"
    )


def create_plain_figure(
    rendered_images: dict[str, dict[str, Image.Image]],
    component_order: list[str],
    output_path: Path,
) -> None:
    """Create the same comparison grid without text labels."""

    cell_width = CELL_WIDTH
    cell_height = CELL_HEIGHT

    canvas_width = (
        3 * cell_width
        + 2 * COLUMN_GAP
    )

    canvas_height = (
        6 * cell_height
        + 5 * ROW_GAP
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

        for column_index, category in enumerate(component_order):
            column_x = column_index * (
                cell_width + COLUMN_GAP
            )

            cell_image = prepare_cell_image(
                rendered_images[object_name][category],
                target_width=cell_width,
                target_height=cell_height,
                zoom_factor=CELL_ZOOM_FACTOR,
            )

            canvas.paste(
                cell_image,
                (
                    column_x,
                    row_y,
                ),
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        output_path,
        dpi=(600, 600),
    )

    print(
        f"Plain figure saved: {output_path}"
    )


def save_metadata_csv(
    metadata_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save the selected placements and associated metrics."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    field_names = [
        "object",
        "component",
        "category",
        "archive_index",
        "x_m",
        "y_m",
        "z_m",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "tilt_deg",
        "center_distance_m",
        "fitness",
        "gfi",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=field_names,
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    print(
        f"Metadata saved: {output_path}"
    )


def append_metadata(
    metadata_rows: list[dict[str, Any]],
    object_name: str,
    component_name: str,
    category: str,
    record: dict[str, Any],
) -> None:
    """Append one selected placement to the metadata list."""

    metadata_rows.append(
        {
            "object": object_name,
            "component": component_name,
            "category": category,
            "archive_index": record["archive_index"],
            "x_m": record["xyz"][0],
            "y_m": record["xyz"][1],
            "z_m": record["xyz"][2],
            "roll_rad": record["roll"],
            "pitch_rad": record["pitch"],
            "yaw_rad": record["yaw"],
            "roll_deg": math.degrees(
                record["roll"]
            ),
            "pitch_deg": math.degrees(
                record["pitch"]
            ),
            "yaw_deg": math.degrees(
                record["yaw"]
            ),
            "tilt_deg": record["tilt_deg"],
            "center_distance_m": record[
                "center_distance"
            ],
            "fitness": record["fitness"],
            "gfi": record["grasp_proxy"],
        }
    )


def print_height_selection(
    object_name: str,
    selected: dict[str, dict[str, Any]],
) -> None:
    """Print the selected low, intermediate, and high placements."""

    print("\nHEIGHT SELECTION")

    for category in HEIGHT_ORDER:
        record = selected[category]

        print(
            f"  {category:>12}: "
            f"index={record['archive_index']:>4d}, "
            f"z={record['height']:.6f} m, "
            f"tilt={record['tilt_deg']:.3f} deg, "
            f"GFI={record['grasp_proxy']:.6f}"
        )


def print_uprightness_selection(
    object_name: str,
    selected: dict[str, dict[str, Any]],
) -> None:
    """Print the selected uprightness examples."""

    print("UPRIGHTNESS SELECTION")

    for category in UPRIGHTNESS_ORDER:
        record = selected[category]

        print(
            f"  {category:>12}: "
            f"index={record['archive_index']:>4d}, "
            f"roll={math.degrees(record['roll']):.3f} deg, "
            f"pitch={math.degrees(record['pitch']):.3f} deg, "
            f"tilt={record['tilt_deg']:.3f} deg, "
            f"GFI={record['grasp_proxy']:.6f}"
        )


def main() -> None:
    """Render and export the height and uprightness comparisons."""

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    height_images: dict[
        str,
        dict[str, Image.Image],
    ] = {}

    uprightness_images: dict[
        str,
        dict[str, Image.Image],
    ] = {}

    metadata_rows: list[dict[str, Any]] = []

    client_id = p.connect(p.DIRECT)

    if client_id < 0:
        raise RuntimeError(
            "Could not connect to PyBullet."
        )

    try:
        for object_name in OBJECT_ORDER:
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
                if "infos" not in archive.files:
                    raise KeyError(
                        f"'infos' missing from {archive_path}"
                    )

                if "fitnesses" not in archive.files:
                    raise KeyError(
                        f"'fitnesses' missing from {archive_path}"
                    )

                records = build_records(
                    infos=archive["infos"],
                    fitnesses=archive["fitnesses"],
                )

            height_selected = select_height_examples(
                records
            )

            uprightness_selected = (
                select_uprightness_examples(
                    records
                )
            )

            print("\n" + "=" * 78)
            print(
                f"Object: {OBJECT_DISPLAY_NAMES[object_name].replace(chr(10), ' ')}"
            )
            print(
                f"Archive: {archive_path}"
            )

            print_height_selection(
                object_name,
                height_selected,
            )

            print_uprightness_selection(
                object_name,
                uprightness_selected,
            )

            height_object_directory = (
                OUTPUT_ROOT
                / "height"
                / object_name
            )

            uprightness_object_directory = (
                OUTPUT_ROOT
                / "uprightness"
                / object_name
            )

            height_object_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            uprightness_object_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            height_images[object_name] = {}
            uprightness_images[object_name] = {}

            for category in HEIGHT_ORDER:
                record = height_selected[category]

                image = render_record(
                    object_name=object_name,
                    object_urdf=object_urdf,
                    record=record,
                )

                image_path = (
                    height_object_directory
                    / f"{category}.png"
                )

                image.save(
                    image_path,
                    dpi=(600, 600),
                )

                height_images[object_name][
                    category
                ] = image

                append_metadata(
                    metadata_rows=metadata_rows,
                    object_name=object_name,
                    component_name="height",
                    category=category,
                    record=record,
                )

                print(
                    f"Saved: {image_path}"
                )

            for category in UPRIGHTNESS_ORDER:
                record = uprightness_selected[
                    category
                ]

                image = render_record(
                    object_name=object_name,
                    object_urdf=object_urdf,
                    record=record,
                )

                image_path = (
                    uprightness_object_directory
                    / f"{category}.png"
                )

                image.save(
                    image_path,
                    dpi=(600, 600),
                )

                uprightness_images[object_name][
                    category
                ] = image

                append_metadata(
                    metadata_rows=metadata_rows,
                    object_name=object_name,
                    component_name="uprightness",
                    category=category,
                    record=record,
                )

                print(
                    f"Saved: {image_path}"
                )

        create_combined_figure(
            rendered_images=height_images,
            component_order=HEIGHT_ORDER,
            column_titles=HEIGHT_TITLES,
            output_path=(
                OUTPUT_ROOT
                / "height"
                / "all_objects_height_comparison.png"
            ),
        )

        create_plain_figure(
            rendered_images=height_images,
            component_order=HEIGHT_ORDER,
            output_path=(
                OUTPUT_ROOT
                / "height"
                / "all_objects_height_comparison_plain.png"
            ),
        )

        create_combined_figure(
            rendered_images=uprightness_images,
            component_order=UPRIGHTNESS_ORDER,
            column_titles=UPRIGHTNESS_TITLES,
            output_path=(
                OUTPUT_ROOT
                / "uprightness"
                / "all_objects_uprightness_comparison.png"
            ),
        )

        create_plain_figure(
            rendered_images=uprightness_images,
            component_order=UPRIGHTNESS_ORDER,
            output_path=(
                OUTPUT_ROOT
                / "uprightness"
                / "all_objects_uprightness_comparison_plain.png"
            ),
        )

        save_metadata_csv(
            metadata_rows=metadata_rows,
            output_path=(
                OUTPUT_ROOT
                / "selected_component_placements.csv"
            ),
        )

    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
