from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


RESULTS = Path("results")

CELL_WIDTH = 700
CELL_HEIGHT = 500

# Increased margins to accommodate larger labels when the
# figure is reduced to one-third page width in the manuscript.
LEFT_MARGIN = 360
TOP_MARGIN = 150
BOTTOM_MARGIN = 30

COLUMN_GAP = 12
ROW_GAP = 12

def load_font(size: int, bold: bool = False):
    """Load an available TrueType font."""

    if bold:
        paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
    else:
        paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]

    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)

    return ImageFont.load_default()


def prepare_image(
    image_path: Path,
    zoom_factor: float = 1.75,
) -> Image.Image:
    """
    Apply the same presentation-only centre crop used in the
    manuscript figure scripts.

    This does not modify the simulated pose or object geometry.
    """

    image = Image.open(image_path).convert("RGB")

    crop_width = max(
        1,
        int(round(image.width / zoom_factor)),
    )

    crop_height = max(
        1,
        int(round(image.height / zoom_factor)),
    )

    left = max(
        0,
        (image.width - crop_width) // 2,
    )

    top = max(
        0,
        (image.height - crop_height) // 2,
    )

    image = image.crop(
        (
            left,
            top,
            left + crop_width,
            top + crop_height,
        )
    )

    image.thumbnail(
        (
            CELL_WIDTH,
            CELL_HEIGHT,
        ),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (
            CELL_WIDTH,
            CELL_HEIGHT,
        ),
        "white",
    )

    x_offset = (
        CELL_WIDTH - image.width
    ) // 2

    y_offset = (
        CELL_HEIGHT - image.height
    ) // 2

    canvas.paste(
        image,
        (
            x_offset,
            y_offset,
        ),
    )

    return canvas


def create_compact_figure(
    objects,
    categories,
    titles,
    source_root: Path,
    output_path: Path,
) -> None:
    """Create one compact two-object, three-column figure."""

    canvas_width = (
        LEFT_MARGIN
        + len(categories) * CELL_WIDTH
        + (len(categories) - 1) * COLUMN_GAP
    )

    canvas_height = (
        TOP_MARGIN
        + len(objects) * CELL_HEIGHT
        + (len(objects) - 1) * ROW_GAP
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

    # Large bold labels for readability after reduction
    # in the final RA-L manuscript figure.
    title_font = load_font(
        78,
        bold=True,
    )

    object_font = load_font(
        82,
        bold=True,
    )

    # ---------------------------------------------------------
    # Column titles
    # ---------------------------------------------------------

    for column_index, category in enumerate(categories):
        title = titles[category]

        column_x = (
            LEFT_MARGIN
            + column_index * (
                CELL_WIDTH + COLUMN_GAP
            )
        )

        box = draw.textbbox(
            (0, 0),
            title,
            font=title_font,
        )

        text_width = (
            box[2] - box[0]
        )

        text_height = (
            box[3] - box[1]
        )

        draw.text(
            (
                column_x
                + (CELL_WIDTH - text_width) / 2,
                (TOP_MARGIN - text_height) / 2
                - box[1],
            ),
            title,
            font=title_font,
            fill="black",
        )

    # ---------------------------------------------------------
    # Object rows
    # ---------------------------------------------------------

    for row_index, (
        object_key,
        object_label,
    ) in enumerate(objects):

        row_y = (
            TOP_MARGIN
            + row_index * (
                CELL_HEIGHT + ROW_GAP
            )
        )

        box = draw.textbbox(
            (0, 0),
            object_label,
            font=object_font,
        )

        text_width = (
            box[2] - box[0]
        )

        text_height = (
            box[3] - box[1]
        )

        draw.text(
            (
                (LEFT_MARGIN - text_width) / 2,
                row_y
                + (CELL_HEIGHT - text_height) / 2
                - box[1],
            ),
            object_label,
            font=object_font,
            fill="black",
        )

        # -----------------------------------------------------
        # Simulation images
        # -----------------------------------------------------

        for column_index, category in enumerate(categories):

            column_x = (
                LEFT_MARGIN
                + column_index * (
                    CELL_WIDTH + COLUMN_GAP
                )
            )

            image_path = (
                source_root
                / object_key
                / f"{category}.png"
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing source image: {image_path}"
                )

            image = prepare_image(
                image_path
            )

            canvas.paste(
                image,
                (
                    column_x,
                    row_y,
                ),
            )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        output_path,
        dpi=(600, 600),
    )

    print(
        f"Saved: {output_path}"
    )


# =============================================================
# CENTRALITY
# Mug + Bowl
# =============================================================

create_compact_figure(
    objects=[
        ("mug", "Mug"),
        ("bowl", "Bowl"),
    ],

    categories=[
        "far_a",
        "centre",
        "far_b",
    ],

    titles={
        "far_a": "Peripheral A",
        "centre": "Central",
        "far_b": "Peripheral B",
    },

    source_root=(
        RESULTS
        / "center_far_placements"
    ),

    output_path=(
        RESULTS
        / "center_far_placements"
        / "compact_mug_bowl_centrality.png"
    ),
)


# =============================================================
# UPRIGHTNESS
# Mug + Bowl
# =============================================================

create_compact_figure(
    objects=[
        ("mug", "Mug"),
        ("bowl", "Bowl"),
    ],

    categories=[
        "upright",
        "intermediate",
        "tilted",
    ],

    titles={
        "upright": "Most Upright",
        "intermediate": "Intermediate",
        "tilted": "Most Tilted",
    },

    source_root=(
        RESULTS
        / "gfi_component_figures"
        / "uprightness"
    ),

    output_path=(
        RESULTS
        / "gfi_component_figures"
        / "uprightness"
        / "compact_mug_bowl_uprightness.png"
    ),
)
