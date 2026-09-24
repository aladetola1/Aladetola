from pathlib import Path
from PIL import Image, ImageDraw

import render_gfi_component_figures as fig3


# =============================================================
# HEIGHT FIGURE
# Banana + Power Drill
# =============================================================

RESULTS = Path("results")

SOURCE_ROOT = (
    RESULTS
    / "gfi_component_figures"
    / "height"
)

OUTPUT_PATH = (
    SOURCE_ROOT
    / "figure3_selected_stable_height_examples.png"
)


# -------------------------------------------------------------
# Layout
# -------------------------------------------------------------

CELL_WIDTH = 700
CELL_HEIGHT = 500

# Larger left margin so "Power Drill" is never clipped.
LEFT_MARGIN = 390

# More room above images for two-line column headings.
TOP_MARGIN = 190

BOTTOM_MARGIN = 30
COLUMN_GAP = 20
ROW_GAP = 12


objects = [
    ("banana", "Banana"),
    ("power_drill", "Power\nDrill"),
]

categories = [
    "low",
    "intermediate",
    "high",
]

titles = {
    "low": "Low Resting\nHeight",
    "intermediate": "Intermediate\nResting Height",
    "high": "High Resting\nHeight",
}


# -------------------------------------------------------------
# Canvas
# -------------------------------------------------------------

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


# -------------------------------------------------------------
# Fonts
# -------------------------------------------------------------

# Height has substantially longer column headings than the
# other GFI panels, so use a slightly smaller heading font.
title_font = fig3.load_font(
    62,
    bold=True,
)

object_font = fig3.load_font(
    76,
    bold=True,
)


# -------------------------------------------------------------
# Column headings
# -------------------------------------------------------------

for column_index, category in enumerate(categories):

    title = titles[category]

    column_x = (
        LEFT_MARGIN
        + column_index
        * (
            CELL_WIDTH
            + COLUMN_GAP
        )
    )

    box = draw.multiline_textbbox(
        (0, 0),
        title,
        font=title_font,
        spacing=4,
        align="center",
    )

    text_width = (
        box[2] - box[0]
    )

    text_height = (
        box[3] - box[1]
    )

    draw.multiline_text(
        (
            column_x
            + (CELL_WIDTH - text_width) / 2,
            (TOP_MARGIN - text_height) / 2
            - box[1],
        ),
        title,
        font=title_font,
        fill="black",
        spacing=4,
        align="center",
    )


# -------------------------------------------------------------
# Object rows
# -------------------------------------------------------------

for row_index, (
    object_key,
    object_label,
) in enumerate(objects):

    row_y = (
        TOP_MARGIN
        + row_index
        * (
            CELL_HEIGHT
            + ROW_GAP
        )
    )

    box = draw.multiline_textbbox(
        (0, 0),
        object_label,
        font=object_font,
        spacing=4,
        align="center",
    )

    text_width = (
        box[2] - box[0]
    )

    text_height = (
        box[3] - box[1]
    )

    draw.multiline_text(
        (
            (LEFT_MARGIN - text_width) / 2,
            row_y
            + (CELL_HEIGHT - text_height) / 2
            - box[1],
        ),
        object_label,
        font=object_font,
        fill="black",
        spacing=4,
        align="center",
    )

    # ---------------------------------------------------------
    # Existing simulation images
    # ---------------------------------------------------------

    for column_index, category in enumerate(categories):

        column_x = (
            LEFT_MARGIN
            + column_index
            * (
                CELL_WIDTH
                + COLUMN_GAP
            )
        )

        image_path = (
            SOURCE_ROOT
            / object_key
            / f"{category}.png"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing source image: {image_path}"
            )

        image = fig3.prepare_image(
            image_path
        )

        canvas.paste(
            image,
            (
                column_x,
                row_y,
            ),
        )


# -------------------------------------------------------------
# Save
# -------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

canvas.save(
    OUTPUT_PATH,
    dpi=(600, 600),
)

print(
    f"Saved: {OUTPUT_PATH}"
)
