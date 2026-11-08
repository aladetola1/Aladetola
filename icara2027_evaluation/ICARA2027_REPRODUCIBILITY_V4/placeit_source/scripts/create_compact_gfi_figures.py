from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


RESULTS = Path("results")

CELL_WIDTH = 700
CELL_HEIGHT = 500
LEFT_MARGIN = 260
TOP_MARGIN = 120
BOTTOM_MARGIN = 30
COLUMN_GAP = 12
ROW_GAP = 12


def font(size, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]

    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)

    return ImageFont.load_default()


def prepare(image_path):
    image = Image.open(image_path).convert("RGB")

    # Same presentation-only 1.75x centre crop used by the
    # existing manuscript figure scripts.
    zoom = 1.75

    crop_w = int(round(image.width / zoom))
    crop_h = int(round(image.height / zoom))

    left = (image.width - crop_w) // 2
    top = (image.height - crop_h) // 2

    image = image.crop(
        (left, top, left + crop_w, top + crop_h)
    )

    image.thumbnail(
        (CELL_WIDTH, CELL_HEIGHT),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (CELL_WIDTH, CELL_HEIGHT),
        "white",
    )

    x = (CELL_WIDTH - image.width) // 2
    y = (CELL_HEIGHT - image.height) // 2

    canvas.paste(image, (x, y))

    return canvas


def make_figure(objects, categories, titles, source_root, output):
    width = (
        LEFT_MARGIN
        + len(categories) * CELL_WIDTH
        + (len(categories) - 1) * COLUMN_GAP
    )

    height = (
        TOP_MARGIN
        + len(objects) * CELL_HEIGHT
        + (len(objects) - 1) * ROW_GAP
        + BOTTOM_MARGIN
    )

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = font(42, bold=True)
    object_font = font(48, bold=True)

    for j, category in enumerate(categories):
        x = LEFT_MARGIN + j * (CELL_WIDTH + COLUMN_GAP)

        box = draw.textbbox((0, 0), titles[category], font=title_font)
        tw = box[2] - box[0]

        draw.text(
            (x + (CELL_WIDTH - tw) / 2, 35),
            titles[category],
            font=title_font,
            fill="black",
        )

    for i, (object_key, object_label) in enumerate(objects):
        y = TOP_MARGIN + i * (CELL_HEIGHT + ROW_GAP)

        box = draw.textbbox((0, 0), object_label, font=object_font)
        tw = box[2] - box[0]
        th = box[3] - box[1]

        draw.text(
            (
                (LEFT_MARGIN - tw) / 2,
                y + (CELL_HEIGHT - th) / 2,
            ),
            object_label,
            font=object_font,
            fill="black",
        )

        for j, category in enumerate(categories):
            x = LEFT_MARGIN + j * (CELL_WIDTH + COLUMN_GAP)

            image = prepare(
                source_root / object_key / f"{category}.png"
            )

            canvas.paste(image, (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(600, 600))

    print(f"Saved: {output}")


# ------------------------------------------------------------
# CENTRALITY: Mug + Bowl
# ------------------------------------------------------------

make_figure(
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
    source_root=RESULTS / "center_far_placements",
    output=(
        RESULTS
        / "center_far_placements"
        / "compact_mug_bowl_centrality.png"
    ),
)


# ------------------------------------------------------------
# UPRIGHTNESS: Mug + Bowl
# ------------------------------------------------------------

make_figure(
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
    output=(
        RESULTS
        / "gfi_component_figures"
        / "uprightness"
        / "compact_mug_bowl_uprightness.png"
    ),
)
