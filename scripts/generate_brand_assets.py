#!/usr/bin/env python3
"""
Generate the icon set that home-assistant/brands expects for this integration.

Home Assistant and HACS do not read icons from the integration repository.
They read them from home-assistant/brands, keyed by domain, which is why this
integration currently shows a blank tile. Getting an icon means opening a pull
request there adding `custom_integrations/mistral_conversation/`.

The mark is Mistral AI's, used to identify the service this integration talks
to. See `brands/README.md` for the standing this repository claims to it --
short version, none, beyond identifying the service.

It is drawn from a grid rather than resampled from a source file. The mark is
flat-coloured pixel art on an exact 7x5 grid -- every cell in the artwork is a
single flat colour, and every row is one colour -- so a grid reproduces it
exactly at any size, with no resampling artefacts.

The grid and the colours below were sampled from Mistral's artwork rather than
eyeballed. That artwork is not kept here, because this reproduces it; it is in
the history at f910fdb if the sampling ever needs rechecking.
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

# The mark, one character per cell. "X" is filled, "." is transparent.
GRID = (
    ".X...X.",
    ".XX.XX.",
    ".XXXXX.",
    ".X.X.X.",
    "XXX.XXX",
)

# One colour per row, top to bottom.
ROW_COLOURS = ("#FFAF01", "#FF8204", "#FA500F", "#E51300", "#C4001D")

# Pillow draws with hard edges, and at 7 cells across a 256px canvas the cell
# boundaries do not land on whole pixels. Rendering large and scaling down is
# what keeps those edges clean.
SUPERSAMPLE = 8

# brands wants icon.png at 256 and icon@2x.png at 512. logo.png is optional
# and falls back to the icon, so none is generated: a logo is a wordmark, and
# Mistral's wordmark is a bigger borrow than the glyph for no benefit here.
OUTPUTS = {"icon.png": 256, "icon@2x.png": 512}

OUTPUT_DIR = (
    Path(__file__).parent.parent
    / "brands"
    / "custom_integrations"
    / "mistral_conversation"
)


def render(size: int) -> Image.Image:
    """Draw the mark centred on a transparent square canvas of `size`."""
    columns = len(GRID[0])
    rows = len(GRID)

    # The icon has to be square, but the mark is wider than it is tall. Fill
    # the width and centre vertically: that leaves the image trimmed on the
    # axis it can be trimmed on, which is what brands checks for.
    cell = size / columns
    top = (size - cell * rows) / 2

    work = size * SUPERSAMPLE
    image = Image.new("RGBA", (work, work), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    for row_index, row in enumerate(GRID):
        for column_index, character in enumerate(row):
            if character != "X":
                continue
            left = round(column_index * cell * SUPERSAMPLE)
            right = round((column_index + 1) * cell * SUPERSAMPLE)
            upper = round((top + row_index * cell) * SUPERSAMPLE)
            lower = round((top + (row_index + 1) * cell) * SUPERSAMPLE)
            draw.rectangle(
                (left, upper, right - 1, lower - 1),
                fill=ROW_COLOURS[row_index],
            )

    return image.resize((size, size), Image.Resampling.LANCZOS)


def main() -> int:
    """Write the icon set, and report what was written."""
    if len(ROW_COLOURS) != len(GRID):
        print(f"❌ {len(GRID)} grid rows but {len(ROW_COLOURS)} colours")
        return 1
    if len({len(row) for row in GRID}) != 1:
        print("❌ Grid rows are not all the same width")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, size in OUTPUTS.items():
        path = OUTPUT_DIR / name
        render(size).save(path, "PNG", optimize=True)
        print(f"✅ Wrote {path.relative_to(Path(__file__).parent.parent)} ({size}px)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
