#!/usr/bin/env python3
"""
Generate the icon set that home-assistant/brands expects for this integration.

Home Assistant and HACS do not read icons from the integration repository.
They read them from home-assistant/brands, keyed by domain, which is why this
integration currently shows a blank tile. Getting an icon means opening a pull
request there adding `custom_integrations/mistral_conversation/`.

The mark drawn here is deliberately *not* Mistral AI's logo. This integration
is not officially associated with Mistral AI, and shipping their artwork under
that banner is not ours to do. It is a speech bubble -- this is a conversation
agent -- filled with a warm yellow-to-red ramp, which places it alongside the
service it talks to without borrowing the mark itself.

If you would rather ship the official artwork, replace the generated files
with it rather than editing this script; see `brands/README.md` for the sizes
and the trim rule they have to satisfy.
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

# Everything below is expressed against a 256x256 design grid and scaled up
# before drawing, so the geometry reads the same whatever we render at.
CANVAS = 256

# Pillow draws with hard edges. Rendering large and scaling down is what
# produces the antialiasing, so the rounded corners and the tail do not stair-
# step at the sizes these are actually seen at.
SUPERSAMPLE = 8
WORK = CANVAS * SUPERSAMPLE

# The bubble body, and the tail hanging off its bottom-left. Together their
# bounding box is the full canvas: brands rejects images with transparent
# padding around the mark.
BODY_BOTTOM = 194
CORNER_RADIUS = 62
TAIL = ((80, 150), (80, CANVAS), (152, 194))

# Five flat bands rather than a smooth gradient. At 32px -- the size in the
# integrations list, which is where this icon does its work -- a smooth ramp
# muddies into a single orange, while bands stay legible.
BANDS = ("#FFD800", "#FFAF00", "#FF8205", "#FA500F", "#E10500")

# brands wants icon.png at 256 and icon@2x.png at 512. logo.png is optional
# and falls back to the icon, so it is not generated: a logo is a wordmark,
# and the only honest wordmark here would be Mistral's own.
OUTPUTS = {"icon.png": 256, "icon@2x.png": 512}

OUTPUT_DIR = (
    Path(__file__).parent.parent
    / "brands"
    / "custom_integrations"
    / "mistral_conversation"
)


def scale(value: int) -> int:
    """Scale a design-grid coordinate up to the working canvas."""
    return round(value * WORK / CANVAS)


def build_mask() -> Image.Image:
    """Draw the bubble silhouette as an alpha mask on the working canvas."""
    mask = Image.new("L", (WORK, WORK), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        (0, 0, WORK - 1, scale(BODY_BOTTOM) - 1),
        radius=scale(CORNER_RADIUS),
        fill=255,
    )
    draw.polygon([(scale(x), scale(y)) for x, y in TAIL], fill=255)
    return mask


def build_colours() -> Image.Image:
    """Draw the colour bands across the whole working canvas."""
    colours = Image.new("RGB", (WORK, WORK))
    draw = ImageDraw.Draw(colours)

    for index, colour in enumerate(BANDS):
        top = scale(round(index * BODY_BOTTOM / len(BANDS)))
        # The last band runs to the foot of the canvas so the tail, which
        # hangs below the body, is a continuation of it rather than a gap.
        if index == len(BANDS) - 1:
            bottom = WORK
        else:
            bottom = scale(round((index + 1) * BODY_BOTTOM / len(BANDS)))
        draw.rectangle((0, top, WORK, bottom), fill=colour)

    return colours


def render(mask: Image.Image, colours: Image.Image, size: int) -> Image.Image:
    """Downsample the mask and the colours separately, then combine them."""
    # Resizing the two layers apart from each other keeps the transparent
    # background out of the arithmetic. Resizing a single RGBA image instead
    # would blend the colours towards black along every edge, because the
    # pixels outside the silhouette are transparent *black*.
    icon = colours.resize((size, size), Image.Resampling.LANCZOS).convert("RGBA")
    icon.putalpha(mask.resize((size, size), Image.Resampling.LANCZOS))
    return icon


def main() -> int:
    """Write the icon set, and report what was written."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mask = build_mask()
    colours = build_colours()

    for name, size in OUTPUTS.items():
        path = OUTPUT_DIR / name
        render(mask, colours, size).save(path, "PNG", optimize=True)
        print(f"✅ Wrote {path.relative_to(Path(__file__).parent.parent)} ({size}px)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
