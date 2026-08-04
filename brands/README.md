# Brand assets

Home Assistant and HACS do not read an integration's icon from the
integration's own repository. They read it from
[home-assistant/brands](https://github.com/home-assistant/brands), keyed by
domain, and fall back to a blank tile when there is no entry. That is why this
integration currently shows no logo anywhere in the UI.

This directory holds the icon set, laid out exactly as `brands` expects it, so
submitting it is a copy rather than a reshuffle:

```text
brands/
└── custom_integrations/
    └── mistral_conversation/
        ├── icon.png      # 256x256
        └── icon@2x.png   # 512x512
```

The files are generated, not hand-drawn — see
[`scripts/generate_brand_assets.py`](../scripts/generate_brand_assets.py).

## Whose mark this is

It is Mistral AI's, and this project claims nothing beyond using it to
identify the service the integration talks to. This integration is not
officially associated with Mistral AI, as the README says. That is the same
footing every other third-party integration in `brands` stands on, including
the ones Home Assistant ships itself.

If Mistral AI would rather it were not used here, that is entirely their call
and the entry should be withdrawn on request — from `brands` as well as from
this directory, since removing it here alone changes nothing users can see.

## How it is drawn

The mark is flat-coloured pixel art, so
[`generate_brand_assets.py`](../scripts/generate_brand_assets.py) reproduces it
from a cell grid rather than resampling a source file. That renders exactly at
any size, keeps the design reviewable as a diff, and avoids carrying someone
else's artwork as an opaque binary.

The cost is that the grid is a transcription. If it disagrees with Mistral's
own artwork, the artwork is right and the grid is a bug — fix the grid and
regenerate rather than editing the PNGs.

## Submitting it

The `brands` repository is not in this project's remit, so this has to be done
by hand:

1. Fork [home-assistant/brands](https://github.com/home-assistant/brands).
2. Copy `custom_integrations/mistral_conversation/` from this directory into
   the fork, keeping the path.
3. Open a pull request against `home-assistant/brands`.

Before opening it, check the requirements in that repository's own
documentation rather than trusting this list — they are enforced in CI, and
they change. At the time of writing they are a square PNG at 256x256 with a
`@2x` variant at 512x512, on a transparent background, trimmed of surplus
padding.

The mark is wider than it is tall, so it cannot fill a square canvas. It fills
the width and is centred vertically: `icon.png` has an alpha bounding box of
`(0, 34, 256, 222)`, which is trimmed on the axis it can be trimmed on. If
`brands` rejects the remaining vertical margin, the fix is to crop to a square
and accept the mark being smaller, not to add matching side padding.

`logo.png` is optional and falls back to the icon, so none is provided. A logo
is a wordmark, and Mistral's wordmark is a larger borrow than the glyph for no
benefit here.

## Afterwards

Once the pull request merges, the icon appears in Home Assistant and HACS with
no change needed here, and `brands` can come out of the `ignore` list in the
HACS job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). Until
then that check has nothing to find and has to stay ignored — it queries the
upstream repository, not this directory.
