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

## What the mark is, and what it is not

It is a speech bubble filled with a warm yellow-to-red ramp. It is **not**
Mistral AI's logo. This integration is not officially associated with Mistral
AI, and shipping their artwork under that banner is not ours to do. The palette
places it next to the service it talks to; the silhouette says what it is, a
conversation agent.

If Mistral AI would prefer their own mark used here, or would prefer this one
not used, that is their call to make and the assets should be replaced or
withdrawn accordingly.

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
`@2x` variant at 512x512, transparent background, and no transparent padding
around the mark. `icon.png` here satisfies all three: its alpha bounding box is
the full 256x256 canvas.

`logo.png` is optional and falls back to the icon, so none is provided. A logo
is a wordmark, and the only honest wordmark for this integration would be
Mistral's own.

## Afterwards

Once the pull request merges, the icon appears in Home Assistant and HACS with
no change needed here, and `brands` can come out of the `ignore` list in the
HACS job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). Until
then that check has nothing to find and has to stay ignored — it queries the
upstream repository, not this directory.
