# Development scripts

Mostly consistency checks that guard against configuration drift between the
files that all have to agree about the same thing. Each check runs as a `local`
pre-commit hook (see [`.pre-commit-config.yaml`](../.pre-commit-config.yaml))
and is only triggered when a file it cares about changes.

All of them exit `0` on success and `1` with an explanation on failure, so they
can be run directly as well as through pre-commit.

The exception is `generate_brand_assets.py`, which is a one-off generator
rather than a check and is not wired into pre-commit.

## `check_ha_version_consistency.py`

Checks that the minimum Home Assistant version agrees across three files:

- `hacs.json` — the `homeassistant` key, which is what HACS enforces at
  install time
- `pyproject.toml` — the `homeassistant` entry in `[project] dependencies`
- `README.md` — the `Home Assistant X.Y.Z or newer` line under Requirements

`pyproject.toml` only has to be *compatible* (its `>=` floor must not exceed
the HACS version), but the README must state the HACS version *exactly*,
because it is what users read before installing.

```bash
# Run directly
uv run python scripts/check_ha_version_consistency.py

# Run via pre-commit
uv run pre-commit run ha-version-consistency --all-files
```

Example failure:

```text
❌ README states Home Assistant 2023.5.0 but hacs.json advertises 2025.8.0
```

## `check_intent_pin_consistency.py`

Checks that the `hassil` and `home-assistant-intents` pins in `pyproject.toml`
match what the *installed* Home Assistant declares in the conversation
component's manifest.

Home Assistant pins both to exact versions, and the pins differ between
releases, so they are pinned per Home Assistant dependency group rather than
shared. Nothing else keeps the two in step: bumping
`pytest-homeassistant-custom-component` changes the Home Assistant version
without touching the pins next to it, and a mismatch can break the build
outright — `hassil` 3.11 dropped `hassil.fuzzy`, which Home Assistant 2026.2
imports.

Only the group matching the currently synced Home Assistant is checked; the
others describe a version that is not installed. Run it under both to cover
both.

```bash
uv run python scripts/check_intent_pin_consistency.py
uv run pre-commit run intent-pin-consistency --all-files

# And against the floor
uv sync --no-default-groups --group dev --group ha-minimum
uv run --no-sync python scripts/check_intent_pin_consistency.py
uv sync
```

Example failure:

```text
❌ No dependency group matches Home Assistant 2026.2.3
```

## `check_manifest_consistency.py`

Checks that `custom_components/mistral_ai/manifest.json` agrees with
`pyproject.toml` — principally the integration version against the project
version, so a release cannot ship a manifest that disagrees with the package
metadata.

```bash
uv run python scripts/check_manifest_consistency.py
uv run pre-commit run sync-manifest --all-files
```

## `check_python_version_consistency.py`

Checks that the Python version agrees across `.python-version`,
`pyproject.toml` (`requires-python`), `.devcontainer.json` and the CI workflow
matrix in `.github/workflows/`, so local development, the devcontainer and CI
cannot silently diverge.

CI is held to a looser rule than the rest: at least one job must run on the
declared version, and jobs may additionally run on *newer* ones. That allows
the `test-latest` job to exercise recent Home Assistant releases, which
require Python 3.14, without the project itself moving off 3.13. A CI job on
an *older* Python than `requires-python` is still an error.

```bash
uv run python scripts/check_python_version_consistency.py
uv run pre-commit run python-version-consistency --all-files
```

## `generate_brand_assets.py`

Draws the icon set that
[home-assistant/brands](https://github.com/home-assistant/brands) expects, into
[`brands/custom_integrations/mistral_ai/`](../brands). Home Assistant
and HACS take an integration's icon from that repository rather than from this
one, so without an entry there the integration shows a blank tile.

The mark is Mistral AI's, drawn from a cell grid rather than resampled from a
source file, so it renders exactly at any size and stays reviewable as a diff.
Not a pre-commit hook: the output is committed, and it only needs rerunning if
the grid changes. See [`brands/README.md`](../brands/README.md) for the
standing this project claims to the mark and how to submit it.

```bash
uv run python scripts/generate_brand_assets.py
```
