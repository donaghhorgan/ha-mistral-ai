# Development scripts

Mostly consistency checks that guard against configuration drift between the
files that all have to agree about the same thing. Each check runs as a `local`
pre-commit hook (see [`.pre-commit-config.yaml`](../.pre-commit-config.yaml))
and is only triggered when a file it cares about changes.

All of them exit `0` on success and `1` with an explanation on failure, so they
can be run directly as well as through pre-commit.

The exceptions are `generate_brand_assets.py`, a one-off generator, and
`measure_tts_chunking.py`, a measurement harness. Neither is a check, and
neither is wired into pre-commit.

## `check_dependabot_coverage.py`

Checks that every package declared in `pyproject.toml` is a decision somebody
has made: allowed in `.github/dependabot.yml`, ignored there, or listed in the
script's own `EXPECTED_ABSENT` with the reason it needs no entry.

This exists because an allowlist fails quietly. A package added to a
dependency group and not added to the allowlist is simply never updated — no
failed check, no warning, just a floor drifting behind. That is how
`ha-ffmpeg` and `mutagen` went a year without updates, with `mutagen` a
release behind its installed version before anyone noticed.

It deliberately does not judge whether a package *should* move, only whether
somebody decided. The three ways to satisfy it map onto the three real
answers: free to move, resolves-then-breaks, or not ours to touch.

```bash
# Run directly -- stdlib only, so no environment needed
python3 scripts/check_dependabot_coverage.py
uv run pre-commit run dependabot-coverage --all-files
```

Example failure:

```text
❌ No Dependabot decision for:

  ha-ffmpeg
  mutagen
```

A package configured but no longer declared is reported as a warning rather
than an error, so that removing a dependency does not block the commit that
removes it.

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
releases, so they are pinned beside each Home Assistant version rather than
shared: `ha-current` in `pyproject.toml`, and the floor in the `env:` block of
`ci.yml`. Both are read. Nothing else keeps the two in step: bumping
`pytest-homeassistant-custom-component` changes the Home Assistant version
without touching the pins next to it, and a mismatch can break the build
outright — `hassil` 3.11 dropped `hassil.fuzzy`, which Home Assistant 2026.2
imports.

Only the source matching the installed Home Assistant is checked; the other
describes a version that is not present. Run it under both to cover both.

```bash
uv run python scripts/check_intent_pin_consistency.py
uv run pre-commit run intent-pin-consistency --all-files

# And against the floor, whose pins live in ci.yml rather than pyproject.toml
uv run --isolated --no-project \
  --with pytest-homeassistant-custom-component==0.13.269 \
  --with hassil==2.2.3 --with home-assistant-intents==2025.7.30 \
  --with pycares==4.9.0 --with ha-ffmpeg --with mutagen \
  --with pymicro-vad --with pyspeex-noise \
  python scripts/check_intent_pin_consistency.py
```

Example failure:

```text
❌ Nothing matches Home Assistant 2026.2.3
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

## `measure_tts_chunking.py`

Compares the ways `tts.py` could carve a reply into speech requests: one
request for the whole thing, the shipped sentence split, that split with the
abbreviation lookbehind or the minimum length taken out, and coarser groupings
up to paragraphs only.

It exists because the shipped split was chosen for being the obvious shape and
never measured ([#160]). Every boundary is a cost — a billed request, and a
chunk the speech model has to read without the context of what came before —
paid for one benefit, which is the first audio arriving early. The script
answers how many boundaries that is worth.

Two halves, and they are not equally trustworthy.

`--offline` needs no key and makes no requests. It reports how many requests
each strategy issues, where the boundaries land, and which of them the
lookbehind and the minimum length actually move. Deterministic: one run is the
whole answer.

A live run additionally times the thing the split is bought for — time to
first audio byte, wall clock to the last byte, and how much audio came back —
at one billed request per chunk per trial. Every figure is a median over
`--trials` runs with the observed spread printed beside it, because network
and server load are not controllable from here and a difference inside that
spread is not a finding.

```bash
# Deterministic half only. No key, no requests, no cost.
uv run python scripts/measure_tts_chunking.py --offline

# The real thing. Needs MISTRAL_API_KEY and a voice id from the account.
uv run python scripts/measure_tts_chunking.py --voice VOICE_ID

# One case, more trials, for a difference that looks marginal.
uv run python scripts/measure_tts_chunking.py --voice VOICE_ID \
  --case long --trials 9
```

Three habits are built in rather than left to whoever runs it, because this
project has twice published a number that was wrong for want of them. The
clock restarts immediately before the first request of each case and strategy,
so no result inherits another's baseline. Requests go out with `stream: true`
and the audio is read from the server-sent events, which is the path `tts.py`
takes — timing the buffered endpoint instead would differ by roughly the whole
effect being measured. And the cell order rotates every trial, so a network
that slows during the run does not charge that to whichever strategy always
went last.

It also refuses to run if its own copy of the splitting rule has drifted from
`_sentences` in `tts.py`, since a plausible-looking measurement of code that
does not ship is worse than no measurement.

**The live half has never run.** It was written against a rotated key and
every request it made came back `401`, so only `--offline` has ever produced
output. Expect to fix something the first time it reaches the API, and treat
its first numbers accordingly.

[#160]: https://github.com/donaghhorgan/ha-mistral-ai/issues/160
