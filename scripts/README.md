# Development scripts

Consistency checks that guard against configuration drift between the files
that all have to agree about the same thing. Each one runs as a `local`
pre-commit hook (see [`.pre-commit-config.yaml`](../.pre-commit-config.yaml))
and is only triggered when a file it cares about changes.

All of them exit `0` on success and `1` with an explanation on failure, so they
can be run directly as well as through pre-commit.

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

## `check_manifest_consistency.py`

Checks that `custom_components/mistral_conversation/manifest.json` agrees with
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
