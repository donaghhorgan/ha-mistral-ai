# AGENTS.md

## Project Structure

This project is a Home Assistant custom integration for Mistral AI
conversation services. The structure follows standard Home Assistant
integration conventions:

```bash
custom_components/mistral_ai/
├── __init__.py     # Integration setup and entry point
├── ai_task.py      # AI Task platform
├── client.py       # Mistral client construction
├── config_flow.py  # Config and subentry flow handlers
├── const.py        # Application constants
├── conversation.py # Conversation platform
├── entity.py       # Shared LLM entity and Mistral API handling
├── helpers.py      # Shared API helpers used by more than one platform
├── manifest.json   # Integration metadata
├── stt.py          # Speech-to-text platform
├── tts.py          # Text-to-speech platform
└── translations/   # Language files
brands/             # Icon set staged for home-assistant/brands
hacs.json           # Home Assistant Community Store (HACS) configuration
scripts/            # Helper scripts for development
tests/              # Unit tests
```

`brands/` is not read by Home Assistant. Icons live in
[home-assistant/brands](https://github.com/home-assistant/brands), and that
directory mirrors the layout that repository expects so the submission is a
copy; see [`brands/README.md`](./brands/README.md).

## Development Workflow

1. Create a plan
2. Make code changes
3. Ensure precommit checks pass: `uv run pre-commit run`
4. Ensure unit tests pass: `uv run pytest`
5. Commit to git and push

### Python Package Management

- Use `uv` for Python package management
- Use groups to separate Python dependencies:
  - For production dependencies: `uv add package-name`
  - For development dependencies: `uv add --dev package-name`
- Use `uv run` to run Python commands and tools
- Be aware that `uv` stores its virtual environment in [`.venv`](./.venv). If
  you are grepping, you should consider whether to exclude `.venv` to speed up
  your search, e.g., if you are searching for info from project files rather
  than Python dependencies.

#### Home Assistant pins almost everything

`pytest-homeassistant-custom-component` pins Home Assistant to an exact
version, and Home Assistant pins its own dependencies exactly — 46 of them
directly in 2026.2.3, plus whatever those pin beneath. So most of the
resolution is frozen, and a package under that subtree cannot be upgraded on
its own. Attempting it does not fail a check, it fails resolution:

```text
Because homeassistant==2025.8.0 depends on voluptuous==0.15.2 and
voluptuous==0.16.0, we can conclude that homeassistant==2025.8.0 cannot be
used.
```

Before adding or raising any dependency, check whether Home Assistant already
pins it. Do not assume from the name, and do not assume that declaring it in
`pyproject.toml` makes it ours to choose — `voluptuous` is declared in
`[project] dependencies` and is still governed entirely by Home Assistant's
pin:

```bash
uv run --no-sync python -c "
from importlib.metadata import requires
print([r for r in requires('homeassistant') if 'PACKAGE' in r.lower()])"
```

If it comes back with `==`, the version is not yours to pick. Match it, or use
a floor at or below it. This has bitten twice: `pillow>=12.0.0` made
`ha-minimum` unresolvable because Home Assistant 2025.8.0 wants `11.3.0`.

#### Dependabot works from an allowlist

[`.github/dependabot.yml`](./.github/dependabot.yml) names the packages
Dependabot may update, rather than listing the ones it may not — the blocklist
could never be complete, for the reason above. **Add a new dev tool to that
allowlist, or it will silently never be updated.**

Do not add a package to the allowlist without checking it is not one Home
Assistant pins. Allowing one puts back exactly the failure the allowlist
exists to prevent, and that failure is invisible from GitHub: no pull request
is opened, every uv update stops, and the only signal is the Dependabot run
page.

### Code Style and Coding Conventions

- Python code should be written for the version in [`.python-version`](.python-version)
- The integration is tested against three Home Assistant versions, because no
  single environment can cover the supported range:
  - `test (ha-current)` — whatever `uv.lock` resolves, on Python 3.13
  - `test (ha-minimum)` — the floor named in `hacs.json`, on Python 3.13
  - `test-latest` — the newest release, on Python 3.14, since Home Assistant
    2026.5.0 and later require it. Advisory only; it tracks a moving target.

  The first two are conflicting dependency groups in `pyproject.toml`, so uv
  locks a resolution for each and both are reproducible. Switch between them
  with `uv sync --no-default-groups --group dev --group <group>`; a plain
  `uv sync` gets `ha-current`. Note that `uv sync --all-groups` cannot work
  here, because the groups conflict by design.

  `test-latest` cannot be a group: it needs Python 3.14, which `uv.lock`
  cannot resolve under `requires-python = ">=3.13.2"`. Marker-gating it makes
  the lock succeed but silently resolves to nothing on 3.13, which would give
  a job that passes while testing the wrong version. It stays an isolated
  `uv run --with` install for that reason.

  The project itself stays on Python 3.13 so that the floor remains
  reachable. Do not raise `requires-python` to 3.14 without also raising the
  minimum Home Assistant version to one that requires it.
- Write unit tests for new functionality
- Consult the [Home Assistant Developer
  Docs](https://developers.home-assistant.io/) when making changes to Home
  Assistant integration code to ensure that best practices are followed.
- Use the [search](https://developers.home-assistant.io/search/?q=query)
  function to search for relevant content.
- Verify third-party APIs against the installed package rather than recalling
  them, e.g. `uv run python -c "import inspect, mistralai.client as m; ..."`.
  Do not assume a method or exception name exists because it looks plausible.
- Fix linting errors:
  - Markdown: `uv run pymarkdown fix file.md`
  - Python: `uv run ruff check --fix`
  - TOML: `uv run toml-sort --in-place file.toml`

### Documentation Standards

- In general, write concise documentation
- For Python, use docstrings
- For utility scripts, use [`scripts/README.md`](./scripts/README.md). Each
  script should have a section with a description and some usage examples.
- For the overall project, use [`README.md`](./README.md)

### Source Code Management

- Write concise but descriptive commit messages

#### Branching

This project is trunk-based. `main` is the only long-lived branch, and it is
always releasable.

- Branch off `main`, and only off `main`. There is no `dev` or `release`
  branch to integrate through, and adding one would reintroduce exactly the
  drift this avoids.
- Keep branches short-lived — hours or days, not weeks. A branch that lives
  long enough to need `main` merged into it twice was too big to begin with.
- Prefer several small pull requests to one large one. Anything already
  correct and reviewable should not wait behind the rest of the work.
- Merge into `main` through a pull request with CI green. Nothing is pushed
  to `main` directly.
- Never build on a branch whose pull request is already merged. Start again
  from `main`; a merged pull request cannot track new work.
- Delete the branch once it is merged.

CI runs on pull requests targeting `main` and on pushes to `main`. It does not
run on pushes to a topic branch, so opening the pull request is what starts
the checks — open it early, in draft if it is not ready.

`main` is additionally protected by a branch ruleset in the repository
settings, which is not version controlled and therefore not visible here. It
requires a pull request and passing checks, and blocks force pushes and
deletion. The required checks are the blocking jobs in
[`ci.yml`](./.github/workflows/ci.yml): `lint`, `Test (ha-current)`,
`Test (ha-minimum)`, `Validate with hassfest` and `Validate with HACS`.

`Test against latest Home Assistant` is deliberately not among them. It is
`continue-on-error` because it tracks a moving upstream target, so requiring
it would let an unrelated Home Assistant release block every merge in the
repository.

The ruleset is also strict: a branch must be up to date with `main` before it
can merge. With short-lived branches that is rarely more than a fast-forward.

#### Merging

Pull requests are merged with a merge commit. Not squashed, not rebased.

- Commit messages here carry the reasoning, not just a label. Squashing
  concatenates or discards them, and moves the record into the pull request
  body, which is not in the repository.
- Pull requests sometimes carry commits from more than one author. Squashing
  collapses them to one, demoting the rest to a trailer at best.
- Commit hashes stay stable, and the documentation cites one:
  [`brands/README.md`](./brands/README.md) points at the artwork it was
  generated from by hash. A squash would rewrite it and leave the citation
  dangling in every clone.

The cost, accepted knowingly: `main`'s history interleaves rather than reading
as one commit per change, and `git bisect` can land on a commit that never
passed CI on its own. Keeping pull requests small is what holds that in check,
which the branching rules above already ask for.

This means "Require linear history" must stay off in the ruleset — it forbids
merge commits outright.
