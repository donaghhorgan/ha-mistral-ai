# AGENTS.md

## Project Structure

This project is a Home Assistant custom integration for Mistral AI
conversation services. The structure follows standard Home Assistant
integration conventions:

```bash
custom_components/mistral_conversation/
├── __init__.py     # Integration setup and entry point
├── ai_task.py      # AI Task platform
├── config_flow.py  # Config and subentry flow handlers
├── const.py        # Application constants
├── conversation.py # Conversation platform
├── entity.py       # Shared LLM entity and Mistral API handling
├── manifest.json   # Integration metadata
└── translations/   # Language files
hacs.json           # Home Assistant Community Store (HACS) configuration
scripts/            # Helper scripts for development
tests/              # Unit tests
```

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
