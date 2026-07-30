# AGENTS.md

## Project Structure

This project is a Home Assistant custom integration for Mistral AI
conversation services. The structure follows standard Home Assistant
integration conventions:

```bash
custom_components/mistral_conversation/
├── __init__.py     # Integration setup and entry point
├── config_flow.py  # Configuration flow handler
├── const.py        # Application constants
├── manifest.json   # Integration metadata
├── translations/   # Language files
└── ...
hacs.json           # Home Assistant Communicty Store (HACS) configuration
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
