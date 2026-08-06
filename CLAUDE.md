# CLAUDE.md

The conventions for this repository live in [`AGENTS.md`](AGENTS.md). Read it
before making changes — it is the single source of truth, and this file
deliberately does not restate it. Two documents describing the same rules
drift apart, and this repository has been bitten by that before.

`AGENTS.md` covers the project layout, the development workflow, `uv` usage,
code style, documentation standards and commit conventions.

## The short version

```bash
uv sync                              # dev tools plus the current Home Assistant
uv run pre-commit run --all-files    # linters, type checks, consistency checks
uv run pytest                        # tests
```

Both must pass before committing.

## Two things that catch people out

**`uv sync --all-groups` does not work here, by design.** `ha-current` and
`ha-minimum` pin different Home Assistant versions and are declared as
conflicting groups, so they cannot be installed together. A plain `uv sync`
gets `ha-current`. To work against the oldest supported version:

```bash
uv sync --no-default-groups --group dev --group ha-minimum
uv run --no-sync pytest
uv sync                              # switch back
```

**Verify third-party APIs against the installed package, not from memory.**
The integration was previously written against Mistral and Home Assistant
methods that never existed, which is why so much of it was rewritten. If you
are unsure whether something exists, import it and check. `AGENTS.md` says the
same thing at greater length, and it is the rule most worth following here.

And existing is not working: two features shipped against types the SDK
accepts and the endpoint does not honour, so for anything that changes what is
sent to the API, the check is a real request rather than a signature. See "A
type signature is not a behaviour" in `AGENTS.md`.
