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

**The oldest supported Home Assistant is not a dependency group.** There is
one resolution in `uv.lock`, and `uv sync` gets it. The floor is tested by the
`Test (ha-minimum)` job, which resolves it on the fly so that a year-old set
of pins never enters this project's dependency graph -- carrying them here
made the project unresolvable for Dependabot and raised security alerts on
versions frozen by design.

To run against the floor locally, use the same command that job does. Its pins
live in the `env:` block of [`ci.yml`](.github/workflows/ci.yml), which is the
only place they are written down:

```bash
uv run --isolated --no-project \
  --with pytest-homeassistant-custom-component==0.13.269 \
  --with hassil==2.2.3 --with home-assistant-intents==2025.7.30 \
  --with pycares==4.9.0 --with ha-ffmpeg --with mutagen \
  --with pymicro-vad --with pyspeex-noise \
  --with "mistralai>=2.1.0" --with PyTurboJPEG \
  pytest tests/ -q --no-cov
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
