# Repository Remediation Plan

Status: **implemented**. Scope agreed with the maintainer on 2026-07-30;
phases 1-4 delivered on the same day. Kept as the record of what was wrong and
why the rebuild took the shape it did. Safe to delete once reviewed.

This document records what is currently broken, the decisions taken about how
to fix it, and the order of work. It is a working document — delete it once the
work lands.

## 1. Baseline

Measured on `claude/repo-cleanup-plan-ymfpxb` (identical to `main`) with
`homeassistant` 2026.2.1 and `mistralai` 2.1.3:

| Check | Result |
| --- | --- |
| `uv run pytest` | 10 failed, 11 passed |
| `uv run ruff format --check` | 2 files would be reformatted |
| `uv run ruff check` | 2 errors (unsorted imports, unused loop variable) |
| `uv run ty check` | 27 diagnostics in source, 49 including tests |
| `uv run pre-commit run --all-files` | 3 hooks fail: Format, Lint, Ty |

The remaining 14 pre-commit hooks pass, including the three local consistency
scripts, `bandit` and `pymarkdown`.

## 2. Root cause

`entity.py` was written against an API surface that does not exist. It is not a
case of drift against a newer version — the names were never real. Nothing in
the tool-calling or streaming paths can execute.

### 2.1 Non-existent Mistral SDK API

`mistralai` 2.x ships **no top-level `__init__.py`**. It is a namespace package
containing `mistralai.client`, `mistralai.azure.client`, `mistralai.gcp` and
`mistralai.extra`, so every `mistralai.<Name>` attribute access fails:

- `mistralai.APIError`, `mistralai.AuthenticationError` and
  `mistralai.RateLimitError` (`entity.py:322,325,331`) do not exist. Because
  they appear in `except` clauses, they raise `AttributeError` *while handling*
  an API error, masking the original failure.
- Real error types live in `mistralai.client.errors`: `SDKError`,
  `HTTPValidationError`, `ResponseValidationError`, `NoResponseError`,
  `ObservabilityError`.
- `client.chat.complete_stream_async` (`entity.py:370`) does not exist. The
  streaming method is `chat.stream_async`, returning
  `EventStreamAsync[CompletionEvent]`.

The one thing that *is* correct is `from mistralai.client import Mistral` —
that is the right import path for 2.x.

### 2.2 Non-existent Home Assistant API

- `conversation.ToolCall`, `conversation.ToolResult` and
  `chat_log.async_add_tool_result` do not exist. Tool calls are
  `llm.ToolInput`; tool execution is driven by iterating
  `chat_log.async_add_assistant_content(...)`, which is an async generator that
  runs each tool and yields `ToolResultContent`.
- `chat_log.llm_context` does not exist.
- `conversation.ConfigSubentry` does not exist — `ConfigSubentry` is in
  `homeassistant.config_entries`.
- `AssistantContent` is frozen, so `assistant_content.tool_calls = [...]`
  (`entity.py:255`) cannot work. Tool calls must be passed to the constructor.
- The hand-rolled tool-iteration loop is therefore entirely wrong and must be
  replaced with the reference pattern used by `ollama` and `openai_conversation`
  in HA core.

### 2.3 Tool schemas are silently wrong

`_format_tool` walks `tool.parameters.schema.items()` and hard-codes every
parameter to `{"type": "string"}`. It should call
`voluptuous_openapi.convert(tool.parameters, custom_serializer=...)`. As
written, no numeric, boolean or enum tool argument could ever be typed
correctly.

### 2.4 Every UI setting is ignored at runtime

This is a functional bug independent of the above, and would survive a naive
API fix:

- `ConfigFlow.async_step_user` collects only `api_key`, so `entry.data` is
  `{"api_key": ...}`.
- `OptionsFlow` writes model, temperature, max tokens, prompt and
  `llm_hass_api` to `entry.options`.
- `entity.py:195` and `entity.py:347` read `self.entry.data`;
  `conversation.py:64` reads `self.entry.data`.

So the model, temperature, token limit, system prompt and LLM API chosen in the
UI are all discarded. Every request runs with the hard-coded defaults, and the
integration can never control devices because `llm_hass_api` never reaches
`async_provide_llm_data`.

`MistralBaseLLMEntity.__init__` has the same problem: it reads the model from
`subentry.data` or falls back to `DEFAULT_MODEL`, never consulting
`entry.options`, so the device is always named `Mistral AI
(mistral-small-latest)`.

### 2.5 Structured output is faked

`_async_handle_chat_log` comments that "Mistral doesn't have direct structured
output" and injects a synthetic `structured_output` function tool. Mistral does
support it: `chat.parse_async(response_format=...)`, and `complete_async`
accepts a `response_format` argument directly.

## 3. Decisions

**Architecture** — adopt the modern **subentry** model, matching HA core's
`ollama` / `openai_conversation`: the parent entry holds the API key, and each
agent is a `ConfigSubentry` with its own settings.

**Tests** — adopt `pytest-homeassistant-custom-component` and rewrite the suite
against a real `hass` fixture.

**Features** — tool calling, reauth, streaming, AI Task platform and native
structured output are all in scope.

**Migration** — no `async_migrate_entry`. There are no git tags and no
releases, so the entry schema is free to change.

**Licence** — GPL-3.0. The `LICENSE` file was already correct; the README and
`pyproject.toml` have been corrected to match. Applied, see Phase 4.1.

**Vibe tooling** — dropped entirely. Applied, see Phase 4.5.

The subentry model is chosen partly because it dissolves 2.4 by construction:
settings live in `subentry.data`, and there is no `entry.data`/`entry.options`
split left to get wrong.

## 4. Target minimum Home Assistant version

The chosen feature set requires, at minimum:

- `chat_log.async_provide_llm_data` and `ConversationInput.as_llm_context`
- `ConfigSubentryFlow` and `async_get_supported_subentry_types`
- `Platform.AI_TASK` and the `ai_task` component

**Verified as 2025.7.0**, by probing real installs rather than reading release
notes. On 2025.6.0, `Platform.AI_TASK`, the `ai_task` module,
`async_provide_llm_data` and `as_llm_context` are all absent; on 2025.7.0 all
four are present. Subentry support is *not* the binding constraint — it exists
in 2025.6 — so dropping AI Task and reverting to the deprecated
`async_update_llm_data` would buy exactly one month for a real feature loss.
Nothing in the agreed feature set needs anything newer, so 2025.7.0 is both the
floor and the right choice.

Now set consistently in `hacs.json`, `pyproject.toml` and `README.md`, and
enforced by `scripts/check_ha_version_consistency.py`, which was extended to
cover the README and verified to fail on drift.

## 5. Work plan

Sequenced so each phase leaves the tree in a state where the checks that
already pass keep passing. Phases 1 and 2 are prerequisites for verifying
anything else.

### Phase 1 — Make the tooling honest (done)

1. Add `[tool.pytest.ini_options]` to `pyproject.toml`. There is none today, so
   `asyncio_mode` is unset and every test needs an explicit
   `@pytest.mark.asyncio`. Set `asyncio_mode = "auto"`, `testpaths`, and
   coverage settings (`pytest-cov` is a dev dependency but unconfigured).
2. Fix `[tool.ruff.lint.isort] known-first-party`, which currently reads
   `custom_components.mistral_stt` — the wrong domain, copied from another
   repository. Should be `custom_components.mistral_conversation`.
3. The `[tool.pymarkdownlnt]` block in `pyproject.toml` is **dead config**. The
   pre-commit hook invokes `pymarkdown scan` with no `--config`, and pymarkdown
   does not read `pyproject.toml` by default, so all nine "disabled" rules —
   including `MD013` line length — are in fact still enforced. Verified by
   adding a long line to a Markdown file and watching `MD013` fire. Either pass
   `--config pyproject.toml` in the hook args and keep the exclusions, or delete
   the block and accept the defaults. The current state is the worst of both:
   the exclusions look configured but do nothing.
4. Add `pytest-homeassistant-custom-component` to the dev group. Note that
   `ai_task` imports `camera` → `stream` → `numpy`, so the test environment
   needs HA's full test dependencies; a bare `homeassistant` install cannot
   import `homeassistant.components.ai_task` (verified).
5. Extend `scripts/` coverage: teach `check_ha_version_consistency.py` about
   the README's stated minimum version.
6. Add `scripts/README.md`. `AGENTS.md` mandates it — one section per script
   with a description and usage examples — and it does not exist.

### Phase 2 — Rebuild the integration (done)

Work against `ollama` and `openai_conversation` in HA core as the reference
implementations throughout.

1. **`const.py`** — add subentry-type constants (`conversation`,
   `ai_task_data`), default subentry titles, and recommended-model constants.
   Keep `DEFAULT_PROMPT`, but note core integrations rely on
   `llm.DEFAULT_INSTRUCTIONS_PROMPT` instead; prefer that as the default and
   treat `CONF_PROMPT` as a pure override. `API_BASE_URL` is unused — remove
   it.
2. **`__init__.py`** — replace `hass.data[DOMAIN][entry.entry_id]` with
   `entry.runtime_data` via a `type MistralConfigEntry = ConfigEntry[Mistral]`
   alias. Construct and validate the client once during
   `async_setup_entry`, raising `ConfigEntryNotReady` on transport failure and
   `ConfigEntryAuthFailed` on a bad key. Add `Platform.AI_TASK` to `PLATFORMS`.
   Replace the hand-rolled `async_reload_entry` (which pops `hass.data` and
   re-enters setup) with `hass.config_entries.async_reload`.
3. **`config_flow.py`** —
   - Delete the duplicated `@staticmethod` on lines 194-195.
   - Replace `OptionsFlow` with a `ConfigSubentryFlow` exposed through
     `async_get_supported_subentry_types`, handling both `conversation` and
     `ai_task_data`, with `async_step_reconfigure` aliased to the options step.
   - Call `async_set_unique_id` / `_abort_if_unique_id_configured` so the
     already-translated `already_configured` abort reason is reachable.
   - Add the reauth flow. `entity.py:327` already calls
     `entry.async_start_reauth`, which currently lands nowhere.
   - Fix `validate_input`, which raises `InvalidAuth` for temperature and
     token-range problems. Those are neither auth nor connection failures;
     range validation belongs in the schema, not in a hand-written checker.
   - Delete the module-level `OPTIONS_SCHEMA`, which is unused and declares
     empty `options=[]` selector lists.
   - Give `get_available_models` a real home; the dropdown should be populated
     from it rather than from `[DEFAULT_MODEL]`.
4. **`entity.py`** — the bulk of the work.
   - Import errors from `mistralai.client.errors` and map `SDKError` status
     codes to the right Home Assistant exceptions (401 → reauth, 429 → rate
     limit, else generic).
   - Rewrite `_format_tool` on `voluptuous_openapi.convert`.
   - Replace the invented tool loop with the reference pattern: iterate
     `chat_log.async_add_assistant_content(...)`, re-derive messages from
     `chat_log.content` each pass, and break on
     `not chat_log.unresponded_tool_results`.
   - Rewrite streaming on `chat.stream_async`, and fix `_transform_stream` —
     the current version speculatively probes for both a `choices` and a `data`
     attribute, and drops tool calls entirely in the `data` branch. Stream
     deltas carry partial JSON argument strings, which must be accumulated per
     tool-call index and parsed once complete; the current code passes
     `tool_call.function.arguments or {}` straight through as if it were
     already a dict.
   - Fix the `ARG`/`B007` lint error at line 373 (`delta_content` is assigned
     but the loop body is `pass`).
   - Take `ConfigSubentry` from `homeassistant.config_entries`, and read
     settings from `subentry.data`.
   - Implement structured output through `response_format` rather than the
     synthetic tool.
5. **`conversation.py`** —
   - Iterate `config_entry.subentries` in `async_setup_entry`, passing
     `config_subentry_id`, and switch to `AddConfigEntryEntitiesCallback`.
   - Remove `self.hass = hass`; `hass` is injected by the platform.
   - Remove the `_attr_name` override. With `_attr_has_entity_name = True` the
     name should stay `None` so the entity inherits the device name.
   - Set `_attr_supported_features = ConversationEntityFeature.CONTROL` when
     `llm_hass_api` is configured, otherwise Home Assistant will not offer the
     agent as a controlling agent.
   - Drop the `if self._attr_supports_streaming` branch and always stream; the
     class-level constant makes the `else` unreachable.
6. **`ai_task.py`** — new file. `MistralTaskEntity(ai_task.AITaskEntity,
   MistralBaseLLMEntity)` over the `ai_task_data` subentry type, supporting
   `GenDataTaskResult` and reusing the shared chat-log handling.
7. **`manifest.json`** — `dependencies` is `[]`; it must declare
   `["conversation", "ai_task"]`, plus `after_dependencies: ["assist_pipeline",
   "intent"]` as the reference integrations do. Drop `aiohttp` from
   `requirements` (it is a Home Assistant core dependency) and pin `mistralai`
   exactly rather than `>=2.0.0`. Verify
   `scripts/check_manifest_consistency.py` still agrees afterwards.
8. **`translations/en.json`** — remove the `services.process` block, which
   describes a service this integration does not register (`conversation.process`
   belongs to core). Add strings for the new subentry and reauth flows.
   Consider adding `icons.json`.

### Phase 3 — Rewrite the tests (done)

The existing suite fails for a structural reason: `MagicMock(spec=HomeAssistant)`
has no `bus`, so the moment real HA code runs — `async_get_chat_session`
registering an `EVENT_HOMEASSISTANT_STOP` listener — it raises
`AttributeError: Mock object has no attribute 'bus'`. Eight of the ten failures
are that exact error. No amount of mock-patching makes this suite meaningful.

1. Replace `tests/conftest.py`, which currently only does `sys.path`
   manipulation, with real fixtures: `enable_custom_integrations`, a
   `MockConfigEntry` with subentries, and a mocked Mistral client.
2. Rewrite `tests/test_config_flow.py`. It currently instantiates
   `ConfigFlow()` directly and asserts `errors["base"] == "unknown"` in the test
   *named* `test_config_flow_user_step_errors` for both the `CannotConnect` and
   `InvalidAuth` cases — it raises bare `Exception` in both, so it asserts the
   fallback path twice and never exercises either real branch. Drive flows
   through `hass.config_entries.flow` instead, and cover the subentry and reauth
   flows.
3. Rewrite `tests/test_conversation.py` against a real chat log. Note
   `test_supported_languages` asserts `languages == [conversation.MATCH_ALL]`
   while the property correctly returns the bare string `MATCH_ALL` — the test
   is wrong, not the code. Delete `test_options_flow_init`, which is an empty
   `pass` body under a docstring saying it is skipped.
4. Add `tests/test_ai_task.py` and `tests/test_init.py` (setup, unload, reauth
   trigger, `ConfigEntryNotReady`).
5. Target meaningful coverage of tool calling, streaming and error mapping —
   the three areas that were entirely untested and entirely broken.

### Phase 4 — Documentation and repository hygiene (done)

1. ~~**Licence conflict.**~~ **Done.** `README.md` claimed MIT while `LICENSE`
   is the 674-line GNU GPL v3. Resolved in favour of GPL-3.0: the README now
   says GPL v3, and `pyproject.toml` (which declared no `license` field at all)
   now carries `license = "GPL-3.0-only"` with `license-files = ["LICENSE"]`.
   `LICENSE` itself was already correct and is unchanged.
2. **README corrections.** The minimum HA version (see section 4); the manual
   installation file tree, which lists `client.py` and `strings.json` (neither
   exists) and omits `entity.py` and `translations/`; the configuration section,
   which describes an options flow that will no longer exist after Phase 2;
   `open-mistral-7b` and `open-mistral-nemo` in the model list, which should be
   checked against what the API currently returns.
3. **CI.** `.github/workflows/ci.yml` runs `uv sync --dev` — deprecated in
   favour of `--all-groups`, and it silently skips the dev group's newer
   spelling. Add the two validation jobs every HACS integration is expected to
   have and this repo lacks: `home-assistant/actions/hassfest` and
   `hacs/action`. Add a release workflow that packages
   `custom_components/mistral_conversation` as a zip on tag, since `README.md`
   links to a releases page that has no releases and the repository has no tags.
4. **Repository files.** Add `.github/dependabot.yml`; Dependabot PRs have been
   merged (`#2`-`#10`) but no configuration file is committed on any branch, so
   the behaviour is unpinned. Add `CODEOWNERS` to match the `codeowners` field
   already in `manifest.json`, plus issue templates and `CONTRIBUTING.md`.
5. ~~**Leftover tooling.**~~ **Done.** `mistral-vibe` has been dropped, along
   with the committed `.vibe/skills/get-api-docs/` skill, the four `.gitignore`
   negation rules that partially committed it, the `npm install -g @aisuite/chub`
   step in `.devcontainer.json`, and the Node 24 devcontainer feature that
   existed only to serve it. This removed `litellm` and roughly 60 other
   transitive packages from `uv.lock`. The skill's one durable lesson — verify
   third-party APIs against the installed package instead of recalling them —
   was rewritten into `AGENTS.md` in tool-agnostic form, since section 2 is
   precisely what happens when that is not done.
6. **`.devcontainer.json`** — the `astral-sh.type` extension ID looks incorrect
   (the Astral type-checker extension is `astral-sh.ty`). Verify.

## 6. Definition of done

- `uv run pre-commit run --all-files` clean, all 17 hooks.
- `uv run pytest` green, with tool calling, streaming, structured output and
  error mapping all covered.
- `uv run ty check` reports zero diagnostics in `custom_components/` and
  `scripts/`.
- hassfest and HACS validation pass in CI.
- The minimum Home Assistant version is stated identically in `hacs.json`,
  `pyproject.toml` and `README.md`, and enforced by a pre-commit hook.
- ~~The licence is stated consistently in `LICENSE`, `README.md` and
  `pyproject.toml`.~~ Done.
- The README describes files and flows that actually exist.

## 7. Deliberately out of scope

- Image generation via AI Task. Mistral's SDK exposes it, but no maintainer
  request exists.
- Migration of existing config entries — see section 3.
- Relicensing. GPL-3.0 is confirmed as the intended licence; only the README and
  `pyproject.toml` were corrected to match the existing `LICENSE`.
