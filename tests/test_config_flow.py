"""Tests for the Mistral AI config and subentry flows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import voluptuous as vol
from homeassistant.config_entries import SOURCE_USER, ConfigEntryState
from homeassistant.const import CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.mistral_ai as integration
from custom_components.mistral_ai.const import (
    CONF_API_KEY,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_VOICE,
    DEFAULT_MODEL,
    DEFAULT_STT_TEMPERATURE,
    DOMAIN,
    REASONING_EFFORTS,
    SUBENTRY_TYPE_AI_TASK_DATA,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_STT,
    SUBENTRY_TYPE_TTS,
)

from .conftest import (
    DEPRECATED_MODEL,
    NON_CHAT_MODEL,
    ORPHANED_MODEL,
    STT_MODEL,
    TTS_MODEL,
    VOICE_ID,
)
from .helpers import make_sdk_error


def _model_values(result: dict) -> list[str]:
    """Return the model IDs a form offers, ignoring their labels.

    The dropdown carries labels as well as values now, because a deprecated
    model is shown with its retirement date rather than looking like any
    other. What these tests care about is which models are offered, so the
    label is stripped here rather than in every assertion.
    """
    return [
        option["value"]
        for option in result["data_schema"].schema[CONF_MODEL].config["options"]
    ]


async def test_user_flow_creates_entry_and_subentry(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    """A valid key creates the entry plus a default conversation agent."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "test-api-key"}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_API_KEY: "test-api-key"}

    subentries = list(result["result"].subentries.values())
    assert len(subentries) == 1
    assert subentries[0].subentry_type == SUBENTRY_TYPE_CONVERSATION


@pytest.mark.parametrize(
    ("side_effect", "expected_error"),
    [
        (make_sdk_error(401), "invalid_auth"),
        (make_sdk_error(403), "forbidden"),
        (make_sdk_error(500), "cannot_connect"),
        (httpx.ConnectError("no route to host"), "cannot_connect"),
        (TimeoutError(), "cannot_connect"),
        (ValueError("something odd"), "unknown"),
    ],
)
async def test_user_flow_errors(
    hass: HomeAssistant,
    mock_client: MagicMock,
    side_effect: Exception,
    expected_error: str,
) -> None:
    """Each failure mode maps onto its own distinct form error."""
    mock_client.models.list_async = AsyncMock(side_effect=side_effect)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "bad-key"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": expected_error}


async def test_user_flow_recovers_after_error(
    hass: HomeAssistant, mock_client: MagicMock, mock_models_response: MagicMock
) -> None:
    """A corrected key after a failure still creates the entry."""
    mock_client.models.list_async = AsyncMock(side_effect=make_sdk_error(401))

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "bad-key"}
    )
    assert result["errors"] == {"base": "invalid_auth"}

    mock_client.models.list_async = AsyncMock(return_value=mock_models_response)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "good-key"}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_API_KEY: "good-key"}


async def test_reauth_flow(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Reauth replaces the stored API key."""
    result = await init_integration.start_reauth_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "new-api-key"}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert init_integration.data[CONF_API_KEY] == "new-api-key"


async def test_reauth_flow_rejects_bad_key(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Reauth keeps the old key when the new one is also rejected."""
    mock_client.models.list_async = AsyncMock(side_effect=make_sdk_error(401))

    result = await init_integration.start_reauth_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "still-bad"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "invalid_auth"}
    assert init_integration.data[CONF_API_KEY] == "test-api-key"


async def test_reauth_flow_reports_a_refusal_separately(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A 403 during reauth says so, rather than blaming the key again.

    The worst place for the old 401/403 conflation: the user is already in a
    dialog that exists because their key was said to be bad. Telling them the
    replacement is bad too, when the account is what is refusing, is a loop
    with no exit.
    """
    mock_client.models.list_async = AsyncMock(side_effect=make_sdk_error(403))

    result = await init_integration.start_reauth_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "a-perfectly-good-key"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "forbidden"}
    assert init_integration.data[CONF_API_KEY] == "test-api-key"


async def test_reconfigure_flow_replaces_the_key(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The key can be rotated deliberately, with everything under it kept.

    Before this step existed the only route to the field was the reauth
    dialog, which opens on a 401 -- so a planned rotation meant revoking the
    working key upstream first, or deleting the entry and taking every
    subentry with it.
    """
    before = set(init_integration.subentries)

    result = await init_integration.start_reconfigure_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "rotated-api-key"}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert init_integration.data[CONF_API_KEY] == "rotated-api-key"

    # The point of the whole thing: the agents, tasks and speech entities
    # underneath survive, which is what deleting the entry cost.
    assert set(init_integration.subentries) == before
    assert init_integration.state is ConfigEntryState.LOADED


@pytest.mark.parametrize(
    ("side_effect", "expected_error"),
    [
        (make_sdk_error(401), "invalid_auth"),
        (make_sdk_error(403), "forbidden"),
        (make_sdk_error(500), "cannot_connect"),
        (httpx.ConnectError("no route to host"), "cannot_connect"),
        (TimeoutError(), "cannot_connect"),
        (ValueError("something odd"), "unknown"),
    ],
)
async def test_reconfigure_flow_errors(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_client: MagicMock,
    side_effect: Exception,
    expected_error: str,
) -> None:
    """A key that does not work is refused, and the stored one is left alone.

    The same ladder the user step renders, because the remedies differ: a 403
    is a valid key with no account access, and telling someone their key is
    invalid sends them to generate another one that will fail the same way.
    That case is why this step exists at all -- a 403 never triggers reauth,
    so there was previously no way to try a different key.
    """
    mock_client.models.list_async = AsyncMock(side_effect=side_effect)

    result = await init_integration.start_reconfigure_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "no-good"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"
    assert result["errors"] == {"base": expected_error}
    assert init_integration.data[CONF_API_KEY] == "test-api-key"


async def test_reconfigure_flow_recovers_after_error(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_client: MagicMock,
    mock_models_response: MagicMock,
) -> None:
    """A working key typed after a refusal still saves."""
    mock_client.models.list_async = AsyncMock(side_effect=make_sdk_error(403))

    result = await init_integration.start_reconfigure_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "no-access"}
    )
    assert result["errors"] == {"base": "forbidden"}

    mock_client.models.list_async = AsyncMock(return_value=mock_models_response)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "a-key-with-access"}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert init_integration.data[CONF_API_KEY] == "a-key-with-access"


async def test_reconfigure_form_does_not_echo_the_stored_key(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The field starts empty rather than prefilled with the secret.

    A suggested value on a password field is sent to the browser and sits in
    the DOM. There is nothing to edit in a key either -- a replacement is
    typed whole -- so prefilling would only leak it.
    """
    result = await init_integration.start_reconfigure_flow(hass)

    assert _suggested(result, CONF_API_KEY) is None
    for marker in result["data_schema"].schema:
        if marker.schema == CONF_API_KEY:
            assert marker.default is vol.UNDEFINED
            break
    else:
        pytest.fail("the reconfigure form does not ask for an API key")


async def test_reconfigure_entry_abort_reason_has_a_message(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Saving ends with a sentence, not a translation key.

    async_update_reload_and_abort picks the reason itself, so the string is
    never written down in this repository and no static check could find it.
    Home Assistant has no default either: its own strings.json carries only a
    `common` section, with nothing under `config`.
    """
    result = await init_integration.start_reconfigure_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "rotated-api-key"}
    )
    await hass.async_block_till_done()

    path = Path(integration.__file__).parent / "translations" / "en.json"
    aborts = json.loads(path.read_text(encoding="utf-8"))["config"]["abort"]

    assert result["reason"] in aborts, (
        f"reconfiguring aborts with {result['reason']!r}, which has no message "
        f"in en.json and renders to the user as that literal string"
    )


@pytest.mark.parametrize(
    "subentry_type", [SUBENTRY_TYPE_CONVERSATION, SUBENTRY_TYPE_AI_TASK_DATA]
)
async def test_add_subentry(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_client: MagicMock,
    subentry_type: str,
) -> None:
    """Both subentry types can be added, and their options are stored."""
    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, subentry_type),
        context={"source": SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "set_options"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "My agent",
            CONF_MODEL: "mistral-large-latest",
            CONF_TEMPERATURE: 0.3,
        },
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "My agent"
    # The name is consumed as the title and must not leak into the options.
    assert CONF_NAME not in result["data"]
    assert result["data"][CONF_MODEL] == "mistral-large-latest"
    assert result["data"][CONF_TEMPERATURE] == 0.3


async def test_subentry_offers_models_from_api(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The model dropdown is populated from the API, not a hard-coded list."""
    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": SOURCE_USER},
    )

    options = _model_values(result)

    assert DEFAULT_MODEL in options
    assert "mistral-large-latest" in options
    mock_client.models.list_async.assert_awaited()


@pytest.mark.parametrize(
    "subentry_type", [SUBENTRY_TYPE_CONVERSATION, SUBENTRY_TYPE_AI_TASK_DATA]
)
async def test_chat_subentries_hide_models_that_cannot_chat(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_client: MagicMock,
    subentry_type: str,
) -> None:
    """A key reaches OCR, transcription and speech models. None belong here.

    Filtered on the completion_chat capability the API reports, so the list
    stays right as Mistral ships and retires models.
    """
    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, subentry_type),
        context={"source": SOURCE_USER},
    )

    options = _model_values(result)

    # Deprecated last, live models first -- see the ordering test below.
    assert options == [
        "mistral-large-latest",
        DEFAULT_MODEL,
        DEPRECATED_MODEL,
        ORPHANED_MODEL,
    ]
    for excluded in (NON_CHAT_MODEL, "voxtral-mini-latest", "voxtral-speech-latest"):
        assert excluded not in options


async def test_reconfigure_subentry(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """An existing subentry can be reconfigured in place."""
    subentry = next(
        s
        for s in init_integration.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_CONVERSATION
    )

    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": "reconfigure", "subentry_id": subentry.subentry_id},
    )
    assert result["type"] is FlowResultType.FORM

    # Reconfiguring must not ask for a name again.
    assert CONF_NAME not in result["data_schema"].schema

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {CONF_MODEL: "mistral-large-latest", CONF_TEMPERATURE: 0.9},
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"

    updated = init_integration.subentries[subentry.subentry_id]
    assert updated.data[CONF_MODEL] == "mistral-large-latest"
    assert updated.data[CONF_TEMPERATURE] == 0.9


async def test_subentry_aborts_when_entry_not_loaded(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Subentries cannot be added while the entry is unloaded."""
    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": SOURCE_USER},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "entry_not_loaded"


async def test_subentry_aborts_when_api_unreachable(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A failure listing models aborts the subentry flow with a reason.

    The cache has to be dropped first. Setup seeds it, so a form opened
    straight afterwards is served from memory and never reaches the API --
    which is the point of the seed, and means this abort is only reachable
    once the entry has been loaded long enough for the list to go stale.
    """
    init_integration.runtime_data.invalidate_models()
    mock_client.models.list_async = AsyncMock(side_effect=make_sdk_error(500))

    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": SOURCE_USER},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "cannot_connect"


async def test_subentry_aborts_with_forbidden(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A 403 listing models aborts with its own reason, not invalid_auth.

    Cache dropped first for the same reason as the test above.
    """
    init_integration.runtime_data.invalidate_models()
    mock_client.models.list_async = AsyncMock(side_effect=make_sdk_error(403))

    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": SOURCE_USER},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "forbidden"


def _suggested(result: dict, key: str) -> object:
    """Return the suggested value a form offers for a key, or None."""
    for marker in result["data_schema"].schema:
        if marker.schema == key:
            return (marker.description or {}).get("suggested_value")
    return None


async def test_new_conversation_agent_can_control_home_assistant(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    """The agent created with the entry gets the Assist API, as HA's own do.

    Without it the agent answers questions and controls nothing, and does not
    advertise ConversationEntityFeature.CONTROL, so Home Assistant will not
    offer it where control is required. That reads as a broken integration
    rather than an unset option.
    """
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "test-api-key"}
    )
    await hass.async_block_till_done()

    subentry = next(iter(result["result"].subentries.values()))
    assert subentry.data[CONF_LLM_HASS_API] == [llm.LLM_API_ASSIST]
    assert subentry.data[CONF_PROMPT] == llm.DEFAULT_INSTRUCTIONS_PROMPT


async def test_new_subentry_form_seeds_assist_for_conversation_only(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Conversation agents are seeded with Assist; AI tasks are not offered it."""
    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": SOURCE_USER},
    )
    assert _suggested(result, CONF_LLM_HASS_API) == [llm.LLM_API_ASSIST]

    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_AI_TASK_DATA),
        context={"source": SOURCE_USER},
    )
    # AI tasks generate data and are never given control, so the field is absent.
    assert not any(
        marker.schema == CONF_LLM_HASS_API for marker in result["data_schema"].schema
    )


async def test_stale_llm_api_is_dropped_from_the_form(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """An API that no longer exists is not offered back as selected.

    Whatever provided it can be removed after the fact. Keeping the id would
    save it straight back and then fail every message, because resolving an
    unknown API id raises.
    """
    subentry = next(
        s
        for s in init_integration.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_CONVERSATION
    )
    hass.config_entries.async_update_subentry(
        init_integration,
        subentry,
        data={**subentry.data, CONF_LLM_HASS_API: [llm.LLM_API_ASSIST, "gone"]},
    )
    await hass.async_block_till_done()

    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": "reconfigure", "subentry_id": subentry.subentry_id},
    )

    assert _suggested(result, CONF_LLM_HASS_API) == [llm.LLM_API_ASSIST]


async def test_stt_subentry_offers_only_transcription_models(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Speech-to-text lists only models that can actually transcribe.

    Filtering on the capability the API reports, rather than on model names,
    is what keeps this correct when Mistral ships or retires a model.
    """
    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_STT),
        context={"source": SOURCE_USER},
    )

    options = _model_values(result)
    assert options == ["voxtral-mini-latest"]
    assert DEFAULT_MODEL not in options

    # A response length is meaningless for a transcription.
    assert not any(
        marker.schema == CONF_MAX_TOKENS for marker in result["data_schema"].schema
    )


async def test_tts_subentry_offers_speech_models_and_voices(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Text-to-speech lists speech-capable models and the account's voices."""
    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_TTS),
        context={"source": SOURCE_USER},
    )

    models = _model_values(result)
    assert models == ["voxtral-speech-latest"]

    voices = result["data_schema"].schema[CONF_VOICE].config["options"]
    # Languages are in the label, since a bare name distinguishes nothing.
    assert voices == [{"label": "Amelie (fr, en)", "value": "voice-abc"}]


async def test_tts_subentry_aborts_when_voices_cannot_be_listed(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """No voice list means no entity, rather than an entity that cannot speak.

    This asserted the opposite until the endpoint was asked: the form was
    shown without the voice field, on the belief that the API would choose a
    voice. It does not -- a request without one is a 400, which
    async_get_tts_audio turns into silence with nothing in the log.

    An empty list is a failure rather than an empty account, because preset
    voices exist on every workspace and async_list_voices returns an empty
    list for any error rather than raising.
    """
    mock_client.audio.voices.list_async.side_effect = make_sdk_error(500)

    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_TTS),
        context={"source": SOURCE_USER},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "no_voices"


async def test_tts_subentry_requires_a_voice(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The voice field is required, because the endpoint requires one."""
    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_TTS),
        context={"source": SOURCE_USER},
    )

    markers = [
        marker for marker in result["data_schema"].schema if marker.schema == CONF_VOICE
    ]
    assert markers, "the text-to-speech form offers no voice field"
    assert isinstance(markers[0], vol.Required)


def _default(result: dict, key: str) -> object:
    """Return the default a form offers for a key, or None if absent."""
    for marker in result["data_schema"].schema:
        if marker.schema == key:
            return marker.default()
    return None


async def test_tts_subentry_does_not_offer_temperature(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The speech endpoint takes no temperature, so the form must not offer one.

    It did once, and the value was stored and never sent anywhere -- a setting
    that looked like it worked and could not.
    """
    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_TTS),
        context={"source": SOURCE_USER},
    )

    assert not any(
        marker.schema == CONF_TEMPERATURE for marker in result["data_schema"].schema
    )


async def test_stt_subentry_defaults_temperature_low(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Transcription still offers temperature, defaulting to faithful.

    The conversational 0.7 is a licence to guess at unclear audio, which is how
    a voice assistant ends up acting on something nobody said.
    """
    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_STT),
        context={"source": SOURCE_USER},
    )

    assert _default(result, CONF_TEMPERATURE) == DEFAULT_STT_TEMPERATURE


async def test_a_second_entry_can_be_added_with_the_same_key(
    hass: HomeAssistant, mock_client: MagicMock
) -> None:
    """Two entries for one key are allowed, deliberately.

    Nothing sets a unique id, so nothing can abort as already_configured, and
    the translation for that reason was removed rather than the check added.
    Two keys -- work and personal -- is a legitimate setup, and running several
    agents off one key is what subentries are already for.

    If a unique id is ever introduced, this fails and the abort string has to
    come back with it.
    """
    for _ in range(2):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_API_KEY: "test-api-key"}
        )
        await hass.async_block_till_done()

        assert result["type"] is FlowResultType.CREATE_ENTRY

    assert len(hass.config_entries.async_entries(DOMAIN)) == 2


def _model_labels(result: dict) -> dict[str, str]:
    """Return the label shown for each model ID a form offers."""
    return {
        option["value"]: option["label"]
        for option in result["data_schema"].schema[CONF_MODEL].config["options"]
    }


async def test_a_deprecated_model_says_so_in_the_dropdown(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A model on its way out is labelled with the date and its successor.

    The API reports a retirement date and a replacement for every model being
    withdrawn, and the dropdown showed those exactly like any other. Six of the
    models the live API lists are retiring within the month, so somebody could
    pick one today and lose it before they had finished setting it up.

    Labelled rather than hidden: it still works until the date, and someone
    with a reason to choose it should be able to.
    """
    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": SOURCE_USER},
    )

    labels = _model_labels(result)

    assert labels[DEPRECATED_MODEL] == (
        f"{DEPRECATED_MODEL} — retires 2026-08-31, replaced by mistral-medium-3-5"
    )
    # A model with no end date is shown as its plain name, with nothing added.
    assert labels[DEFAULT_MODEL] == DEFAULT_MODEL


async def test_deprecated_models_sort_below_live_ones(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The warning is worthless at position forty of fifty.

    Sorting purely by name buries a retiring model among the ones that are
    fine, so they are grouped: everything current first, everything ending
    after it, each group alphabetical.
    """
    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": SOURCE_USER},
    )

    values = _model_values(result)

    # Both retiring models sit at the end, in name order among themselves.
    assert values[-2:] == [DEPRECATED_MODEL, ORPHANED_MODEL]


async def test_a_model_the_api_no_longer_lists_stays_selectable(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A model that has actually been withdrawn must not vanish from the form.

    This is when someone most needs to open it. Dropping the configured value
    from the options would mean the fix for a retired model is "your model
    disappeared and the form now says something else".
    """
    entry = next(
        subentry
        for subentry in init_integration.subentries.values()
        if subentry.subentry_type == SUBENTRY_TYPE_CONVERSATION
    )
    hass.config_entries.async_update_subentry(
        init_integration, entry, data={CONF_MODEL: "mistral-tiny-2312"}
    )
    await hass.async_block_till_done()

    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": SOURCE_USER},
    )
    assert "mistral-tiny-2312" not in _model_values(result)

    reconfigure = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": "reconfigure", "subentry_id": entry.subentry_id},
    )
    assert "mistral-tiny-2312" in _model_values(reconfigure)


@pytest.mark.parametrize(
    ("subentry_type", "maximum"),
    [
        (SUBENTRY_TYPE_CONVERSATION, 1.0),
        (SUBENTRY_TYPE_AI_TASK_DATA, 1.5),
        (SUBENTRY_TYPE_STT, 1.5),
    ],
)
async def test_temperature_slider_stops_where_the_api_does(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_client: MagicMock,
    subentry_type: str,
    maximum: float,
) -> None:
    """The slider cannot offer a temperature the endpoint will reject.

    It went to 2.0, and nothing accepts that: chat completions caps at 1.5,
    transcription at 1.5, and the conversations endpoint at 1.0. So the top of
    the slider produced a 422 on every request that used it.

    Conversation agents get the lowest of the three because web search moves
    them to the conversations endpoint, and it is a checkbox on this same
    form -- a temperature that works until an unrelated setting is switched on
    is the worse failure.
    """
    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, subentry_type),
        context={"source": SOURCE_USER},
    )

    for marker, selector in result["data_schema"].schema.items():
        if marker.schema == CONF_TEMPERATURE:
            assert selector.config["max"] == maximum
            break
    else:
        pytest.fail(f"{subentry_type} offers no temperature field")


@pytest.mark.parametrize(
    ("subentry_type", "offered"),
    [
        (SUBENTRY_TYPE_CONVERSATION, True),
        (SUBENTRY_TYPE_AI_TASK_DATA, True),
        (SUBENTRY_TYPE_STT, False),
        (SUBENTRY_TYPE_TTS, False),
    ],
)
async def test_top_p_is_offered_only_where_it_does_something(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_client: MagicMock,
    subentry_type: str,
    offered: bool,
) -> None:
    """Neither speech endpoint takes top_p, so neither form offers it.

    The same reasoning that keeps temperature off the text-to-speech form: a
    setting that is stored and then never sent is worse than an absent one,
    because it looks like it is doing something.
    """
    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, subentry_type),
        context={"source": SOURCE_USER},
    )

    present = any(
        marker.schema == CONF_TOP_P for marker in result["data_schema"].schema
    )
    assert present is offered


async def test_opening_the_form_twice_fetches_the_model_list_once(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The end-to-end half: rendering the form again reuses the list.

    Setup fetches once to validate the key. Opening the subentry form twice
    used to add two more round trips, each one delaying the form appearing.
    """
    before = mock_client.models.list_async.await_count

    for _ in range(2):
        await hass.config_entries.subentries.async_init(
            (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
            context={"source": SOURCE_USER},
        )

    # Zero, not one. Setup fetches the list to validate the key and now seeds
    # the cache with it, so neither render costs a round trip.
    assert mock_client.models.list_async.await_count == before


def _has_field(result: dict, field: str) -> bool:
    """Return whether a form offers a named field."""
    return any(marker.schema == field for marker in result["data_schema"].schema)


@pytest.mark.parametrize(
    "subentry_type", [SUBENTRY_TYPE_CONVERSATION, SUBENTRY_TYPE_AI_TASK_DATA]
)
async def test_reasoning_effort_offered_for_a_reasoning_model(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_client: MagicMock,
    subentry_type: str,
) -> None:
    """The field appears when the default model advertises reasoning.

    DEFAULT_MODEL is mistral-small-latest, which reports reasoning: true on
    the live API, so a new subentry starts on a model that accepts the field.
    """
    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, subentry_type),
        context={"source": SOURCE_USER},
    )

    assert _has_field(result, CONF_REASONING_EFFORT)


async def test_reasoning_effort_hidden_for_a_model_that_cannot_reason(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A model without the capability is not offered the field.

    Not cosmetic. Such a model rejects every value including "none" with
    400 "reasoning_effort is not enabled for this model", so offering the
    field would hand someone a setting that breaks their agent.
    """
    subentry = next(
        s
        for s in init_integration.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_CONVERSATION
    )
    hass.config_entries.async_update_subentry(
        init_integration,
        subentry,
        data={**subentry.data, CONF_MODEL: "mistral-large-latest"},
    )
    await hass.async_block_till_done()

    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": "reconfigure", "subentry_id": subentry.subentry_id},
    )

    assert _has_field(result, CONF_MODEL)
    assert not _has_field(result, CONF_REASONING_EFFORT)


async def test_switching_to_a_non_reasoning_model_drops_the_setting(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The stored effort is pruned when the model can no longer accept it.

    This is the gap a config flow cannot close any other way: the form decides
    whether to show the field from the model already stored, so switching to a
    non-reasoning model submits a field that was valid when the form was drawn.
    Saving it would produce a subentry that 400s on every request, from a form
    that was filled in correctly.
    """
    subentry = next(
        s
        for s in init_integration.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_CONVERSATION
    )

    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": "reconfigure", "subentry_id": subentry.subentry_id},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_MODEL: "mistral-large-latest",
            CONF_TEMPERATURE: 0.5,
            CONF_REASONING_EFFORT: "high",
        },
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    updated = init_integration.subentries[subentry.subentry_id]
    assert updated.data[CONF_MODEL] == "mistral-large-latest"
    assert CONF_REASONING_EFFORT not in updated.data


async def test_reasoning_effort_survives_a_reasoning_model(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Pruning is narrow: a model that reasons keeps the setting.

    Asserted because a prune that fired too widely would silently delete a
    working setting, which is harder to notice than one that never fires.
    """
    subentry = next(
        s
        for s in init_integration.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_CONVERSATION
    )

    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": "reconfigure", "subentry_id": subentry.subentry_id},
    )
    await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_MODEL: DEFAULT_MODEL,
            CONF_TEMPERATURE: 0.5,
            CONF_REASONING_EFFORT: "high",
        },
    )
    await hass.async_block_till_done()

    updated = init_integration.subentries[subentry.subentry_id]
    assert updated.data[CONF_REASONING_EFFORT] == "high"


async def test_reasoning_effort_kept_for_an_unlisted_model(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A hand-typed model is left alone rather than assumed incapable.

    The dropdown allows a custom value, and the API stops listing retired
    models. Nothing is known about either, so pruning on "not in the list"
    would delete a setting that may well work.
    """
    subentry = next(
        s
        for s in init_integration.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_CONVERSATION
    )

    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": "reconfigure", "subentry_id": subentry.subentry_id},
    )
    await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_MODEL: "magistral-medium-2512",
            CONF_TEMPERATURE: 0.5,
            CONF_REASONING_EFFORT: "high",
        },
    )
    await hass.async_block_till_done()

    updated = init_integration.subentries[subentry.subentry_id]
    assert updated.data[CONF_REASONING_EFFORT] == "high"


def test_reasoning_effort_offers_only_the_values_the_api_accepts() -> None:
    """Two options, not the six the spec declares nor the seven it 422s with.

    The schema layer names none, minimal, low, medium, high, xhigh and max;
    the model layer then rejects five of those with 400 and code 3051. A
    selector built from either list would offer options that fail.
    """
    assert REASONING_EFFORTS == ("none", "high")


async def test_reauth_dialog_is_given_the_entry_name(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The reauth description names the entry rather than showing {name}.

    Home Assistant fills this one in: ConfigFlow.async_show_form injects
    name=entry.title for any reauth flow that has an entry_id, so the
    integration does not pass it and must not -- the check there is
    `if description_placeholders.get(CONF_NAME) is None`, and supplying it
    here would only duplicate what core already does.

    Asserted anyway, because the translation string depends on that behaviour
    and nothing else in this repository would notice if it went away.
    """
    result = await init_integration.start_reauth_flow(hass)

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"
    assert result["description_placeholders"]["name"] == init_integration.title


def _abort_strings(subentry_type: str) -> dict[str, str]:
    """Return the abort messages declared for a subentry type."""
    path = Path(integration.__file__).parent / "translations" / "en.json"
    translations = json.loads(path.read_text(encoding="utf-8"))
    return translations["config_subentries"][subentry_type]["abort"]


@pytest.mark.parametrize(
    ("subentry_type", "options"),
    [
        (SUBENTRY_TYPE_CONVERSATION, {CONF_MODEL: DEFAULT_MODEL}),
        (SUBENTRY_TYPE_AI_TASK_DATA, {CONF_MODEL: DEFAULT_MODEL}),
        (SUBENTRY_TYPE_STT, {CONF_MODEL: STT_MODEL}),
        (SUBENTRY_TYPE_TTS, {CONF_MODEL: TTS_MODEL, CONF_VOICE: VOICE_ID}),
    ],
)
async def test_reconfigure_abort_reason_has_a_message(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_client: MagicMock,
    subentry_type: str,
    options: dict,
) -> None:
    """Saving a subentry ends with a sentence, not a translation key.

    async_update_and_abort picks its own reason -- reconfigure_successful for
    a reconfigure flow -- so the key is never written down in this repository
    and no static check could find it. Home Assistant has no default to fall
    back on either: its strings.json carries only a `common` section, with
    nothing for config_subentries. So a missing entry renders as the raw key,
    which is what every save did.

    Driven to completion rather than asserted against the file, because the
    point is to catch a reason the integration does not name itself.
    """
    subentry = next(
        s
        for s in init_integration.subentries.values()
        if s.subentry_type == subentry_type
    )

    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, subentry_type),
        context={"source": "reconfigure", "subentry_id": subentry.subentry_id},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], options
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] in _abort_strings(subentry_type), (
        f"{subentry_type} aborts with {result['reason']!r}, which has no message "
        f"in en.json and renders to the user as that literal string"
    )


def test_every_subentry_abort_reason_used_in_code_has_a_message() -> None:
    """The reasons the flow raises itself are covered too.

    The test above catches the success path, which is the one that was broken.
    These are the failure paths, which are harder to reach -- each needs the
    API to fail a particular way -- so they are read from the source instead.
    """
    source = (Path(integration.__file__).parent / "config_flow.py").read_text()
    raised = set(re.findall(r'async_abort\(\s*reason="([^"]+)"', source))

    assert raised, "no abort reasons found -- the pattern may have moved"

    for subentry_type in (
        SUBENTRY_TYPE_CONVERSATION,
        SUBENTRY_TYPE_AI_TASK_DATA,
        SUBENTRY_TYPE_STT,
        SUBENTRY_TYPE_TTS,
    ):
        declared = _abort_strings(subentry_type)
        # no_voices is only reachable for text-to-speech, so the others are
        # not expected to declare it.
        expected = raised - (
            {"no_voices"} if subentry_type != SUBENTRY_TYPE_TTS else set()
        )
        missing = expected - set(declared)
        assert not missing, f"{subentry_type} is missing messages for {sorted(missing)}"
