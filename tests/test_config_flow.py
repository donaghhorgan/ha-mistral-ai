"""Tests for the Mistral AI config and subentry flows."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mistral_ai.const import (
    CONF_API_KEY,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_VOICE,
    DEFAULT_MODEL,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK_DATA,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_STT,
    SUBENTRY_TYPE_TTS,
)

from .conftest import NON_CHAT_MODEL
from .helpers import make_sdk_error


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
        (make_sdk_error(403), "invalid_auth"),
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

    model_selector = result["data_schema"].schema[CONF_MODEL]
    options = model_selector.config["options"]

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

    options = result["data_schema"].schema[CONF_MODEL].config["options"]

    assert options == ["mistral-large-latest", DEFAULT_MODEL]
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
        {CONF_MODEL: "mistral-large-latest", CONF_TEMPERATURE: 1.5},
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"

    updated = init_integration.subentries[subentry.subentry_id]
    assert updated.data[CONF_MODEL] == "mistral-large-latest"
    assert updated.data[CONF_TEMPERATURE] == 1.5


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
    """A failure listing models aborts the subentry flow with a reason."""
    mock_client.models.list_async = AsyncMock(side_effect=make_sdk_error(500))

    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_CONVERSATION),
        context={"source": SOURCE_USER},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "cannot_connect"


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

    options = result["data_schema"].schema[CONF_MODEL].config["options"]
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

    models = result["data_schema"].schema[CONF_MODEL].config["options"]
    assert models == ["voxtral-speech-latest"]

    voices = result["data_schema"].schema[CONF_VOICE].config["options"]
    # Languages are in the label, since a bare name distinguishes nothing.
    assert voices == [{"label": "Amelie (fr, en)", "value": "voice-abc"}]


async def test_tts_subentry_without_voices_omits_the_field(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """No voices means no dropdown, rather than an empty one.

    An empty dropdown reads as a fault, when in fact the endpoint is happy to
    choose a voice itself.
    """
    mock_client.audio.voices.list_async.side_effect = make_sdk_error(500)

    result = await hass.config_entries.subentries.async_init(
        (init_integration.entry_id, SUBENTRY_TYPE_TTS),
        context={"source": SOURCE_USER},
    )

    assert result["type"] is FlowResultType.FORM
    assert not any(
        marker.schema == CONF_VOICE for marker in result["data_schema"].schema
    )
