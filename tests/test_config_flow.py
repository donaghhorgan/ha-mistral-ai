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
    DEFAULT_STT_TEMPERATURE,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK_DATA,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_STT,
    SUBENTRY_TYPE_TTS,
)

from .conftest import DEPRECATED_MODEL, NON_CHAT_MODEL
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
    assert options == ["mistral-large-latest", DEFAULT_MODEL, DEPRECATED_MODEL]
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

    assert values[-1] == DEPRECATED_MODEL


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
