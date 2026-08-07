"""Fixtures for the Mistral AI tests."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mistral_ai.const import (
    CONF_API_KEY,
    CONF_MODEL,
    CONF_VOICE,
    DEFAULT_MODEL,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK_DATA,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_STT,
    SUBENTRY_TYPE_TTS,
)

from .helpers import make_chunk, make_sdk_error, make_stream
from .openapi import problems

if TYPE_CHECKING:
    from collections.abc import Generator

STT_MODEL = "voxtral-mini-latest"
TTS_MODEL = "voxtral-speech-latest"
VOICE_ID = "voice-abc"
DEPRECATED_MODEL = "mistral-medium-2508"
ORPHANED_MODEL = "mistral-nemo-2407"
NON_CHAT_MODEL = "mistral-ocr-latest"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(
    enable_custom_integrations: None,
) -> Generator[None]:
    """Enable loading of this custom integration in every test."""
    yield


@pytest.fixture(autouse=True)
async def setup_homeassistant(hass: HomeAssistant) -> None:
    """Set up the core integration.

    The conversation component's default agent reads exposed-entity data on
    start, which only exists once `homeassistant` itself has been set up.
    """
    assert await async_setup_component(hass, "homeassistant", {})


def _model_card(
    model_id: str,
    *,
    deprecation: datetime | None = None,
    replacement: str | None = None,
    **capabilities: bool,
) -> MagicMock:
    """Build a models.list_async() entry with real boolean capabilities.

    The flags have to be plain booleans rather than MagicMock attributes:
    every attribute of a MagicMock is truthy, so a capability filter would
    match everything and the test would pass without testing anything.
    """
    card = MagicMock()
    card.id = model_id
    # Real values rather than invented MagicMock attributes, for the same
    # reason as the capabilities below: every attribute of a MagicMock is
    # truthy, so a card built without these would look deprecated.
    card.deprecation = deprecation
    card.deprecation_replacement_model = replacement
    card.capabilities = SimpleNamespace(
        **{
            "audio_transcription": False,
            "audio_speech": False,
            "completion_chat": False,
            "function_calling": False,
            "reasoning": False,
            **capabilities,
        }
    )
    return card


@pytest.fixture
def mock_models_response() -> MagicMock:
    """Return a models.list_async() response covering each capability."""
    response = MagicMock()
    response.data = [
        # Reasoning, like the real mistral-small-latest, so reasoning_effort
        # is offered on the default model rather than only on an odd one out.
        _model_card(
            DEFAULT_MODEL, completion_chat=True, function_calling=True, reasoning=True
        ),
        _model_card(
            "mistral-large-latest", completion_chat=True, function_calling=True
        ),
        _model_card(STT_MODEL, audio_transcription=True),
        _model_card(TTS_MODEL, audio_speech=True),
        # A key can reach plenty of models that have no business in any of
        # these dropdowns -- OCR, embeddings, coding models.
        _model_card(NON_CHAT_MODEL),
        # Retiring, with a named successor. Six of the models the live API
        # lists are in this state today.
        _model_card(
            DEPRECATED_MODEL,
            completion_chat=True,
            function_calling=True,
            deprecation=datetime(2026, 8, 31, 12, 0, tzinfo=UTC),
            replacement="mistral-medium-3-5",
        ),
        # Retiring with nothing named to move to. The API does this, and it
        # is the case where a repair cannot offer a button.
        _model_card(
            ORPHANED_MODEL,
            completion_chat=True,
            deprecation=datetime(2026, 8, 31, 12, 0, tzinfo=UTC),
        ),
    ]
    return response


@pytest.fixture
def mock_client(mock_models_response: MagicMock) -> Generator[MagicMock]:
    """Patch the Mistral client everywhere it is constructed."""
    client = MagicMock()
    client.models.list_async = AsyncMock(return_value=mock_models_response)

    # retrieve_async answers for one model. The AI task entity uses it to ask
    # whether the configured model can call tools, which is a precondition for
    # image generation.
    async def _retrieve(model_id: str, **_kwargs: object) -> MagicMock:
        for card in mock_models_response.data:
            if card.id == model_id:
                return card
        raise make_sdk_error(404)

    client.models.retrieve_async = AsyncMock(side_effect=_retrieve)
    client.chat.stream_async = AsyncMock(
        return_value=make_stream([make_chunk(content="Hello there")])
    )
    transcription = MagicMock()
    transcription.text = "turn on the kitchen light"
    client.audio.transcriptions.complete_async = AsyncMock(return_value=transcription)

    speech = MagicMock()
    speech.audio_data = base64.b64encode(b"ID3 fake mp3 bytes").decode()
    client.audio.speech.complete_async = AsyncMock(return_value=speech)

    voice = MagicMock()
    voice.id = VOICE_ID
    voice.name = "Amelie"
    voice.languages = ["fr", "en"]
    voices = MagicMock()
    voices.items = [voice]
    client.audio.voices.list_async = AsyncMock(return_value=voices)

    # One patch point now: both the integration and the config flow build
    # their client through custom_components.mistral_ai.client.
    with patch("custom_components.mistral_ai.client.Mistral", return_value=client):
        yield client

    # Every request the suite built, checked against the vendored spec on the
    # way out. Done here rather than in a test of its own so that each test
    # already written pays for a schema check without being touched, and so a
    # path nobody thought to check by hand is still covered the moment any
    # test drives it.
    _assert_requests_match_the_spec(client)


@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Return a config entry with one conversation and one AI task subentry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Mistral AI",
        data={CONF_API_KEY: "test-api-key"},
        subentries_data=[
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_CONVERSATION,
                data={CONF_MODEL: DEFAULT_MODEL},
                title="Mistral AI Conversation",
                unique_id=None,
            ),
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_AI_TASK_DATA,
                data={CONF_MODEL: DEFAULT_MODEL},
                title="Mistral AI Task",
                unique_id=None,
            ),
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_STT,
                data={CONF_MODEL: STT_MODEL},
                title="Mistral AI STT",
                unique_id=None,
            ),
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_TTS,
                data={CONF_MODEL: TTS_MODEL, CONF_VOICE: VOICE_ID},
                title="Mistral AI TTS",
                unique_id=None,
            ),
        ],
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
async def init_integration(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_client: MagicMock,
) -> MockConfigEntry:
    """Set up the integration and return its config entry."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    return mock_config_entry


# Which schema describes the body of each call the integration makes. A call
# site missing from here is a failure rather than a skip: a validator that
# quietly ignores what it does not recognise reads as a pass, which is the
# property that let three malformed requests ship.
REQUEST_SCHEMAS = {
    ("chat", "stream_async"): "ChatCompletionRequest",
    ("chat", "complete_async"): "ChatCompletionRequest",
    ("beta.conversations", "start_async"): "ConversationRequestBase",
    ("beta.conversations", "start_stream_async"): "ConversationRequestBase",
    ("audio.speech", "complete_async"): "SpeechRequest",
}


def _assert_requests_match_the_spec(client: MagicMock) -> None:
    """Fail if any recorded call carries a field or value the API rejects."""
    found: list[str] = []

    for (attribute, method), schema in REQUEST_SCHEMAS.items():
        target = client
        for part in attribute.split("."):
            target = getattr(target, part)
        mock = getattr(target, method, None)

        for call in getattr(mock, "await_args_list", None) or []:
            found.extend(
                f"{attribute}.{method}: {problem}"
                for problem in problems(schema, call.kwargs)
            )

    assert not found, "requests do not match the OpenAPI spec:\n  " + "\n  ".join(found)
