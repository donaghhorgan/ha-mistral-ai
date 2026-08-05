"""Fixtures for the Mistral AI tests."""

from __future__ import annotations

import base64
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

from .helpers import make_chunk, make_stream

if TYPE_CHECKING:
    from collections.abc import Generator

STT_MODEL = "voxtral-mini-latest"
TTS_MODEL = "voxtral-speech-latest"
VOICE_ID = "voice-abc"


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


def _model_card(model_id: str, **capabilities: bool) -> MagicMock:
    """Build a models.list_async() entry with real boolean capabilities.

    The flags have to be plain booleans rather than MagicMock attributes:
    every attribute of a MagicMock is truthy, so a capability filter would
    match everything and the test would pass without testing anything.
    """
    card = MagicMock()
    card.id = model_id
    card.capabilities = SimpleNamespace(
        **{"audio_transcription": False, "audio_speech": False, **capabilities}
    )
    return card


@pytest.fixture
def mock_models_response() -> MagicMock:
    """Return a models.list_async() response covering each capability."""
    response = MagicMock()
    response.data = [
        _model_card(DEFAULT_MODEL),
        _model_card("mistral-large-latest"),
        _model_card(STT_MODEL, audio_transcription=True),
        _model_card(TTS_MODEL, audio_speech=True),
    ]
    return response


@pytest.fixture
def mock_client(mock_models_response: MagicMock) -> Generator[MagicMock]:
    """Patch the Mistral client everywhere it is constructed."""
    client = MagicMock()
    client.models.list_async = AsyncMock(return_value=mock_models_response)
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

    with (
        patch("custom_components.mistral_ai.Mistral", return_value=client),
        patch(
            "custom_components.mistral_ai.config_flow.Mistral",
            return_value=client,
        ),
    ):
        yield client


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
