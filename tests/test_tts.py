"""Tests for the Mistral AI text-to-speech platform."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import httpx
import pytest
from homeassistant.components import tts
from homeassistant.core import HomeAssistant

from .conftest import TTS_MODEL, VOICE_ID
from .helpers import make_sdk_error

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_homeassistant_custom_component.common import MockConfigEntry

ENTITY_ID = "tts.mistral_ai_text_to_speech"
AUDIO = b"ID3 fake mp3 bytes"


def _entity(hass: HomeAssistant) -> tts.TextToSpeechEntity:
    """Return the text-to-speech entity under test."""
    entity = hass.data[tts.DOMAIN].get_entity(ENTITY_ID)
    assert entity is not None
    return entity


async def test_entity_is_created(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """The TTS subentry produces an entity."""
    assert hass.states.get(ENTITY_ID) is not None


async def test_generates_audio(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A message is synthesised and returned as decoded bytes."""
    extension, audio = await _entity(hass).async_get_tts_audio(
        "the kitchen light is on", "en", {}
    )

    assert extension == "mp3"
    assert audio == AUDIO

    kwargs = mock_client.audio.speech.complete_async.await_args.kwargs
    assert kwargs["input"] == "the kitchen light is on"
    assert kwargs["model"] == TTS_MODEL
    assert kwargs["voice_id"] == VOICE_ID
    assert kwargs["response_format"] == "mp3"


async def test_audio_is_base64_decoded(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The API returns base64 text, and Home Assistant needs bytes.

    Handing the string through unchanged produces silence at playback rather
    than an error here, so this asserts on the decoded value specifically.
    """
    encoded = base64.b64encode(b"different audio").decode()
    mock_client.audio.speech.complete_async.return_value.audio_data = encoded

    _, audio = await _entity(hass).async_get_tts_audio("hello", "en", {})

    assert audio == b"different audio"
    assert audio != encoded


async def test_requested_voice_overrides_the_configured_one(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A caller can ask for a different voice per request."""
    await _entity(hass).async_get_tts_audio(
        "hello", "en", {tts.ATTR_VOICE: "voice-other"}
    )

    kwargs = mock_client.audio.speech.complete_async.await_args.kwargs
    assert kwargs["voice_id"] == "voice-other"


async def test_language_is_not_sent(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The voice carries the language, so no language code is sent.

    Sending one alongside a voice is how you get a French voice reading
    English badly.
    """
    await _entity(hass).async_get_tts_audio("bonjour", "fr", {})

    assert "language" not in mock_client.audio.speech.complete_async.await_args.kwargs


@pytest.mark.parametrize(
    "side_effect",
    [
        make_sdk_error(500),
        httpx.ConnectError("no route to host"),
        TimeoutError(),
        ValueError("something unexpected"),
    ],
)
async def test_api_failures_return_no_audio(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_client: MagicMock,
    side_effect: Exception,
) -> None:
    """Failures return (None, None), which is how the TTS layer reports one."""
    mock_client.audio.speech.complete_async.side_effect = side_effect

    assert await _entity(hass).async_get_tts_audio("hello", "en", {}) == (None, None)


@pytest.mark.parametrize("audio_data", ["not base64 at all!", "", None])
async def test_undecodable_audio_returns_no_audio(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_client: MagicMock,
    audio_data: str | None,
) -> None:
    """Anything that is not decodable base64 fails here, not at playback.

    Passing it through would surface as a silent media player, which is a
    miserable thing to trace back to the API.
    """
    mock_client.audio.speech.complete_async.return_value.audio_data = audio_data

    assert await _entity(hass).async_get_tts_audio("hello", "en", {}) == (None, None)


async def test_declared_contract(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """Languages, options and defaults are what the pipeline matches on."""
    entity = _entity(hass)

    assert entity.default_language == "en"
    assert "en" in entity.supported_languages
    assert entity.supported_languages == sorted(entity.supported_languages)
    assert entity.supported_options == [tts.ATTR_VOICE]
    assert entity.default_options == {tts.ATTR_VOICE: VOICE_ID}
