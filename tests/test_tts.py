"""Tests for the Mistral AI text-to-speech platform."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import httpx
import pytest
from homeassistant.components import tts
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import UNDEFINED

from custom_components.mistral_ai.const import CONF_MODEL

from .conftest import TTS_MODEL, VOICE_ID
from .helpers import make_sdk_error

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_homeassistant_custom_component.common import MockConfigEntry

ENTITY_ID = "tts.mistral_ai_tts"
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


async def test_supported_voices_are_reported(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """The account's voices reach Home Assistant's voice picker.

    Without this the picker is empty and the configured voice is the only one
    a user can reach, even though a per-request voice is supported.
    """
    assert _entity(hass).async_get_supported_voices("en") == [
        tts.Voice("voice-abc", "Amelie (fr, en)")
    ]


async def test_voices_are_not_filtered_by_language(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """A voice is offered for a language it does not list.

    Reading a language it was not built for is a worse result, not an error,
    and hiding it would leave nobody able to pick a voice they can hear working.
    """
    assert _entity(hass).async_get_supported_voices("ja") == [
        tts.Voice("voice-abc", "Amelie (fr, en)")
    ]


async def test_no_voices_reports_none(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A failed listing reports no voice list rather than an empty one.

    None is what the base class returns to mean "no list". An empty list reads
    as a fault, when the endpoint is happy to choose a voice itself.
    """
    mock_client.audio.voices.list_async.side_effect = make_sdk_error(500)

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert _entity(hass).async_get_supported_voices("en") is None


async def test_entity_has_a_resolvable_name(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """The entity must have a name, not inherit one from its device.

    Home Assistant's text-to-speech component refuses an engine whose name
    is None or UNDEFINED:

        if engine_instance.name is None or engine_instance.name is UNDEFINED:
            raise HomeAssistantError("TTS engine name is not set.")

    Using _attr_has_entity_name with _attr_name = None gives exactly that,
    and it fails before any request is made -- so it looks identical in the
    UI, logs nothing from this integration, and every playback fails. The
    other tests here call async_get_tts_audio directly on the entity, which
    bypasses that check entirely, which is how it shipped.
    """
    entity = _entity(hass)

    assert entity.name is not None
    assert entity.name is not UNDEFINED
    assert isinstance(entity.name, str)
    assert entity.name


async def test_no_configured_voice_says_so_rather_than_going_quiet(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_client: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An entity with no voice says why, rather than failing anonymously.

    Reachable for entities created before the voice field was required, when
    the form omitted it entirely if the voice listing had failed.

    Home Assistant renders a (None, None) return as "No TTS from <entity>",
    which says something failed and nothing about what -- the same message as
    every other reason speech can fail. The log line is the only thing that
    distinguishes it, and it names the fix.

    Asserted against the entity rather than through async_get_media_source_audio
    because that path depends on the TTS manager's caching and on when the
    entity was rebuilt, so it passes or fails on timing rather than on this
    behaviour. What matters here is ours: no request, and a message that says
    what to do.
    """
    entry = next(
        subentry
        for subentry in init_integration.subentries.values()
        if subentry.subentry_type == "tts"
    )
    hass.config_entries.async_update_subentry(
        init_integration, entry, data={CONF_MODEL: TTS_MODEL}
    )
    await hass.async_block_till_done()

    entity = hass.data["entity_components"]["tts"].get_entity(ENTITY_ID)

    assert await entity.async_get_tts_audio("hello", "en", {}) == (None, None)
    assert "requires one" in caplog.text

    # Not sent at all, rather than sent and refused: the endpoint returns 400
    # without a voice, so the round trip would only confirm what is already
    # known here.
    mock_client.audio.speech.complete_async.assert_not_awaited()
