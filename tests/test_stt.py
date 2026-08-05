"""Tests for the Mistral AI speech-to-text platform."""

from __future__ import annotations

import io
import wave
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import httpx
import pytest
from homeassistant.components import stt
from homeassistant.core import HomeAssistant

from custom_components.mistral_ai.const import (
    CONF_TEMPERATURE,
    DEFAULT_STT_TEMPERATURE,
    SUBENTRY_TYPE_STT,
)

from .helpers import make_sdk_error

if TYPE_CHECKING:
    from collections.abc import AsyncIterable

    from pytest_homeassistant_custom_component.common import MockConfigEntry

ENTITY_ID = "stt.mistral_ai_speech_to_text"

# 16-bit 16kHz mono, which is what the assist pipeline delivers.
PCM = b"\x01\x02" * 1600


def _metadata() -> stt.SpeechMetadata:
    """Return metadata matching what the assist pipeline sends."""
    return stt.SpeechMetadata(
        language="en-GB",
        format=stt.AudioFormats.WAV,
        codec=stt.AudioCodecs.PCM,
        bit_rate=stt.AudioBitRates.BITRATE_16,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )


async def _stream(*chunks: bytes) -> AsyncIterable[bytes]:
    """Yield audio chunks the way the pipeline does."""
    for chunk in chunks:
        yield chunk


def _entity(hass: HomeAssistant) -> stt.SpeechToTextEntity:
    """Return the speech-to-text entity under test."""
    entity = hass.data[stt.DOMAIN].get_entity(ENTITY_ID)
    assert entity is not None
    return entity


async def test_entity_is_created(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """The STT subentry produces an entity."""
    assert hass.states.get(ENTITY_ID) is not None


async def test_transcribes_audio(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A stream is transcribed and the text returned."""
    result = await _entity(hass).async_process_audio_stream(
        _metadata(), _stream(PCM[:1000], PCM[1000:])
    )

    assert result.result is stt.SpeechResultState.SUCCESS
    assert result.text == "turn on the kitchen light"

    call = mock_client.audio.transcriptions.complete_async.await_args
    assert call.kwargs["model"] == "voxtral-mini-latest"
    # Home Assistant sends a full locale; the API wants a bare language code.
    assert call.kwargs["language"] == "en"


async def test_audio_is_sent_as_a_wav_file(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The raw PCM is wrapped in a WAV container before it is sent.

    Home Assistant reports the format as WAV but streams bare PCM frames. The
    transcription endpoint infers the format from the file it is given, so
    without a header it receives noise -- and would fail at the API rather
    than here, which is a much worse place to find out.
    """
    await _entity(hass).async_process_audio_stream(_metadata(), _stream(PCM))

    sent = mock_client.audio.transcriptions.complete_async.await_args.kwargs["file"]
    assert sent["file_name"].endswith(".wav")

    with wave.open(io.BytesIO(sent["content"]), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.readframes(wav.getnframes()) == PCM


@pytest.mark.parametrize(
    "side_effect",
    [
        make_sdk_error(500),
        httpx.ConnectError("no route to host"),
        TimeoutError(),
        ValueError("something unexpected"),
    ],
)
async def test_api_failures_return_an_error_result(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_client: MagicMock,
    side_effect: Exception,
) -> None:
    """Failures come back as an ERROR result rather than an exception.

    The pipeline reports a failed transcription and carries on; an exception
    escaping here surfaces as an unhandled integration error instead.
    """
    mock_client.audio.transcriptions.complete_async.side_effect = side_effect

    result = await _entity(hass).async_process_audio_stream(_metadata(), _stream(PCM))

    assert result.result is stt.SpeechResultState.ERROR
    assert result.text is None


async def test_empty_audio_is_not_sent(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """An empty stream fails locally instead of paying for a round trip."""
    result = await _entity(hass).async_process_audio_stream(_metadata(), _stream(b""))

    assert result.result is stt.SpeechResultState.ERROR
    mock_client.audio.transcriptions.complete_async.assert_not_awaited()


async def test_empty_transcription_is_an_error(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Silence transcribes to nothing, which is a failure, not an empty command."""
    empty = MagicMock()
    empty.text = "   "
    mock_client.audio.transcriptions.complete_async.return_value = empty

    result = await _entity(hass).async_process_audio_stream(_metadata(), _stream(PCM))

    assert result.result is stt.SpeechResultState.ERROR
    assert result.text is None


async def test_declared_audio_contract(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """The entity accepts exactly what the assist pipeline produces.

    These properties are a contract: Home Assistant will not route audio to
    an entity that does not declare the format it is about to send, so a
    wrong value here means the entity is silently never used.
    """
    entity = _entity(hass)

    assert entity.supported_formats == [stt.AudioFormats.WAV]
    assert entity.supported_codecs == [stt.AudioCodecs.PCM]
    assert entity.supported_bit_rates == [stt.AudioBitRates.BITRATE_16]
    assert entity.supported_sample_rates == [stt.AudioSampleRates.SAMPLERATE_16000]
    assert entity.supported_channels == [stt.AudioChannels.CHANNEL_MONO]

    # Bare language codes, since that is what is matched against a pipeline.
    assert "en" in entity.supported_languages
    assert entity.supported_languages == sorted(entity.supported_languages)


async def test_temperature_defaults_to_faithful(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """An unconfigured subentry transcribes at the low default.

    The setting used to be offered by the form and then dropped on the floor.
    This asserts it reaches the API, and at a value that does not invite the
    model to guess.
    """
    await _entity(hass).async_process_audio_stream(_metadata(), _stream(PCM))

    kwargs = mock_client.audio.transcriptions.complete_async.await_args.kwargs
    assert kwargs["temperature"] == DEFAULT_STT_TEMPERATURE


async def test_configured_temperature_is_sent(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A temperature set on the subentry is the one used."""
    subentry = next(
        s
        for s in init_integration.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_STT
    )
    hass.config_entries.async_update_subentry(
        init_integration, subentry, data={**subentry.data, CONF_TEMPERATURE: 0.4}
    )
    await hass.async_block_till_done()

    await _entity(hass).async_process_audio_stream(_metadata(), _stream(PCM))

    kwargs = mock_client.audio.transcriptions.complete_async.await_args.kwargs
    assert kwargs["temperature"] == 0.4
