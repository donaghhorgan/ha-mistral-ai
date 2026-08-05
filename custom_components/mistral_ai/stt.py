"""Speech-to-text platform for the Mistral AI integration."""

from __future__ import annotations

import io
import logging
import wave
from typing import TYPE_CHECKING

import httpx
from homeassistant.components import stt
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from mistralai.client.errors import SDKError

from .const import (
    CONF_MODEL,
    CONF_TEMPERATURE,
    DEFAULT_STT_TEMPERATURE,
    SPEECH_LANGUAGES,
    SUBENTRY_TYPE_STT,
    TIMEOUT,
)
from .entity import MistralBaseEntity

if TYPE_CHECKING:
    from collections.abc import AsyncIterable

    from . import MistralConfigEntry

_LOGGER = logging.getLogger(__name__)

# What Home Assistant is allowed to hand us. The assist pipeline delivers raw
# 16-bit 16kHz mono PCM, which is why the list is this narrow: these are the
# properties of that stream, not a limit of the transcription API.
SUPPORTED_FORMATS = [stt.AudioFormats.WAV]
SUPPORTED_CODECS = [stt.AudioCodecs.PCM]
SUPPORTED_BIT_RATES = [stt.AudioBitRates.BITRATE_16]
SUPPORTED_SAMPLE_RATES = [stt.AudioSampleRates.SAMPLERATE_16000]
SUPPORTED_CHANNELS = [stt.AudioChannels.CHANNEL_MONO]


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: MistralConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up speech-to-text entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_STT:
            continue

        async_add_entities(
            [MistralSTTEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


def _to_wav(pcm: bytes, metadata: stt.SpeechMetadata) -> bytes:
    """Wrap raw PCM in a WAV container.

    Home Assistant reports the format as WAV but streams bare PCM frames --
    the metadata describes the samples, not a container. The transcription
    endpoint is given a file and infers the format from it, so without a
    header it sees noise and returns either nothing or nonsense.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(metadata.channel)
        wav.setsampwidth(metadata.bit_rate // 8)
        wav.setframerate(metadata.sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


class MistralSTTEntity(stt.SpeechToTextEntity, MistralBaseEntity):
    """Mistral AI speech-to-text entity."""

    def __init__(self, entry: MistralConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)

    @property
    def supported_languages(self) -> list[str]:
        """Return the languages this entity accepts.

        Home Assistant requires an explicit list -- there is no "any language"
        sentinel as there is for conversation agents. The transcription models
        also detect the language themselves, and the code is passed through
        only as a hint, so this list governs what the pipeline will offer
        rather than what the model can do.
        """
        return sorted(SPEECH_LANGUAGES)

    @property
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return the supported formats."""
        return SUPPORTED_FORMATS

    @property
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return the supported codecs."""
        return SUPPORTED_CODECS

    @property
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return the supported bit rates."""
        return SUPPORTED_BIT_RATES

    @property
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return the supported sample rates."""
        return SUPPORTED_SAMPLE_RATES

    @property
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return the supported channel counts."""
        return SUPPORTED_CHANNELS

    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Transcribe an audio stream."""
        pcm = b"".join([chunk async for chunk in stream])
        if not pcm:
            _LOGGER.debug("Empty audio stream, nothing to transcribe")
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        client = self.entry.runtime_data
        try:
            response = await client.audio.transcriptions.complete_async(
                model=self.subentry.data[CONF_MODEL],
                file={
                    "file_name": "audio.wav",
                    "content": _to_wav(pcm, metadata),
                    "content_type": "audio/wav",
                },
                language=metadata.language.split("-")[0],
                temperature=self.subentry.data.get(
                    CONF_TEMPERATURE, DEFAULT_STT_TEMPERATURE
                ),
                timeout_ms=TIMEOUT * 1000,
            )
        except (SDKError, TimeoutError, httpx.HTTPError) as err:
            # Returning ERROR rather than raising: the pipeline reports a
            # failed transcription to the user and carries on, where an
            # exception would surface as an unhandled integration error.
            _LOGGER.error("Error transcribing audio with Mistral AI: %s", err)
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
        except Exception:
            _LOGGER.exception("Unexpected error transcribing audio with Mistral AI")
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        text = (response.text or "").strip()
        if not text:
            _LOGGER.debug("Mistral AI returned an empty transcription")
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        return stt.SpeechResult(text, stt.SpeechResultState.SUCCESS)
