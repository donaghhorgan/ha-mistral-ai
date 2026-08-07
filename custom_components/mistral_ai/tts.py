"""Text-to-speech platform for the Mistral AI integration."""

from __future__ import annotations

import base64
import binascii
import logging
from typing import TYPE_CHECKING, Any

import httpx
from homeassistant.components import tts
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from mistralai.client.errors import SDKError

from .const import (
    CONF_MODEL,
    CONF_VOICE,
    SPEECH_LANGUAGES,
    SUBENTRY_TYPE_TTS,
    TIMEOUT,
    TTS_AUDIO_FORMAT,
)
from .entity import MistralBaseEntity
from .helpers import async_list_voices

if TYPE_CHECKING:
    from . import MistralConfigEntry

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: MistralConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up text-to-speech entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_TTS:
            continue

        async_add_entities(
            [MistralTTSEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class MistralTTSEntity(tts.TextToSpeechEntity, MistralBaseEntity):
    """Mistral AI text-to-speech entity."""

    def __init__(self, entry: MistralConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)
        self._voices: list[tts.Voice] = []

    async def async_added_to_hass(self) -> None:
        """Fetch the account's voices once the entity is running.

        `async_get_supported_voices` is a synchronous callback, so the list has
        to be in hand before Home Assistant asks for it. Fetching it here means
        one call per entity per reload, rather than one per request.

        This is why the list cannot be hard-coded the way
        google_generative_ai_conversation hard-codes its own: custom voices are
        created against an account, so what exists depends on the API key.
        """
        await super().async_added_to_hass()
        self._voices = await async_list_voices(self.entry.runtime_data.client)

    @callback
    def async_get_supported_voices(self, language: str) -> list[tts.Voice] | None:
        """Return the voices this entity can offer.

        Deliberately not filtered by `language`. Voices report the languages
        they were built for, but a voice reading a language it does not list is
        a worse result rather than an error, and hiding it would leave the user
        unable to pick a voice they can hear working.

        None rather than an empty list when there are none, which is what the
        base class returns to mean "no voice list". That is a display choice
        only -- it does not mean a voice is optional. The endpoint refuses a
        request without one, so an entity that reaches that state is already
        broken and the dropdown is not what will tell anyone.
        """
        return self._voices or None

    @property
    def supported_languages(self) -> list[str]:
        """Return the languages this entity accepts."""
        return sorted(SPEECH_LANGUAGES)

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return "en"

    @property
    def supported_options(self) -> list[str]:
        """Return the options callers may pass per request."""
        return [tts.ATTR_VOICE]

    @property
    def default_options(self) -> dict[str, Any]:
        """Return the options used when a caller passes none.

        The configured voice is the default rather than a hard-coded one:
        which voices exist depends on the account, since custom voices are
        created against it.
        """
        voice = self.subentry.data.get(CONF_VOICE)
        return {tts.ATTR_VOICE: voice} if voice else {}

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> tts.TtsAudioType:
        """Synthesise speech for a message."""
        client = self.entry.runtime_data.client

        # The language is not sent. The speech endpoint takes a voice, and the
        # voice carries its language; passing a conflicting code would be a way
        # to get a French voice reading English badly.
        request: dict[str, Any] = {
            "input": message,
            "model": self.subentry.data[CONF_MODEL],
            "response_format": TTS_AUDIO_FORMAT,
            "timeout_ms": TIMEOUT * 1000,
        }
        voice = options.get(tts.ATTR_VOICE) or self.subentry.data.get(CONF_VOICE)
        if not voice:
            # The endpoint refuses a request with no voice, so this would come
            # back as a 400 and then as silence. Said plainly here instead,
            # because the entity looks configured and nothing else explains it.
            #
            # Reachable only for an entity created before the voice field was
            # made required -- the form used to omit it entirely whenever the
            # voice listing failed.
            _LOGGER.error(
                "No voice is configured for %s, and Mistral AI requires one. "
                "Reconfigure the entity and choose a voice",
                self.entity_id,
            )
            return None, None

        request["voice_id"] = voice

        try:
            response = await client.audio.speech.complete_async(**request)
        except (SDKError, TimeoutError, httpx.HTTPError) as err:
            _LOGGER.error("Error generating speech with Mistral AI: %s", err)
            return None, None
        except Exception:
            _LOGGER.exception("Unexpected error generating speech with Mistral AI")
            return None, None

        # audio_data is documented as base64, and is typed as str. Decoding
        # strictly rather than trusting it: handing Home Assistant a string, or
        # the wrong bytes, produces silence at playback time rather than an
        # error here, which is a miserable thing to debug.
        try:
            audio = base64.b64decode(response.audio_data, validate=True)
        except (AttributeError, TypeError, binascii.Error):
            _LOGGER.error("Mistral AI returned audio that is not valid base64")
            return None, None

        if not audio:
            _LOGGER.error("Mistral AI returned no audio")
            return None, None

        return TTS_AUDIO_FORMAT, audio
