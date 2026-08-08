"""Text-to-speech platform for the Mistral AI integration."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
from contextlib import aclosing
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, cast

import httpx
from homeassistant.components import tts
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.httpx_client import get_async_client
from mistralai.client.errors import SDKError

from .const import (
    CONF_API_KEY,
    CONF_MODEL,
    CONF_VOICE,
    DOMAIN,
    SPEECH_LANGUAGES,
    SUBENTRY_TYPE_TTS,
    TIMEOUT,
    TTS_AUDIO_FORMAT,
)
from .entity import MistralBaseEntity, _validation_detail_from_body
from .helpers import async_list_voices

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from . import MistralConfigEntry

_LOGGER = logging.getLogger(__name__)


# Sentences are the unit a speech request is issued in. Home Assistant hands
# the reply over as the conversation agent writes it, so waiting for the whole
# message means waiting for the whole model response before any audio starts.
#
# Split on terminators followed by whitespace, so "20.5 degrees" is not an
# end -- the digit fails the \s+. A minimum length stops every comma-free
# fragment becoming its own billed request with an audible seam at each join.
#
# The negative lookbehind covers dotted abbreviations: in "e.g." the final
# period *is* followed by a space, so the first version of this split there
# and spoke a fragment ending "e.g." -- exactly what its comment claimed it
# did not do. Three characters back being dot-letter-dot identifies the shape
# and rules it out, which catches "e.g.", "i.e." and "U.S." alike.
#
# It does not catch titles: "Dr. Smith" still splits, because nothing about
# "r." distinguishes it from a real sentence end without a list of words.
# That costs a seam and one extra request, and full sentence segmentation is
# a rabbit hole this does not need to enter.
_SENTENCE_END = re.compile(r"(?<=[.!?])(?<!\.\w\.)\s+")

# Below this, keep accumulating rather than issuing a request. Roughly a short
# clause -- long enough that "Yes." and "OK." join whatever follows them,
# short enough that a normal sentence still goes out on its own.
MIN_SPEECH_CHARS = 40


async def _sentences(message_gen: AsyncGenerator[str]) -> AsyncGenerator[str]:
    """Regroup streamed text into speakable chunks.

    The incoming generator is chunked however the model streamed it, which is
    by token and therefore mid-word. Speaking those directly would be one
    request per fragment.

    Slices the accumulated buffer rather than splitting and rejoining it, so
    the whitespace between sentences survives. Rejoining lost it, and "Yes."
    followed by "OK." was spoken as "Yes.OK.".
    """
    buffer = ""

    # Closed with this generator rather than left to the garbage collector.
    # Home Assistant owns message_gen, but we are the ones who stop consuming
    # it early -- when a speech request fails, _sentences is closed while
    # suspended right here, and dropping the reference schedules the
    # finalisation as a task instead of awaiting it.
    #
    # The last of five generators in this chain to be closed properly, and the
    # one that kept the leak alive after the other four were fixed. It only
    # showed intermittently, because whether the collector had run by the time
    # the check looked depended on test ordering.
    async with aclosing(message_gen) as chunks:
        async for chunk in chunks:
            buffer += chunk

            while True:
                # The first boundary with enough text before it to be worth
                # a request. Earlier ones are left in place, so a short reply
                # keeps accumulating instead of going out a clause at a time.
                boundary = next(
                    (
                        match
                        for match in _SENTENCE_END.finditer(buffer)
                        if match.start() >= MIN_SPEECH_CHARS
                    ),
                    None,
                )
                if boundary is None:
                    break

                yield buffer[: boundary.start()].strip()
                buffer = buffer[boundary.end() :]

    if final := buffer.strip():
        yield final


async def _speech_events(response: httpx.Response) -> AsyncGenerator[dict[str, Any]]:
    """Yield the decoded payload of each event in a speech stream.

    Server-sent events are framed by blank lines, and a single event's `data`
    may be split across several lines which the client joins with newlines.
    The first version of this treated every `data:` line as a whole JSON
    document, which is true of every response seen from this endpoint so far
    and not promised by the format.

    The joining rule bounds how bad that could get: a server can only break a
    line where a newline is valid JSON whitespace, so the base64 string itself
    is never split -- that would corrupt it. But a payload broken between
    fields would have failed to parse and been dropped, losing audio quietly,
    which is the failure this file keeps guarding against.

    Non-data lines are skipped rather than read. `event:` names the event, but
    audio is recognised by carrying an `audio_data` field, so a renamed event
    keeps working and an unfamiliar one is simply ignored.
    """
    data: list[str] = []

    # Closed with the generator rather than left to the garbage collector,
    # for the same reason as the callers below: it holds the open response.
    #
    # Cast because httpx annotates this as AsyncIterator, which has no
    # aclose, while returning an async generator that does. Narrowing to what
    # it actually returns rather than dropping the close, since dropping it is
    # the bug being fixed.
    lines_gen = cast("AsyncGenerator[str]", response.aiter_lines())

    async with aclosing(lines_gen) as lines:
        async for line in lines:
            if line.startswith("data:"):
                # One optional space after the colon belongs to the framing.
                data.append(line[5:].removeprefix(" "))
                continue

            if line.strip():
                # event:, id:, retry:, or a comment.
                continue

            # A blank line ends the event.
            if data:
                payload, data = "\n".join(data), []
                if (event := _decode_event(payload)) is not None:
                    yield event

    # A final event with no trailing blank line before the stream closed.
    if data and (event := _decode_event("\n".join(data))) is not None:
        yield event


def _decode_event(payload: str) -> dict[str, Any] | None:
    """Return one event's JSON, or None having said it could not be read.

    Logged at warning rather than debug: by this point a whole event has been
    assembled, so failing to read it means audio is being dropped. Not raised,
    because an unfamiliar event that is not audio is harmless -- the empty
    stream is caught by the caller instead, which is the case that matters.
    """
    try:
        event = json.loads(payload)
    except ValueError:
        _LOGGER.warning(
            "Could not read a Mistral AI speech event, skipping it: %s",
            payload[:120],
        )
        return None

    return event if isinstance(event, dict) else None


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

    def _resolve_voice(self, options: dict[str, Any]) -> str | None:
        """Return the voice to speak with, or None having said why.

        The endpoint refuses a request without one -- "Either ref_audio or
        voice must be provided" -- and that 400 reaches the user as silence.
        Reachable only for an entity created before the voice field was made
        required, since the form used to omit it whenever the listing failed.
        """
        voice = options.get(tts.ATTR_VOICE) or self.subentry.data.get(CONF_VOICE)
        if not voice:
            _LOGGER.error(
                "No voice is configured for %s, and Mistral AI requires one. "
                "Reconfigure the entity and choose a voice",
                self.entity_id,
            )
            return None
        return voice

    async def _async_stream_speech(
        self, message: str, voice: str
    ) -> AsyncGenerator[bytes]:
        """Yield audio for one message as the endpoint produces it.

        Sent over httpx rather than through the SDK, which does not expose
        this: `audio.speech` has `complete_async` and no streaming
        counterpart, though the endpoint has taken `stream` all along and
        answers with `text/event-stream`.

        The base URL is read off the SDK so it stays defined in one place,
        and the client is Home Assistant's shared one -- the same instance
        async_create_client hands the SDK, so no second connection pool is
        built here. Taken from get_async_client rather than off the SDK
        configuration, where it is typed as possibly absent.

        Building the request by hand means the failure handling the SDK path
        gets from _convert_error has to be asked for explicitly, which is
        what _error_for_status is for.
        """
        base_url, _ = (
            self.entry.runtime_data.client.sdk_configuration.get_server_details()
        )

        payload = {
            "input": message,
            "model": self.subentry.data[CONF_MODEL],
            "response_format": TTS_AUDIO_FORMAT,
            "voice_id": voice,
            "stream": True,
        }

        spoke = False

        try:
            async with get_async_client(self.hass).stream(
                "POST",
                f"{base_url.rstrip('/')}/v1/audio/speech",
                json=payload,
                headers={"Authorization": f"Bearer {self.entry.data[CONF_API_KEY]}"},
                timeout=TIMEOUT,
            ) as response:
                if response.status_code != HTTPStatus.OK:
                    # The body has to be pulled in before it can be read at
                    # all on a streamed response, and it is small: these are
                    # JSON errors, not audio.
                    await response.aread()
                    raise self._error_for_status(
                        response.status_code,
                        response.text[:500],
                        _validation_detail_from_body(response.text),
                    )

                # Closed explicitly: the decode below raises, and abandoning
                # this generator is what left an async_generator_athrow task
                # pending after the test that caused it.
                async with aclosing(_speech_events(response)) as events:
                    # Drained rather than abandoned, in the finally below.
                    # Walking away from a partly-read response leaves httpx's
                    # own generators for the garbage collector: aiter_lines
                    # iterates aiter_text, which iterates aiter_bytes, both
                    # with bare `async for` loops, so closing the outer one
                    # does not close the inner two. Nothing we close on our
                    # side reaches them.
                    #
                    # Measured on a streamed response: abandoning after one
                    # line leaves one finaliser task, draining to the end
                    # leaves none -- and that holds for aiter_bytes too, so
                    # picking a different iterator does not help. Consuming
                    # the rest is the only thing that does.
                    #
                    # Cheap in practice: this only runs when a request is cut
                    # short, and the remainder is at most one sentence of
                    # audio.
                    try:
                        async for event in events:
                            if not (encoded := event.get("audio_data")):
                                # The terminating speech.audio.done carries no
                                # audio.
                                continue

                            # Decoded strictly. Handing Home Assistant the wrong
                            # bytes produces silence at playback rather than an
                            # error here, which is a miserable thing to debug.
                            try:
                                chunk = base64.b64decode(encoded, validate=True)
                            except (TypeError, binascii.Error) as err:
                                raise HomeAssistantError(
                                    translation_domain=DOMAIN,
                                    translation_key="speech_not_base64",
                                ) from err

                            spoke = True
                            yield chunk
                    finally:
                        # However this loop ends -- normally, on the
                        # decode error above, or because the caller
                        # stopped consuming -- the rest of the response
                        # is read before letting go of it.
                        async for _ in events:
                            pass

        except (TimeoutError, httpx.HTTPError) as err:
            # The non-streaming path catches these and this did not, so a
            # connection reset part way through a reply surfaced as a raw
            # httpx traceback rather than as an error about speech.
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="speech_transport_error",
                translation_placeholders={"error": str(err)},
            ) from err

        if not spoke:
            # A stream that parsed cleanly and produced nothing is the failure
            # this file keeps guarding against: silence with no explanation.
            # Reachable if the event shape moves under us, which is exactly
            # when nobody is looking for it.
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="speech_no_audio",
            )

    async def async_stream_tts_audio(
        self, request: tts.TTSAudioRequest
    ) -> tts.TTSAudioResponse:
        """Speak a reply as it is written, rather than once it is finished.

        Overriding this is what makes async_supports_streaming_input true, and
        it replaces a shim that joined the whole message, waited for the whole
        audio file and yielded it as a single chunk -- so nothing played until
        the last byte had arrived.

        Two delays go away. The endpoint returns its first audio in about
        0.7s streamed against about 2.4s complete, measured over five trials
        each; and chunking the incoming text means the first sentence is
        spoken while the model is still writing the rest.
        """
        voice = self._resolve_voice(request.options)

        async def data_gen() -> AsyncGenerator[bytes]:
            if voice is None:
                # Nothing to speak with. Ending the stream empty rather than
                # raising matches what the non-streaming path does, and the
                # reason has already been logged.
                return

            # Closed deterministically rather than left to the garbage
            # collector's async-generator hook. Each speech stream holds an
            # open httpx response, so abandoning one part way through -- which
            # is what happens when any of these raise -- would keep the
            # connection until finalisation caught up. Home Assistant's
            # lingering-task check sees that hook running after the test that
            # caused it, which is how it surfaced.
            async with aclosing(_sentences(request.message_gen)) as sentences:
                async for sentence in sentences:
                    async with aclosing(
                        self._async_stream_speech(sentence, voice)
                    ) as speech:
                        async for chunk in speech:
                            yield chunk

        return tts.TTSAudioResponse(TTS_AUDIO_FORMAT, data_gen())

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
        voice = self._resolve_voice(options)
        if voice is None:
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
