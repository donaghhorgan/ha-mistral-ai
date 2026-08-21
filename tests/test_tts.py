"""Tests for the Mistral AI text-to-speech platform."""

from __future__ import annotations

import ast
import asyncio
import base64
import gc
import json
import logging
from contextlib import aclosing, contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx
from homeassistant.components import tts
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.typing import UNDEFINED
from packaging.version import Version

from custom_components.mistral_ai import tts as tts_module
from custom_components.mistral_ai.const import CONF_MODEL, VOICE_PAGE_SIZE
from custom_components.mistral_ai.helpers import async_list_voices
from custom_components.mistral_ai.tts import _sentences

from .conftest import TTS_MODEL, VOICE_ID
from .helpers import make_sdk_error

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator

    from pytest_homeassistant_custom_component.common import MockConfigEntry

ENTITY_ID = "tts.mistral_ai_tts"
AUDIO = b"ID3 fake mp3 bytes"


@contextmanager
def caplog_at_warning() -> Iterator[list[str]]:
    """Collect warning messages from the integration's logger."""
    records: list[str] = []
    logger = logging.getLogger("custom_components.mistral_ai.tts")

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


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


async def test_every_voice_is_listed_not_just_the_first_page(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Voices are paged, and all the pages are read.

    The listing endpoint defaults to ten per page, in the API and in the SDK
    signature both, and this used to call it with no arguments. On the account
    it was found on that showed 10 of 31 voices -- no error, no warning, just
    a dropdown missing two thirds of its entries and no way to tell.
    """

    def _voice(index: int) -> MagicMock:
        voice = MagicMock()
        voice.id = f"voice-{index}"
        voice.name = f"Voice {index:03d}"
        voice.languages = ["en"]
        return voice

    # 150 voices over two pages: a full one, then a short one that ends it.
    pages = [_voice(i) for i in range(150)]

    async def _list(*, limit: int, offset: int, **_: object) -> MagicMock:
        response = MagicMock()
        response.items = pages[offset : offset + limit]
        response.total = len(pages)
        return response

    mock_client.audio.voices.list_async = AsyncMock(side_effect=_list)

    voices = await async_list_voices(mock_client)

    assert len(voices) == 150

    # And it asked for the biggest page it is allowed, rather than paging 15
    # times at the default of 10.
    first = mock_client.audio.voices.list_async.await_args_list[0]
    assert first.kwargs["limit"] == VOICE_PAGE_SIZE


def _sse(*payloads: dict) -> bytes:
    """Build a speech event stream the way the endpoint sends one."""
    lines = []
    for payload in payloads:
        lines.append(f"event: {payload['type']}")
        lines.append(f"data: {json.dumps(payload)}")
        lines.append("")
    return "\n".join(lines).encode()


def _delta(audio: bytes) -> dict:
    """Build one speech.audio.delta event carrying audio."""
    return {
        "type": "speech.audio.delta",
        "audio_data": base64.b64encode(audio).decode(),
    }


DONE = {"type": "speech.audio.done"}


async def _gen(*chunks: str) -> AsyncGenerator[str]:
    """Yield text the way a streaming conversation agent does."""
    for chunk in chunks:
        yield chunk


async def _collect(entity: tts.TextToSpeechEntity, *chunks: str) -> tuple[str, bytes]:
    """Drive the streaming path and return its extension and audio."""
    response = await entity.async_stream_tts_audio(
        tts.TTSAudioRequest(language="en", options={}, message_gen=_gen(*chunks))
    )
    # Closed even when consuming it raises, which several of these tests do on
    # purpose. Leaving it to the garbage collector schedules the finalisation
    # after the test has ended, and Home Assistant reports that as a lingering
    # task on newer releases.
    async with aclosing(response.data_gen) as data:
        return response.extension, b"".join([part async for part in data])


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        # Split on terminators, but only once a chunk is worth a request.
        (
            [
                "This is the first sentence, and it is long enough on its "
                "own to be worth a request. ",
                "And a second one.",
            ],
            [
                "This is the first sentence, and it is long enough on its "
                "own to be worth a request.",
                "And a second one.",
            ],
        ),
        # Short leading fragments join what follows rather than each becoming
        # a billed request with an audible seam at the join.
        (
            ["Yes. ", "OK. ", "The kitchen light is on and the hall light is off."],
            ["Yes. OK. The kitchen light is on and the hall light is off."],
        ),
        # A decimal point is not the end of a sentence.
        (
            ["The thermostat is set to 20.5 degrees in the living room right now."],
            ["The thermostat is set to 20.5 degrees in the living room right now."],
        ),
        # Nor is a dotted abbreviation, which the comment claimed and the
        # regex did not do: "e.g." ends in a period followed by a space, so
        # it split there and spoke a fragment ending "e.g.". Only caught
        # past the minimum length, which is how it went unnoticed.
        (
            [
                "The sensor readings look normal today, e.g. the kitchen "
                "is at 20.5 degrees right now."
            ],
            [
                "The sensor readings look normal today, e.g. the kitchen "
                "is at 20.5 degrees right now."
            ],
        ),
        # A real sentence end still splits when an abbreviation precedes it.
        (
            [
                "Several rooms are warm right now, e.g. the kitchen and the "
                "front hall. The porch light is still on."
            ],
            [
                "Several rooms are warm right now, e.g. the kitchen and the "
                "front hall.",
                "The porch light is still on.",
            ],
        ),
        # Token-by-token arrival, which is how it actually turns up.
        (
            ["The kit", "chen ligh", "t is on and the hallway light is off."],
            ["The kitchen light is on and the hallway light is off."],
        ),
        # A title is not a sentence end. The regex this replaced split here,
        # because nothing about "r." tells a lookbehind it is part of a name,
        # and request two began "Seuss." with no idea what preceded it.
        (
            [
                "The author published under the pen name Dr. Seuss for most "
                "of his career."
            ],
            [
                "The author published under the pen name Dr. Seuss for most "
                "of his career."
            ],
        ),
        # Chinese has no space after its full stop, so the old regex found no
        # boundary at all and spoke a whole reply as one request. The
        # integration ships a zh-Hans translation and an agent answers in
        # whatever language it is asked in, so this was reachable.
        (
            ["天气很好。明天会下雨。后天转晴，气温回升到二十度左右。"],
            ["天气很好。明天会下雨。后天转晴，气温回升到二十度左右。"],
        ),
        # Markdown is written, not spoken. The model emits it and the library
        # strips it, so the asterisks never reach the speech endpoint.
        (
            ["The hallway light is **on** and the porch light is *off* now."],
            ["The hallway light is on and the porch light is off now."],
        ),
        # Nothing said means nothing spoken.
        ([], []),
    ],
)
async def test_sentences_regroups_streamed_text(
    chunks: list[str], expected: list[str]
) -> None:
    """Text is regrouped into speakable units before any request is made.

    The incoming generator is chunked however the model streamed it, which is
    by token and therefore mid-word. Speaking those directly would be one
    request per fragment.
    """
    assert [sentence async for sentence in _sentences(_gen(*chunks))] == expected


@respx.mock
async def test_streaming_yields_audio_as_it_arrives(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Deltas are decoded and yielded rather than buffered into one blob.

    The endpoint answers a streamed speech request with text/event-stream,
    which the SDK has no method for -- audio.speech offers complete_async and
    nothing else -- so this path is built on httpx directly.
    """
    route = respx.post("https://api.mistral.ai/v1/audio/speech").mock(
        return_value=httpx.Response(
            200,
            content=_sse(_delta(b"ID3 first"), _delta(b" second"), DONE),
            headers={"content-type": "text/event-stream"},
        )
    )

    extension, audio = await _collect(
        _entity(hass), "The kitchen light is on and the hallway light is off."
    )

    assert extension == "mp3"
    assert audio == b"ID3 first second"

    sent = json.loads(route.calls.last.request.content)
    assert sent["stream"] is True
    assert sent["voice_id"] == VOICE_ID
    assert sent["model"] == TTS_MODEL
    assert sent["response_format"] == "mp3"
    assert "Bearer" in route.calls.last.request.headers["authorization"]


@pytest.mark.skipif(
    Version(version("sentence-stream")) < Version("1.3"),
    reason=(
        "sentence_stream 1.1.0, which Home Assistant 2025.8.0 pins exactly, "
        "holds a boundary that lands at the end of a streamed chunk until "
        "more text arrives, so this reply goes out as one request rather "
        "than two. Speech still works and nothing is lost -- the first audio "
        "just starts later, which is the whole point of splitting. Skipped "
        "rather than loosened, so the assertion keeps its teeth on every "
        "version that can honour it."
    ),
)
@respx.mock
async def test_each_sentence_is_spoken_as_it_is_written(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A sentence goes out before the model has written the next one.

    This is the half that matters most for a long reply: the shim this
    replaces joined the entire message first, so audio could not start until
    the conversation agent had finished composing.
    """
    route = respx.post("https://api.mistral.ai/v1/audio/speech").mock(
        return_value=httpx.Response(
            200,
            content=_sse(_delta(b"audio"), DONE),
            headers={"content-type": "text/event-stream"},
        )
    )

    _, audio = await _collect(
        _entity(hass),
        "The kitchen light is on right now and the hall light is off as well. ",
        "The garage door is closed and the porch light is currently on too.",
    )

    assert route.call_count == 2
    assert audio == b"audioaudio"

    spoken = [json.loads(call.request.content)["input"] for call in route.calls]
    assert spoken == [
        "The kitchen light is on right now and the hall light is off as well.",
        "The garage door is closed and the porch light is currently on too.",
    ]


@respx.mock
async def test_streaming_reports_a_refused_request(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """An error body is raised rather than played as silence.

    Built by hand rather than via the SDK, so the status mapping has to be
    asked for explicitly -- and a streamed response has to be read before its
    body can be touched at all.

    Asserted on the translation key rather than the text, because that is what
    reaches the user: this path used to raise an English f-string with the raw
    response body in it, the only user-facing error in the integration without
    a key.
    """
    respx.post("https://api.mistral.ai/v1/audio/speech").mock(
        return_value=httpx.Response(
            400, json={"message": "Either ref_audio or voice must be provided."}
        )
    )

    with pytest.raises(HomeAssistantError) as raised:
        await _collect(_entity(hass), "The kitchen light is on, and the hall is dark.")

    assert raised.value.translation_key == "api_error"


@respx.mock
async def test_streaming_rejects_audio_that_is_not_base64(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Bad audio is an error here rather than silence at playback.

    Decoded strictly for the same reason the non-streaming path does it:
    handing Home Assistant the wrong bytes fails at the speaker, a long way
    from the cause.
    """
    respx.post("https://api.mistral.ai/v1/audio/speech").mock(
        return_value=httpx.Response(
            200,
            content=b'event: speech.audio.delta\ndata: {"audio_data": "not base64!!"}\n\n',
            headers={"content-type": "text/event-stream"},
        )
    )

    with pytest.raises(HomeAssistantError) as raised:
        await _collect(_entity(hass), "The kitchen light is on, and the hall is dark.")

    assert raised.value.translation_key == "speech_not_base64"


@respx.mock
async def test_streaming_without_a_voice_says_nothing(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """An entity with no voice ends the stream instead of calling the API.

    The endpoint refuses a request with no voice, and that 400 reaches the
    user as silence either way -- but not spending the request, and logging
    the reason, is the difference between silence and unexplained silence.
    """
    entity = _entity(hass)
    hass.config_entries.async_update_subentry(
        init_integration, entity.subentry, data={CONF_MODEL: TTS_MODEL}
    )
    await hass.async_block_till_done()

    _, audio = await _collect(_entity(hass), "The kitchen light is on.")

    assert audio == b""
    assert not respx.calls


async def test_streaming_input_is_advertised(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """Home Assistant only streams text in when the entity says it can.

    Detected from the override rather than declared, so this asserts the
    method is actually overridden -- inheriting it silently would put the
    entity back on the joined-message shim.
    """
    assert _entity(hass).async_supports_streaming_input() is True


@respx.mock
async def test_an_event_split_across_data_lines_is_reassembled(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Server-sent events may split one payload over several data lines.

    The client joins them with newlines, so a server can only break the line
    where a newline is valid JSON whitespace -- between tokens, never inside
    the base64 string, which would corrupt it. That bounds the risk but does
    not remove it: the first version of this treated every data line as a
    whole JSON document, so a payload broken between fields would have failed
    to parse and been dropped, losing audio quietly.

    Every response seen from this endpoint so far is single-line. The format
    does not promise that.
    """
    delta = _delta(b"ID3 split across lines")
    body = (
        b"event: speech.audio.delta\n"
        + f'data: {{"type": "{delta["type"]}",\n'.encode()
        + f'data:  "audio_data": "{delta["audio_data"]}"}}\n'.encode()
        + b"\n"
        + _sse(DONE)
    )

    respx.post("https://api.mistral.ai/v1/audio/speech").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )

    _, audio = await _collect(
        _entity(hass), "The kitchen light is on and the hall light is off."
    )

    assert audio == b"ID3 split across lines"


@respx.mock
async def test_a_dropped_connection_is_reported_as_a_speech_error(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A transport failure becomes an error about speech, not a raw traceback.

    The non-streaming path catches these; this one did not, so a connection
    reset part way through a reply surfaced as an httpx exception from inside
    an async generator.
    """
    respx.post("https://api.mistral.ai/v1/audio/speech").mock(
        side_effect=httpx.ConnectError("connection reset")
    )

    with pytest.raises(HomeAssistantError) as raised:
        await _collect(_entity(hass), "The kitchen light is on, and the hall is dark.")

    assert raised.value.translation_key == "speech_transport_error"


@respx.mock
async def test_a_stream_that_produces_no_audio_is_reported(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Parsing cleanly and yielding nothing is a failure, not a quiet success.

    Reachable if the event shape moves under us -- an audio field renamed,
    say -- which is exactly when nobody is looking for it. Without this the
    entity would return an empty stream and the speaker would simply stay
    silent.
    """
    respx.post("https://api.mistral.ai/v1/audio/speech").mock(
        return_value=httpx.Response(
            200,
            content=_sse({"type": "speech.audio.delta", "audio": "renamed"}, DONE),
            headers={"content-type": "text/event-stream"},
        )
    )

    with pytest.raises(HomeAssistantError) as raised:
        await _collect(_entity(hass), "The kitchen light is on, and the hall is dark.")

    assert raised.value.translation_key == "speech_no_audio"


@respx.mock
async def test_an_unreadable_event_is_skipped_loudly(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """One bad event does not sink the reply, but it is not silent either.

    Skipped rather than raised, because an unfamiliar event that carries no
    audio is harmless. The case that matters -- ending up with no audio at
    all -- is caught separately, so this can afford to be forgiving.
    """
    body = b"event: speech.audio.delta\ndata: {not json at all\n\n" + _sse(
        _delta(b"ID3 good"), DONE
    )
    respx.post("https://api.mistral.ai/v1/audio/speech").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )

    with caplog_at_warning() as records:
        _, audio = await _collect(
            _entity(hass), "The kitchen light is on and the hall light is off."
        )

    assert audio == b"ID3 good"
    assert any("Could not read" in message for message in records)


def test_every_generator_in_the_speech_chain_is_closed() -> None:
    """No async generator in tts.py is left to the garbage collector.

    The speech path is five generators deep -- message_gen, _sentences,
    _async_stream_speech, _speech_events, aiter_lines -- and any one of them
    abandoned mid-iteration is finalised by the collector's hook, which
    schedules the close as a task and holds an HTTP response open until it
    runs. Home Assistant reports that as a lingering task.

    Asserted structurally because the runtime symptom is not reachable here:
    it needs Python 3.14 with the newest Home Assistant, and even there it
    only appears when the collector happens to run late, which made it
    intermittent. Two rounds of fixing this closed four of the five, and a
    green run was mistaken for proof each time.

    Every `async for` in this module must therefore iterate a name bound by an
    `async with aclosing(...)`, which is checkable without running anything.
    """
    source = Path(tts_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    closed: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncWith):
            continue
        for item in node.items:
            call = item.context_expr
            if (
                isinstance(call, ast.Call)
                and getattr(call.func, "id", None) == "aclosing"
                and isinstance(item.optional_vars, ast.Name)
            ):
                closed.add(item.optional_vars.id)

    unclosed = [
        ast.unparse(node.iter)
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFor)
        and not (isinstance(node.iter, ast.Name) and node.iter.id in closed)
    ]

    assert not unclosed, (
        "these async iterations are not wrapped in aclosing, so the generator "
        f"is left to the garbage collector: {unclosed}"
    )


@respx.mock
async def test_abandoning_the_speech_stream_leaves_no_finaliser_tasks(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Stopping part way through leaves nothing for the garbage collector.

    An async generator abandoned rather than closed is finalised by asyncio's
    hook, which schedules the close as a *task*. Home Assistant reports that
    as a lingering task, and it holds the HTTP response open until it runs.

    Three attempts at this closed our own generators and none of them fixed
    it, because the abandonment is inside httpx: `aiter_lines` iterates
    `aiter_text`, which iterates `aiter_bytes`, both with bare `async for`
    loops. Closing the outer one does not close the inner two, and nothing we
    close on our side reaches them.

    Measured on a streamed response, per iterator: abandoning after one line
    leaves one finaliser task, draining to the end leaves none -- and that
    holds for `aiter_bytes` as well, so choosing a different iterator does
    not help. Consuming the rest is the only thing that does, which is what
    _async_stream_speech now does in a finally.

    Counting tasks reproduces here, unlike the Home Assistant check that
    found it, which needs Python 3.14.2 and a collector that has already run.
    That is the point of this test: the earlier fixes were called verified on
    the strength of one green CI run each.
    """
    respx.post("https://api.mistral.ai/v1/audio/speech").mock(
        return_value=httpx.Response(
            200,
            content=_sse(*([_delta(b"chunk")] * 20), DONE),
            headers={"content-type": "text/event-stream"},
        )
    )

    response = await _entity(hass).async_stream_tts_audio(
        tts.TTSAudioRequest(
            language="en",
            options={},
            message_gen=_gen("The kitchen light is on and the hall light is off."),
        )
    )

    before = asyncio.all_tasks()

    # Take one chunk and walk away, which is what an error part way through
    # amounts to.
    async with aclosing(response.data_gen) as data:
        async for _chunk in data:
            break

    gc.collect()
    await asyncio.sleep(0)

    lingering = [
        task
        for task in asyncio.all_tasks() - before
        if "athrow" in repr(task.get_coro()) or "aclose" in repr(task.get_coro())
    ]
    assert not lingering, f"generator finalisers left pending: {lingering}"
