"""The requests this integration builds are accepted by the live API.

Status codes and the presence of fields, never what a model said. A test that
asserts on generated text fails on sampling drift, gets muted, and takes the
useful assertions with it.

Every test here guards a failure that actually happened, or a bound that was
measured rather than read. Where a case exists because of a specific bug, the
docstring says which -- a contract test whose reason has been forgotten is one
that gets deleted the first time it is inconvenient.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from .conftest import MAX_TOKENS, MODEL, NON_REASONING_MODEL, TTS_MODEL

HELLO: list[dict[str, str]] = [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


async def test_a_rejected_key_is_401_not_403(client: httpx.AsyncClient) -> None:
    """Authentication failures are 401, so 403 must not start a reauth flow.

    The integration treated 401 and 403 alike and asked for a new key on
    either. 403 means the key authenticated and the account is not permitted,
    which no replacement key fixes -- worst inside the reauth dialog, where
    entering a good key was told it was bad with no way out.

    If Mistral ever moves authentication failures to 403 this fails, and the
    handling in _convert_error, __init__ and config_flow has to move with it.
    """
    response = await client.get(
        "/models", headers={"Authorization": "Bearer not-a-real-key-0000"}
    )

    assert response.status_code == 401


# --------------------------------------------------------------------------
# Chat completions
# --------------------------------------------------------------------------


async def test_chat_completion_has_the_fields_we_read(post: Callable) -> None:
    """The response carries the path _transform_stream walks."""
    response = await post(
        "/chat/completions",
        {"model": MODEL, "messages": HELLO, "max_tokens": MAX_TOKENS},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "content" in body["choices"][0]["message"]


async def test_chat_completion_streams_deltas(client: httpx.AsyncClient) -> None:
    """Streaming yields the delta shape the chat log is fed from."""
    payload = {
        "model": MODEL,
        "messages": HELLO,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }

    async with client.stream("POST", "/chat/completions", json=payload) as response:
        assert response.status_code == 200

        deltas = 0
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            if "delta" in json.loads(line[6:])["choices"][0]:
                deltas += 1

    assert deltas


@pytest.mark.parametrize(
    ("temperature", "accepted"),
    [(1.5, True), (1.6, False)],
)
async def test_chat_completion_temperature_ceiling(
    post: Callable, temperature: float, accepted: bool
) -> None:
    """Chat completions accepts up to 1.5 and no further.

    The slider offered 2.0, so its top quarter produced a 422 on every request
    that used it. MAX_TEMPERATURE encodes 1.5; this is what holds that honest.
    """
    response = await post(
        "/chat/completions",
        {
            "model": MODEL,
            "messages": HELLO,
            "max_tokens": MAX_TOKENS,
            "temperature": temperature,
        },
    )

    assert (response.status_code == 200) is accepted


async def test_structured_output_returns_parseable_json(post: Callable) -> None:
    """json_schema with strict returns JSON, which is what AI tasks parse."""
    response = await post(
        "/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "A room and a temperature."}],
            "max_tokens": 80,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "reading",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "room": {"type": "string"},
                            "temp": {"type": "number"},
                        },
                        "required": ["room", "temp"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        },
    )

    assert response.status_code == 200
    # Parsed, not inspected. What the values are is the model's business; that
    # they parse at all is the contract ai_task.py depends on.
    assert json.loads(response.json()["choices"][0]["message"]["content"])


async def test_a_function_tool_comes_back_as_a_tool_call(post: Callable) -> None:
    """Tool calls arrive in the shape the tool loop dispatches from."""
    response = await post(
        "/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Turn on the kitchen light."}],
            "max_tokens": 120,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "turn_on",
                        "description": "Turn on a light",
                        "parameters": {
                            "type": "object",
                            "properties": {"area": {"type": "string"}},
                            "required": ["area"],
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    calls = response.json()["choices"][0]["message"]["tool_calls"]
    assert calls[0]["function"]["name"] == "turn_on"
    # Streamed responses always deliver this as a JSON string, so it is parsed
    # rather than read -- _parse_arguments exists for exactly that.
    assert json.loads(calls[0]["function"]["arguments"])


# --------------------------------------------------------------------------
# Conversations -- where the connectors run, and where the bugs were
# --------------------------------------------------------------------------


def _conversation(**extra: Any) -> dict[str, Any]:
    """Return a model-based conversation request, as the integration sends."""
    return {
        "model": MODEL,
        "inputs": HELLO,
        "store": False,
        "completion_args": {"max_tokens": MAX_TOKENS},
        **extra,
    }


async def test_handoff_execution_is_rejected_with_a_model(post: Callable) -> None:
    """The regression guard for the bug that broke web search entirely.

    The integration sent handoff_execution="client" alongside a model, and the
    endpoint refuses that combination, so every web search turn returned 422
    from the day the feature shipped.

    Nothing else could have caught it. The SDK accepts the argument, the
    OpenAPI schema declares the field on the request base both variants
    inherit, and a MagicMock accepts any keyword -- signature, spec and mock
    all agreed with each other and with nothing.

    Asserted as a rejection, so that if Mistral ever permits it this fails and
    somebody decides deliberately whether to start sending it again.
    """
    response = await post("/conversations", _conversation(handoff_execution="client"))

    assert response.status_code == 422
    assert "handoff_execution" in response.text


async def test_a_model_conversation_is_accepted(post: Callable) -> None:
    """The control for the test above: the same request without the field."""
    response = await post("/conversations", _conversation())

    assert response.status_code == 200
    assert response.json()["outputs"]


async def test_web_search_connector_is_accepted(post: Callable) -> None:
    """A connector runs here and only here -- chat completions cannot carry it."""
    response = await post(
        "/conversations", _conversation(tools=[{"type": "web_search"}])
    )

    assert response.status_code == 200


async def test_a_function_tool_is_handed_back_by_a_model_conversation(
    post: Callable,
) -> None:
    """Home Assistant tools still come back without handoff_execution.

    This is what made removing that field safe rather than merely necessary:
    client-side handoff is what a model-based conversation does anyway, so
    asking for it explicitly bought nothing and cost the whole feature.
    """
    response = await post(
        "/conversations",
        {
            "model": MODEL,
            "inputs": [{"role": "user", "content": "Turn on the kitchen light."}],
            "instructions": "You control a smart home. Use the provided tools.",
            "store": False,
            "completion_args": {"max_tokens": 120},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "turn_on",
                        "description": "Turn on a light",
                        "parameters": {
                            "type": "object",
                            "properties": {"area": {"type": "string"}},
                            "required": ["area"],
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    kinds = [entry.get("type") for entry in response.json()["outputs"]]
    assert "function.call" in kinds


@pytest.mark.parametrize(
    ("temperature", "accepted"),
    [(1.0, True), (1.2, False)],
)
async def test_completion_args_temperature_ceiling(
    post: Callable, temperature: float, accepted: bool
) -> None:
    """This endpoint stops at 1.0, where chat completions allows 1.5.

    The divergence is the point. Web search moves a conversation agent from
    one endpoint to the other, and it is a checkbox on the same form, so a
    temperature of 1.2 worked until an unrelated setting was switched on.
    Conversation agents are capped at the lower of the two because of this.
    """
    response = await post(
        "/conversations",
        {
            "model": MODEL,
            "inputs": HELLO,
            "store": False,
            "completion_args": {"max_tokens": MAX_TOKENS, "temperature": temperature},
        },
    )

    assert (response.status_code == 200) is accepted


async def test_completion_args_rejects_unknown_fields(post: Callable) -> None:
    """The strictness every other conclusion about this endpoint rests on.

    Because completion_args refuses a field it does not know, a rejection
    means a parameter is genuinely absent rather than merely unvalidated. If
    it ever started accepting unknown keys, every "this endpoint does not
    support X" finding would need re-checking -- silently, since the requests
    would start succeeding.
    """
    response = await post(
        "/conversations",
        {
            "model": MODEL,
            "inputs": HELLO,
            "store": False,
            "completion_args": {"max_tokens": MAX_TOKENS, "not_a_real_parameter": 1},
        },
    )

    assert response.status_code == 422


# --------------------------------------------------------------------------
# Audio and models
# --------------------------------------------------------------------------


async def test_voice_listing_pages_and_reports_a_total(
    client: httpx.AsyncClient,
) -> None:
    """Voices are paged, ten at a time by default.

    Listing without a limit showed the first ten voices of an account that had
    thirty-one, with no error and no way to tell. Both halves are asserted: the
    small default that caused it, and the total that proves a page is partial.
    """
    default = await client.get("/audio/voices")
    assert default.status_code == 200
    body = default.json()

    assert body["page_size"] == 10
    assert "total" in body

    full = await client.get("/audio/voices", params={"limit": 100})
    assert full.status_code == 200
    assert len(full.json()["items"]) >= len(body["items"])


async def test_speech_requires_a_voice(post: Callable) -> None:
    """The speech endpoint will not choose a voice for you.

    The spec disagrees -- it marks only `input` as required -- and so did this
    integration, in three separate comments. A request without one is a 400:

        Either ref_audio or voice must be provided.

    That matters because the voice field is only offered when the account has
    voices to list, and async_list_voices returns an empty list on any failure
    to list them. An entity configured while that call was failing stores no
    voice, sends no voice, and every attempt to speak gets a 400 that
    async_get_tts_audio turns into silence.
    """
    response = await post(
        "/audio/speech",
        {"input": "Hello.", "model": TTS_MODEL, "response_format": "mp3"},
    )

    assert response.status_code == 400
    assert "voice" in response.text


async def test_speech_returns_base64_audio_not_bytes(
    client: httpx.AsyncClient, post: Callable
) -> None:
    """The endpoint answers with JSON carrying base64, not an audio body.

    tts.py decodes audio_data strictly, and handing Home Assistant a string or
    the wrong bytes produces silence at playback rather than an error, so the
    shape matters more than it looks.
    """
    voices = await client.get("/audio/voices", params={"limit": 1})
    voice_id = voices.json()["items"][0]["id"]

    response = await post(
        "/audio/speech",
        {
            "input": "Hello.",
            "model": TTS_MODEL,
            "response_format": "mp3",
            "voice_id": voice_id,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    audio = base64.b64decode(response.json()["audio_data"], validate=True)
    assert audio[:3] == b"ID3" or audio[:2] == b"\xff\xfb"


async def test_transcription_accepts_a_wav_and_a_two_letter_language(
    client: httpx.AsyncClient,
) -> None:
    """Home Assistant's audio, wrapped as the endpoint expects.

    The language must be a two-letter code -- a three-letter one is rejected
    with "Invalid language alpha2 code". Every language in SPEECH_LANGUAGES is
    two letters today, so this holds that true rather than discovering it via
    a user whose pipeline stopped transcribing.
    """
    import io
    import struct
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

    response = await client.post(
        "/audio/transcriptions",
        files={"file": ("audio.wav", buffer.getvalue(), "audio/wav")},
        data={"model": "voxtral-mini-latest", "language": "en"},
    )

    assert response.status_code == 200
    assert "text" in response.json()


async def test_model_listing_reports_the_capabilities_we_filter_on(
    client: httpx.AsyncClient,
) -> None:
    """The dropdowns are built from these flags rather than from model names.

    Names come and go; the integration filters on capabilities so the speech
    platforms survive a Mistral release. That only holds while the flags are
    reported and keep their names.
    """
    response = await client.get("/models")

    assert response.status_code == 200
    models = response.json()["data"]
    assert models

    flags = {flag for model in models for flag in (model.get("capabilities") or {})}
    assert {"completion_chat", "audio_transcription", "audio_speech"} <= flags


# --------------------------------------------------------------------------
# Reasoning effort
# --------------------------------------------------------------------------


@pytest.mark.parametrize("effort", ["none", "high"])
async def test_reasoning_effort_accepts_the_values_we_offer(
    post: Callable, effort: str
) -> None:
    """The two values the selector offers are the two the API takes.

    The published enum declares six -- none, minimal, low, medium, high,
    xhigh -- and the 422 body names a seventh, max. The model layer accepts
    two. A selector built from either list would offer options that fail, so
    this pins the pair that were measured.
    """
    response = await post(
        "/chat/completions",
        {
            "model": MODEL,
            "messages": HELLO,
            "max_tokens": MAX_TOKENS,
            "reasoning_effort": effort,
        },
    )

    assert response.status_code == 200


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "xhigh", "max"])
async def test_the_values_we_do_not_offer_are_rejected(
    post: Callable, effort: str
) -> None:
    """The other five fail, which is why the selector is not built from the spec.

    The control half of the pair above. Without it, the test that "none" and
    "high" work would pass just as happily if the field were being ignored.
    """
    response = await post(
        "/chat/completions",
        {
            "model": MODEL,
            "messages": HELLO,
            "max_tokens": MAX_TOKENS,
            "reasoning_effort": effort,
        },
    )

    assert response.status_code == 400


async def test_a_model_without_the_capability_rejects_even_none(
    post: Callable,
) -> None:
    """The gate exists because "off" is not a safe thing to send.

    This is the measurement the whole design rests on. A model that does not
    advertise reasoning rejects *every* value, including "none" -- so the
    field cannot be sent unconditionally with "none" meaning off, and the
    config flow has to hide it and prune it instead.

    If Mistral ever makes "none" universally accepted this fails, and the
    capability gate in _subentry_schema and _prune_unsupported can go.
    """
    response = await post(
        "/chat/completions",
        {
            "model": NON_REASONING_MODEL,
            "messages": HELLO,
            "max_tokens": MAX_TOKENS,
            "reasoning_effort": "none",
        },
    )

    assert response.status_code == 400
    assert "not enabled for this model" in response.text


async def test_the_capability_flag_predicts_acceptance(
    client: httpx.AsyncClient,
) -> None:
    """capabilities.reasoning is what the config flow gates on, so it must hold.

    The flag is read off the model card and decides whether the field is
    offered at all. Asserted against the live listing rather than assumed,
    because a card that stopped reporting it would silently hide the setting
    for every model.
    """
    listing = await client.get("/models")
    assert listing.status_code == 200

    cards = {
        card["id"]: card
        for card in listing.json()["data"]
        if card["id"] in (MODEL, NON_REASONING_MODEL)
    }

    assert cards[MODEL]["capabilities"]["reasoning"] is True
    assert cards[NON_REASONING_MODEL]["capabilities"]["reasoning"] is False


async def test_reasoning_effort_is_accepted_inside_completion_args(
    post: Callable,
) -> None:
    """And on the conversations endpoint, where it nests.

    completion_args rejects unknown keys outright, so a 200 here is proof the
    field is parsed rather than tolerated -- which is what makes it safe to
    send from the web search path.
    """
    response = await post(
        "/conversations",
        {
            "model": MODEL,
            "inputs": "hi",
            "store": False,
            "completion_args": {
                "max_tokens": MAX_TOKENS,
                "reasoning_effort": "high",
            },
        },
    )

    assert response.status_code == 200


# --------------------------------------------------------------------------
# Conversations truncation
# --------------------------------------------------------------------------


async def test_a_truncated_conversation_spends_the_whole_budget(
    post: Callable,
) -> None:
    """Reaching the cap is how truncation is detected without a finish reason.

    No conversation event carries one -- the SDK has finish_reason only on the
    two chat-completions shapes -- so _transform_conversation_stream infers
    truncation from usage instead. That inference is only sound if a truncated
    response really does spend the cap exactly, which is what this pins.

    Reasoning on with a tiny cap is the reproducer from #139: the model thinks
    until the budget is gone and the content list holds only thinking, so the
    turn is "successful" with no answer in it.
    """
    response = await post(
        "/conversations",
        {
            "model": MODEL,
            "inputs": "Explain the history of the Roman empire in detail.",
            "store": False,
            "completion_args": {"max_tokens": 12, "reasoning_effort": "high"},
        },
    )

    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] >= 12


async def test_a_conversation_that_finishes_stays_under_the_budget(
    post: Callable,
) -> None:
    """The control, without which the test above proves nothing.

    If every response reported usage at the cap, inferring truncation from it
    would raise on every web search reply. A short answer with room to spare
    is what makes reaching the ceiling meaningful.
    """
    response = await post(
        "/conversations",
        {
            "model": MODEL,
            "inputs": "Say hi.",
            "store": False,
            "completion_args": {"max_tokens": 500},
        },
    )

    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] < 500


async def test_streamed_speech_is_an_event_stream_of_base64_deltas(
    client: httpx.AsyncClient,
) -> None:
    """The streamed shape the TTS entity parses by hand.

    The SDK has no method for this -- audio.speech offers complete_async and
    nothing else -- so tts.py builds the request and parses the response
    itself. Nothing else would notice if the event names, the base64 encoding
    or the content type moved.

    Asserted over the wire rather than against the spec: `stream` is declared
    in SpeechRequest, but so was `handoff_execution` on a request the endpoint
    rejects outright.
    """
    voices = await client.get("/audio/voices", params={"limit": 1})
    voice_id = voices.json()["items"][0]["id"]

    payload = {
        "input": "The kitchen light is on and the hallway light is off.",
        "model": TTS_MODEL,
        "response_format": "mp3",
        "voice_id": voice_id,
        "stream": True,
    }

    audio = b""
    events: list[str] = []

    async with client.stream("POST", "/audio/speech", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        async for line in response.aiter_lines():
            if line.startswith("event:"):
                events.append(line.removeprefix("event:").strip())
            elif line.startswith("data:"):
                event = json.loads(line.removeprefix("data:").strip())
                if encoded := event.get("audio_data"):
                    audio += base64.b64decode(encoded, validate=True)

    assert "speech.audio.delta" in events
    # The terminator, which carries no audio and must not be read as a chunk.
    assert "speech.audio.done" in events
    # mp3, so the extension the entity reports is not a guess.
    assert audio.startswith(b"ID3")


async def test_streamed_speech_arrives_in_more_than_one_piece(
    client: httpx.AsyncClient,
) -> None:
    """The streamed response is genuinely incremental, not one blob in an event.

    This is what the latency win rests on: audio can start playing before the
    endpoint has finished generating it. A stream that delivered everything in
    a single delta would satisfy the shape test above while being no faster
    than complete_async, and the hand-written parser would be buying nothing.

    Deliberately not a timing assertion. Measured over five trials each while
    implementing, the medians were about 0.67s streamed against about 2.44s
    complete -- but individual samples overlapped, so any threshold here would
    flake. Counting deltas tests the mechanism rather than the weather.
    """
    voices = await client.get("/audio/voices", params={"limit": 1})
    voice_id = voices.json()["items"][0]["id"]

    payload = {
        # Long enough to be chunked. A short phrase comes back as one delta,
        # which is correct behaviour and would make this test meaningless.
        "input": (
            "The kitchen light is on. The hallway light is off. "
            "The porch light is on. The garage door is closed. "
            "The thermostat is set to twenty degrees in the living room."
        ),
        "model": TTS_MODEL,
        "response_format": "mp3",
        "voice_id": voice_id,
        "stream": True,
    }

    deltas = 0
    async with client.stream("POST", "/audio/speech", json=payload) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                event = json.loads(line.removeprefix("data:").strip())
                if event.get("audio_data"):
                    deltas += 1

    assert deltas > 1
