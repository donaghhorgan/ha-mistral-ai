"""Tests for the Mistral AI conversation entity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent
from mistralai.client.errors import SDKError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mistral_ai.const import (
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_CITATIONS,
    MAX_TOOL_ITERATIONS,
    SUBENTRY_TYPE_CONVERSATION,
    WEB_SEARCH_CITATIONS_PROMPT,
)
from custom_components.mistral_ai.entity import LAZY_SEARCH_TOOL_NAME

from .helpers import make_chunk, make_sdk_error, make_tool_call, stream_of

ENTITY_ID = "conversation.mistral_ai_conversation"


async def converse(
    hass: HomeAssistant, text: str = "hello"
) -> conversation.ConversationResult:
    """Send a sentence to the agent under test."""
    return await conversation.async_converse(
        hass, text, None, Context(), agent_id=ENTITY_ID
    )


def speech(result: conversation.ConversationResult) -> str:
    """Return the spoken text from a conversation result."""
    return result.response.speech["plain"]["speech"]


async def test_entity_created(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """The conversation subentry produces an entity."""
    state = hass.states.get(ENTITY_ID)
    assert state is not None
    assert state.attributes["supported_features"] == 0


async def test_supported_languages(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """The agent advertises support for every language."""
    entity = hass.data["entity_components"]["conversation"].get_entity(ENTITY_ID)
    assert entity.supported_languages == MATCH_ALL


async def test_simple_message(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A plain response is returned to the caller."""
    mock_client.chat.stream_async = AsyncMock(
        side_effect=stream_of(make_chunk(content="Hi, how can I help?"))
    )

    result = await converse(hass)

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert speech(result) == "Hi, how can I help?"


async def test_streamed_chunks_are_concatenated(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Content spread over several chunks is joined back together."""
    mock_client.chat.stream_async = AsyncMock(
        side_effect=stream_of(
            make_chunk(content="The kitchen "),
            make_chunk(content="light is "),
            make_chunk(content="on."),
        )
    )

    result = await converse(hass)

    assert speech(result) == "The kitchen light is on."


async def test_empty_stream(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A model that returns nothing degrades to an error, not an exception."""
    mock_client.chat.stream_async = AsyncMock(side_effect=stream_of())

    result = await converse(hass)

    assert result.response.response_type == intent.IntentResponseType.ERROR


async def test_model_options_are_sent(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Options configured on the subentry reach the API call.

    This is the regression test for settings being written to the subentry but
    read from the config entry, which silently discarded every one of them.
    """
    subentry = next(
        s
        for s in init_integration.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_CONVERSATION
    )
    hass.config_entries.async_update_subentry(
        init_integration,
        subentry,
        data={CONF_MODEL: "mistral-large-latest", CONF_TEMPERATURE: 0.1},
    )
    await hass.async_block_till_done()

    mock_client.chat.stream_async = AsyncMock(
        side_effect=stream_of(make_chunk(content="ok"))
    )

    await converse(hass)

    kwargs = mock_client.chat.stream_async.await_args.kwargs
    assert kwargs["model"] == "mistral-large-latest"
    assert kwargs["temperature"] == 0.1


async def test_tool_call(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A tool call is executed and its result fed back to the model."""
    subentry = next(
        s
        for s in init_integration.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_CONVERSATION
    )
    hass.config_entries.async_update_subentry(
        init_integration,
        subentry,
        data={CONF_MODEL: "mistral-small-latest", CONF_LLM_HASS_API: ["assist"]},
    )
    await hass.async_block_till_done()

    # First pass asks for a tool, second pass answers.
    streams = [
        stream_of(
            make_chunk(
                tool_calls=[
                    make_tool_call(
                        index=0,
                        call_id="call_1",
                        name="test_tool",
                        arguments='{"param": "value"}',
                    )
                ]
            )
        ),
        stream_of(make_chunk(content="Done.")),
    ]

    async def next_stream(*args, **kwargs):
        return await streams.pop(0)(*args, **kwargs)

    mock_client.chat.stream_async = AsyncMock(side_effect=next_stream)

    mock_tool = AsyncMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.parameters = vol.Schema({})
    mock_tool.async_call.return_value = {"result": "ok"}

    with patch(
        "homeassistant.helpers.llm.AssistAPI._async_get_tools",
        return_value=[mock_tool],
    ):
        result = await converse(hass, "turn on the light")

    assert speech(result) == "Done."

    # The tool actually ran, with parsed arguments rather than a raw string.
    mock_tool.async_call.assert_awaited_once()
    tool_input = mock_tool.async_call.await_args.args[1]
    assert tool_input.tool_name == "test_tool"
    assert tool_input.tool_args == {"param": "value"}

    # The second request carried the tool result back to the model.
    second_call = mock_client.chat.stream_async.await_args.kwargs
    roles = [message["role"] for message in second_call["messages"]]
    assert "tool" in roles


async def test_tool_call_arguments_split_across_chunks(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Tool arguments streamed as JSON fragments are reassembled.

    Mistral emits arguments a few characters at a time. Home Assistant runs a
    tool the moment it is yielded, so a partial one would be called with
    broken arguments.
    """
    subentry = next(
        s
        for s in init_integration.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_CONVERSATION
    )
    hass.config_entries.async_update_subentry(
        init_integration,
        subentry,
        data={CONF_MODEL: "mistral-small-latest", CONF_LLM_HASS_API: ["assist"]},
    )
    await hass.async_block_till_done()

    streams = [
        stream_of(
            make_chunk(
                tool_calls=[
                    make_tool_call(
                        index=0, call_id="call_1", name="test_tool", arguments='{"na'
                    )
                ]
            ),
            make_chunk(tool_calls=[make_tool_call(index=0, arguments='me": "kit')]),
            make_chunk(tool_calls=[make_tool_call(index=0, arguments='chen"}')]),
        ),
        stream_of(make_chunk(content="Done.")),
    ]

    async def next_stream(*args, **kwargs):
        return await streams.pop(0)(*args, **kwargs)

    mock_client.chat.stream_async = AsyncMock(side_effect=next_stream)

    mock_tool = AsyncMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.parameters = vol.Schema({})
    mock_tool.async_call.return_value = {"result": "ok"}

    with patch(
        "homeassistant.helpers.llm.AssistAPI._async_get_tools",
        return_value=[mock_tool],
    ):
        await converse(hass, "turn on the kitchen light")

    tool_input = mock_tool.async_call.await_args.args[1]
    assert tool_input.tool_args == {"name": "kitchen"}


async def test_control_feature_requires_llm_api(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """The CONTROL feature is only advertised when an LLM API is selected."""
    assert hass.states.get(ENTITY_ID).attributes["supported_features"] == 0

    subentry = next(
        s
        for s in init_integration.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_CONVERSATION
    )
    hass.config_entries.async_update_subentry(
        init_integration, subentry, data={CONF_LLM_HASS_API: ["assist"]}
    )
    await hass.async_block_till_done()

    assert (
        hass.states.get(ENTITY_ID).attributes["supported_features"]
        == conversation.ConversationEntityFeature.CONTROL
    )


@pytest.mark.parametrize(
    ("status_code", "translation_key"),
    [
        (401, "invalid_api_key"),
        (403, "forbidden"),
        (429, "rate_limited"),
        (500, "api_error"),
    ],
)
async def test_api_errors_are_reported(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_client: MagicMock,
    status_code: int,
    translation_key: str,
) -> None:
    """API failures surface as an error response rather than an exception.

    Asserted on the translation key rather than the wording, which is now
    translatable and so no longer the contract. The rendered English still
    reaches the user, and that is covered by the translation test.
    """
    mock_client.chat.stream_async = AsyncMock(side_effect=make_sdk_error(status_code))

    result = await converse(hass)

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert speech(result)


async def test_auth_error_starts_reauth(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A rejected key mid-conversation raises a reauth flow."""
    mock_client.chat.stream_async = AsyncMock(side_effect=make_sdk_error(401))

    await converse(hass)
    await hass.async_block_till_done()

    flows = hass.config_entries.flow.async_progress()
    assert [flow["context"]["source"] for flow in flows] == ["reauth"]


async def test_forbidden_does_not_start_reauth(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A 403 mid-conversation reports the refusal and asks for no new key.

    The pair to the test above, and the reason this one exists: 401 and 403
    were handled together, so a refusal the user could do nothing about
    arrived as "your key was rejected" plus a dialog demanding a replacement.
    Asserting the error alone would not have caught it -- both paths produce
    one -- so what is asserted here is the absence of the flow.
    """
    mock_client.chat.stream_async = AsyncMock(side_effect=make_sdk_error(403))

    result = await converse(hass)
    await hass.async_block_till_done()

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert speech(result)
    assert not hass.config_entries.flow.async_progress()


async def test_prompt_and_history_are_sent(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The system prompt and prior turns are included in the request."""
    mock_client.chat.stream_async = AsyncMock(
        side_effect=stream_of(make_chunk(content="First answer"))
    )
    first = await converse(hass, "first question")

    mock_client.chat.stream_async = AsyncMock(
        side_effect=stream_of(make_chunk(content="Second answer"))
    )
    await conversation.async_converse(
        hass,
        "second question",
        first.conversation_id,
        Context(),
        agent_id=ENTITY_ID,
    )

    messages = mock_client.chat.stream_async.await_args.kwargs["messages"]
    roles = [message["role"] for message in messages]

    assert roles[0] == "system"
    assert "first question" in json.dumps(messages)
    assert "First answer" in json.dumps(messages)
    assert messages[-1]["content"] == "second question"


async def test_transport_error_is_reported(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A network failure surfaces as an error response, not a traceback."""
    mock_client.chat.stream_async = AsyncMock(
        side_effect=httpx.ConnectError("no route to host")
    )

    result = await converse(hass)

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert "Error talking to Mistral AI" in speech(result)


async def test_timeout_is_reported(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A timeout surfaces as an error response."""
    mock_client.chat.stream_async = AsyncMock(side_effect=TimeoutError)

    result = await converse(hass)

    assert result.response.response_type == intent.IntentResponseType.ERROR


async def test_unexpected_sdk_error_is_reported(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """An SDK exception of an unfamiliar type still degrades gracefully.

    The Mistral SDK adds exception types over time that inherit from neither
    SDKError nor httpx.HTTPError -- 2.8.0 added StreamDisconnectedError for
    mid-stream SSE errors -- and they are only reachable from private
    modules. Whatever the type, the user should get an error response rather
    than a traceback.
    """

    class StreamDisconnectedError(Exception):
        """Stand-in with the same shape: a plain Exception subclass."""

    mock_client.chat.stream_async = AsyncMock(
        side_effect=StreamDisconnectedError("stream closed unexpectedly")
    )

    result = await converse(hass)

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert "Unexpected error from Mistral AI" in speech(result)


async def test_home_assistant_error_is_not_rewrapped(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """An error we already raised keeps its own message."""
    mock_client.chat.stream_async = AsyncMock(
        side_effect=HomeAssistantError("something specific went wrong")
    )

    result = await converse(hass)

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert "something specific went wrong" in speech(result)
    assert "Unexpected error" not in speech(result)


def _entry(**fields: object) -> MagicMock:
    """Return an object with exactly these attributes and no others.

    spec matters: every attribute of a bare MagicMock exists and is truthy, so
    a reader that decides what to do by probing for attributes would take
    every branch at once.
    """
    entry = MagicMock(spec=list(fields))
    for name, value in fields.items():
        setattr(entry, name, value)
    return entry


def _event(**fields: object) -> MagicMock:
    """Return one conversation stream event wrapping a data payload."""
    event = MagicMock(spec=["data"])
    event.data = _entry(**fields)
    return event


def _usage(completion_tokens: int) -> MagicMock:
    """Return the usage payload the closing conversation event carries."""
    usage = MagicMock(spec=["completion_tokens"])
    usage.completion_tokens = completion_tokens
    return usage


def _thinking_chunk(text: str) -> MagicMock:
    """Return a content chunk holding reasoning rather than an answer."""
    thought = MagicMock(spec=["type", "text"])
    thought.type = "text"
    thought.text = text
    chunk = MagicMock(spec=["type", "thinking"])
    chunk.type = "thinking"
    chunk.thinking = [thought]
    return chunk


def _conversation_stream(*events: MagicMock):
    """Return a side effect producing a fresh event stream on each call.

    One per tool-calling iteration, since an async generator cannot be
    consumed twice.
    """

    async def _side_effect(*args: object, **kwargs: object):
        async def _stream():
            for event in events:
                yield event

        return _stream()

    return _side_effect


def _enable_web_search(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    mock_client: MagicMock,
    tier: str = "web_search",
    **extra: object,
) -> None:
    """Turn web search on for the conversation subentry.

    Also arms the lazy stand-in tool to get "called" on the very next chat-
    completions turn, so every test written against the old unconditional
    connector still reaches beta.conversations.start_stream_async exactly as
    it did before lazy escalation existed -- see
    _async_detect_search_escalation. A test of the lazy path itself
    overwrites mock_client.chat.stream_async afterwards, which is exactly
    what test_web_search_skips_the_connector_when_search_is_not_needed and
    its neighbours do.
    """
    subentry = next(
        s
        for s in entry.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_CONVERSATION
    )
    hass.config_entries.async_update_subentry(
        entry,
        subentry,
        data={CONF_MODEL: "mistral-small-latest", CONF_WEB_SEARCH: tier, **extra},
    )
    mock_client.chat.stream_async = AsyncMock(
        side_effect=stream_of(
            make_chunk(
                tool_calls=[
                    make_tool_call(call_id="lazy-1", name=LAZY_SEARCH_TOOL_NAME)
                ],
                finish_reason="tool_calls",
            )
        )
    )


async def test_web_search_escalates_to_the_conversations_endpoint(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A turn that calls the lazy stand-in escalates to the real connector.

    Connectors do not run on chat completions: the endpoint accepts the tool
    and then never runs it, because its responses have nowhere to carry what
    a connector returns -- so calling the stand-in there is only ever a
    signal, never the search itself. See #176.
    """
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_conversation_stream(
            _event(type="message.output.delta", content="It is raining.")
        )
    )
    _enable_web_search(hass, init_integration, mock_client)
    await hass.async_block_till_done()

    assert speech(await converse(hass, "what is the weather")) == "It is raining."

    # Exactly one detection round trip, not a retry loop.
    mock_client.chat.stream_async.assert_awaited_once()

    request = mock_client.beta.conversations.start_stream_async.await_args.kwargs
    assert {"type": "web_search"} in request["tools"]

    # The endpoint retains conversations by default, where chat completions
    # stores nothing, so this is set rather than inherited.
    assert request["store"] is False

    # No system role exists on this endpoint; the prompt goes to instructions.
    assert request["instructions"]
    assert all(entry.get("role") != "system" for entry in request["inputs"])

    # handoff_execution is rejected by the endpoint whenever the request
    # carries a model rather than an agent_id, and this used to send it:
    #
    #   422 Conversation with a 'model' can't contain the following fields
    #       handoff_execution
    #
    # So every web search turn failed. Nothing asserted the request shape
    # beyond the fields above, and a mock accepts any keyword, which is how it
    # shipped. Asserted as an absence because that is what the API requires.
    assert "handoff_execution" not in request


async def test_web_search_is_off_by_default(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Nothing is sent, and nothing changes path, unless the tier is chosen.

    Mistral bills per search, so this must never be on by accident.
    """
    await converse(hass)

    mock_client.chat.stream_async.assert_awaited()
    assert "tools" not in mock_client.chat.stream_async.await_args.kwargs


async def test_web_search_skips_the_connector_when_search_is_not_needed(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A turn that never calls the stand-in never touches the connector.

    This is the point of lazy escalation -- #176: the connector cost every
    turn ~0.7s of time-to-first-token to serve a feature roughly 1 turn in 40
    actually needed. Verified against the real API before this shipped: the
    stand-in was never called on trivial, small-talk, arithmetic, general
    knowledge or plain Home-Assistant-tool probes, only on ones that actually
    needed a search.
    """
    _enable_web_search(hass, init_integration, mock_client)
    await hass.async_block_till_done()
    mock_client.chat.stream_async = AsyncMock(
        side_effect=stream_of(make_chunk(content="Yes."))
    )

    assert speech(await converse(hass, "say yes")) == "Yes."

    mock_client.chat.stream_async.assert_awaited_once()
    mock_client.beta.conversations.start_stream_async.assert_not_called()


class _FakeEventStream:
    """Mimics EventStreamAsync's shape: chunks plus a closeable .response.

    The mocks built by stream_of/make_stream are bare async generators, which
    cannot carry an arbitrary .response attribute -- CPython's async
    generator objects have no __dict__. This stands in wherever a test needs
    to assert on the underlying connection being closed.
    """

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.response = AsyncMock()

    def __aiter__(self):
        return self._generator()

    async def _generator(self):
        for chunk in self._chunks:
            yield chunk


async def test_web_search_escalation_closes_the_abandoned_detection_stream(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Escalating closes the detection request's connection, not just drops it.

    EventStreamAsync only closes the underlying response once its stream is
    read to the SSE sentinel or EOF. Escalating stops reading as soon as the
    stand-in's name is seen, well before either -- and escalating is the
    success case for a search turn, not an error path, so leaving the
    connection to whenever garbage collection gets to it would leak one on
    every turn that actually searches.
    """
    detection_stream = _FakeEventStream(
        [
            make_chunk(
                tool_calls=[
                    make_tool_call(call_id="lazy-1", name=LAZY_SEARCH_TOOL_NAME)
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_conversation_stream(
            _event(type="message.output.delta", content="It is raining.")
        )
    )
    _enable_web_search(hass, init_integration, mock_client)
    await hass.async_block_till_done()
    mock_client.chat.stream_async = AsyncMock(return_value=detection_stream)

    assert speech(await converse(hass, "what is the weather")) == "It is raining."

    detection_stream.response.aclose.assert_awaited_once()


async def test_web_search_lazy_tool_is_offered_alongside_real_tools(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The detection request carries the stand-in next to real HA tools."""
    _enable_web_search(
        hass, init_integration, mock_client, **{CONF_LLM_HASS_API: ["assist"]}
    )
    await hass.async_block_till_done()
    mock_client.chat.stream_async = AsyncMock(
        side_effect=stream_of(make_chunk(content="Done."))
    )

    mock_tool = AsyncMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.parameters = vol.Schema({})

    with patch(
        "homeassistant.helpers.llm.AssistAPI._async_get_tools",
        return_value=[mock_tool],
    ):
        await converse(hass, "turn on the light")

    tools = mock_client.chat.stream_async.await_args.kwargs["tools"]
    names = {tool["function"]["name"] for tool in tools}
    assert names == {"test_tool", LAZY_SEARCH_TOOL_NAME}


async def test_web_search_escalation_discards_a_combined_detection_call(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A real tool called alongside the stand-in is never run from detection.

    Verified against the real API: asking for both a Home Assistant action
    and a search in the same sentence calls both tools in the same turn, in
    either order. Whichever one it is, detection only ever reads the names --
    it never dispatches anything, so the real tool has to run again, for
    real, once the turn hands off to the endpoint that can actually search.
    """
    streams = [
        _conversation_stream(
            _event(
                type="function.call.delta",
                tool_call_id="call-1",
                name="test_tool",
                arguments='{"param": "value"}',
            ),
        ),
        _conversation_stream(
            _event(type="message.output.delta", content="Done, and it is raining.")
        ),
    ]

    async def _next_stream(*args: object, **kwargs: object):
        return await streams.pop(0)(*args, **kwargs)

    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_next_stream
    )
    _enable_web_search(
        hass, init_integration, mock_client, **{CONF_LLM_HASS_API: ["assist"]}
    )
    await hass.async_block_till_done()
    # _enable_web_search armed a plain detection stub; this test needs its own.
    mock_client.chat.stream_async = AsyncMock(
        side_effect=stream_of(
            make_chunk(
                tool_calls=[
                    make_tool_call(index=0, call_id="ha-1", name="test_tool"),
                    make_tool_call(
                        index=1, call_id="lazy-1", name=LAZY_SEARCH_TOOL_NAME
                    ),
                ],
                finish_reason="tool_calls",
            )
        )
    )

    mock_tool = AsyncMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.parameters = vol.Schema({})
    mock_tool.async_call.return_value = {"result": "ok"}

    with patch(
        "homeassistant.helpers.llm.AssistAPI._async_get_tools",
        return_value=[mock_tool],
    ):
        result = await converse(hass, "turn on the light and check the weather")

    assert speech(result) == "Done, and it is raining."
    # Exactly once: detection saw the name but never ran it.
    mock_tool.async_call.assert_awaited_once()


async def test_web_search_falls_back_when_a_real_tool_claims_the_lazy_name(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A real tool named the same as the stand-in disables lazy escalation.

    An edge case worth not colliding with rather than one worth solving
    cleverly -- the turn falls back to what every web-search turn did before
    lazy escalation existed: the connector attached unconditionally, with no
    detection round trip at all.
    """
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_conversation_stream(
            _event(type="message.output.delta", content="It is raining.")
        )
    )
    _enable_web_search(
        hass, init_integration, mock_client, **{CONF_LLM_HASS_API: ["assist"]}
    )
    await hass.async_block_till_done()
    mock_client.chat.stream_async.reset_mock()

    mock_tool = AsyncMock()
    mock_tool.name = LAZY_SEARCH_TOOL_NAME
    mock_tool.description = "A user-defined tool that happens to share a name"
    mock_tool.parameters = vol.Schema({})

    with patch(
        "homeassistant.helpers.llm.AssistAPI._async_get_tools",
        return_value=[mock_tool],
    ):
        result = await converse(hass, "what is the weather")

    assert speech(result) == "It is raining."
    mock_client.chat.stream_async.assert_not_awaited()
    mock_client.beta.conversations.start_stream_async.assert_awaited_once()


async def test_web_search_agent_still_streams(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Enabling web search does not cost the agent its streaming.

    It did in the first cut, because the conversations endpoint reports a
    different event shape. An agent that silently stopped streaming while
    every other one continued was a surprise worth removing.
    """
    component = hass.data["conversation"]
    assert component.get_entity(ENTITY_ID).supports_streaming is True

    _enable_web_search(hass, init_integration, mock_client)
    await hass.async_block_till_done()

    assert component.get_entity(ENTITY_ID).supports_streaming is True


async def test_web_search_streams_the_reply_in_pieces(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Text deltas are concatenated rather than one of them winning."""
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_conversation_stream(
            _event(type="conversation.response.started"),
            _event(type="message.output.delta", content="It is "),
            _event(type="message.output.delta", content="raining in Dublin."),
            _event(type="conversation.response.done"),
        )
    )
    _enable_web_search(hass, init_integration, mock_client)
    await hass.async_block_till_done()

    assert speech(await converse(hass, "weather")) == "It is raining in Dublin."


async def test_web_search_error_event_becomes_an_error(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A failure arriving as an event is raised rather than ignored.

    Chat completions raises; here the failure is just another item in the
    stream, so a reader that only looked for content would report success
    having said nothing.
    """
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_conversation_stream(
            _event(type="conversation.response.started"),
            _event(
                type="conversation.response.error",
                message="connector unavailable",
                code=1800,
            ),
        )
    )
    _enable_web_search(hass, init_integration, mock_client)
    await hass.async_block_till_done()

    result = await converse(hass, "weather")

    assert result.response.response_type == intent.IntentResponseType.ERROR
    # Named specifically, because an ignored error event produces an empty
    # stream, and an empty stream is also an error response -- asserting only
    # on the type would pass without the event being read at all.
    assert "connector unavailable" in speech(result)
    assert "1800" in speech(result)


async def test_web_search_runs_home_assistant_tools(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A connector and a Home Assistant tool coexist in one turn.

    Mistral executes the connector itself and hands back a function call for
    anything Home Assistant owns, so the loop has to run that and send the
    result on. This is the composition question #65 opened with.
    """
    streams = [
        _conversation_stream(
            _event(type="tool.execution.started", name="web_search", arguments="{}"),
            # Arguments arrive in fragments, as they do on chat completions.
            _event(
                type="function.call.delta",
                tool_call_id="call-1",
                name="test_tool",
                arguments='{"param": ',
            ),
            _event(
                type="function.call.delta",
                tool_call_id="call-1",
                name="test_tool",
                arguments='"value"}',
            ),
        ),
        _conversation_stream(
            _event(type="message.output.delta", content="Done, and it is raining.")
        ),
    ]

    async def _next_stream(*args: object, **kwargs: object):
        return await streams.pop(0)(*args, **kwargs)

    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_next_stream
    )
    _enable_web_search(
        hass, init_integration, mock_client, **{CONF_LLM_HASS_API: ["assist"]}
    )
    await hass.async_block_till_done()

    mock_tool = AsyncMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.parameters = vol.Schema({})
    mock_tool.async_call.return_value = {"result": "ok"}

    with patch(
        "homeassistant.helpers.llm.AssistAPI._async_get_tools",
        return_value=[mock_tool],
    ):
        result = await converse(hass, "turn on the light and check the weather")

    assert speech(result) == "Done, and it is raining."
    assert mock_tool.async_call.await_args.args[1].tool_name == "test_tool"

    # The second pass carries the tool result back as its own entry, because
    # store is off and the endpoint is holding no history for us.
    second = mock_client.beta.conversations.start_stream_async.await_args_list[1].kwargs
    results = [e for e in second["inputs"] if e.get("type") == "function.result"]
    assert results and results[0]["tool_call_id"] == "call-1"


async def test_web_search_drops_a_phantom_tool_call(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A function.call.delta naming no real tool is dropped, not dispatched.

    Observed once on the web-search path: a tool_call_id whose name was a
    comma-joined run of other tool-call IDs and digits, belonging to neither
    Home Assistant nor the connector. Dispatching it can only fail ("Tool ...
    not found"), costing a full round trip while the model recovers -- #177.
    A real call in the same turn still has to go through.
    """
    streams = [
        _conversation_stream(
            _event(
                type="function.call.delta",
                tool_call_id="WyqrjjA9B",
                name="TJv1Kxiq,niCOZXrz,MPt03GYA,5,7,6,4,3,9,8,2,1,0",
                arguments="{}",
            ),
            _event(
                type="function.call.delta",
                tool_call_id="call-1",
                name="test_tool",
                arguments='{"param": "value"}',
            ),
        ),
        _conversation_stream(
            _event(type="message.output.delta", content="Done, and it is raining.")
        ),
    ]

    async def _next_stream(*args: object, **kwargs: object):
        return await streams.pop(0)(*args, **kwargs)

    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_next_stream
    )
    _enable_web_search(
        hass, init_integration, mock_client, **{CONF_LLM_HASS_API: ["assist"]}
    )
    await hass.async_block_till_done()

    mock_tool = AsyncMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.parameters = vol.Schema({})
    mock_tool.async_call.return_value = {"result": "ok"}

    with patch(
        "homeassistant.helpers.llm.AssistAPI._async_get_tools",
        return_value=[mock_tool],
    ):
        result = await converse(hass, "check the weather")

    assert speech(result) == "Done, and it is raining."
    assert mock_tool.async_call.await_count == 1
    assert mock_tool.async_call.await_args.args[1].tool_name == "test_tool"

    # Only the real call gets a result sent back. The phantom name never
    # reached Home Assistant, so nothing came back for it to report as
    # "not found" and burn a second round trip on.
    second = mock_client.beta.conversations.start_stream_async.await_args_list[1].kwargs
    results = [e for e in second["inputs"] if e.get("type") == "function.result"]
    assert [r["tool_call_id"] for r in results] == ["call-1"]


@pytest.mark.parametrize(
    "side_effect",
    [
        make_sdk_error(500),
        httpx.ConnectError("no route to host"),
        TimeoutError(),
        ValueError("something unexpected"),
    ],
)
async def test_web_search_api_errors_are_reported(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_client: MagicMock,
    side_effect: Exception,
) -> None:
    """A failure on the conversations endpoint reaches the user as an error.

    The same shapes the chat completions path handles, since this endpoint
    raises through the same SDK.
    """
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=side_effect
    )
    _enable_web_search(hass, init_integration, mock_client)
    await hass.async_block_till_done()

    result = await converse(hass, "what is the weather")

    assert result.response.response_type == intent.IntentResponseType.ERROR


async def test_web_search_unsupported_model_names_the_conflict(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A model that cannot take the connector says so, not the raw API body.

    The model picker offers models the conversations endpoint then refuses --
    #179. Every turn used to read the raw response body aloud:

        Error talking to Mistral AI: API error occurred: Status 400.
        Body: {"message":"Model ministral-8b-2512 currently does not support
        builtin connectors.", ...}

    The configured model is named from local state, not parsed out of the
    API's text, so this does not depend on Mistral's wording holding still.
    """
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=SDKError(
            "API error occurred",
            httpx.Response(
                400,
                text=(
                    '{"object":"Error","message":"Model ministral-8b-2512 '
                    'currently does not support builtin connectors.",'
                    '"type":"invalid_request_error"}'
                ),
            ),
        )
    )
    _enable_web_search(
        hass, init_integration, mock_client, **{CONF_MODEL: "ministral-8b-2512"}
    )
    await hass.async_block_till_done()

    result = await converse(hass, "what is the weather")

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert "ministral-8b-2512" in speech(result)
    assert "web search" in speech(result)
    # Not the raw body, which named a Mistral-internal field nobody here set.
    assert "builtin connector" not in speech(result)


async def test_web_search_citations_do_not_break_the_reply(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Search references are dropped and the sentence still reads.

    A searched reply interleaves text with tool_reference chunks carrying the
    sources, and the split can fall mid-sentence -- the live API returned the
    body, then two references, then a lone full stop. Those chunks have no
    text of their own, so they are skipped and the remaining pieces join back
    into the sentence the model wrote.

    They are dropped rather than rendered because this reply is also spoken by
    a voice assistant, where reading out URLs would be worse than useless.
    """
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_conversation_stream(
            _event(type="tool.execution.started", name="web_search", arguments="{}"),
            _event(
                type="message.output.delta",
                content=_entry(type="text", text="It is 21 degrees in Dublin"),
            ),
            _event(
                type="message.output.delta",
                content=_entry(
                    type="tool_reference",
                    tool="web_search",
                    title="Hour-by-Hour Forecast",
                    url="https://example.invalid/dublin",
                ),
            ),
            _event(type="message.output.delta", content=_entry(type="text", text=".")),
        )
    )
    _enable_web_search(hass, init_integration, mock_client)
    await hass.async_block_till_done()

    assert speech(await converse(hass, "weather")) == "It is 21 degrees in Dublin."


async def test_web_search_asks_the_model_to_name_its_sources(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The attribution instruction rides along on the system prompt.

    Default on, and asserted on a subentry that stores no value for it: the
    turn that switches web search on cannot have submitted one, because the
    checkbox is not on the form until the next time it is opened. Those agents
    still have to attribute, so the default is applied at request time rather
    than only in the form.
    """
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_conversation_stream(
            _event(type="message.output.delta", content="ok")
        )
    )
    _enable_web_search(hass, init_integration, mock_client)
    await hass.async_block_till_done()

    await converse(hass, "weather")

    request = mock_client.beta.conversations.start_stream_async.await_args.kwargs
    assert WEB_SEARCH_CITATIONS_PROMPT in request["instructions"]


async def test_web_search_citations_can_be_turned_off(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Nothing is appended once the option is cleared.

    The whole reason this is an option rather than a default prompt edit: a
    phrase on every searched reply is useful to some people and grating to
    others, and there has to be a way off it that does not mean rewriting the
    instructions.
    """
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_conversation_stream(
            _event(type="message.output.delta", content="ok")
        )
    )
    _enable_web_search(
        hass, init_integration, mock_client, **{CONF_WEB_SEARCH_CITATIONS: False}
    )
    await hass.async_block_till_done()

    await converse(hass, "weather")

    request = mock_client.beta.conversations.start_stream_async.await_args.kwargs
    assert WEB_SEARCH_CITATIONS_PROMPT not in (request["instructions"] or "")


async def test_web_search_citations_keep_the_configured_prompt(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The instruction is added to the user's prompt, not put in place of it.

    Appending to a string that the endpoint takes as one block is easy to get
    wrong in the direction that loses the original, and losing it is silent --
    the agent simply stops behaving as configured.
    """
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_conversation_stream(
            _event(type="message.output.delta", content="ok")
        )
    )
    _enable_web_search(
        hass, init_integration, mock_client, **{CONF_PROMPT: "Answer like a pirate."}
    )
    await hass.async_block_till_done()

    await converse(hass, "weather")

    instructions = mock_client.beta.conversations.start_stream_async.await_args.kwargs[
        "instructions"
    ]
    assert "Answer like a pirate." in instructions
    assert WEB_SEARCH_CITATIONS_PROMPT in instructions


async def test_web_search_citations_are_not_appended_twice(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A second tool-calling pass carries one copy, not two.

    The instructions are rebuilt from the chat log on every pass of the loop,
    so an implementation that appended to the stored prompt instead would
    accumulate a copy per round trip.
    """
    streams = [
        _conversation_stream(
            _event(
                type="function.call.delta",
                tool_call_id="call-1",
                name="test_tool",
                arguments="{}",
            )
        ),
        _conversation_stream(_event(type="message.output.delta", content="Done.")),
    ]

    async def _next_stream(*args: object, **kwargs: object):
        return await streams.pop(0)(*args, **kwargs)

    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_next_stream
    )
    _enable_web_search(
        hass, init_integration, mock_client, **{CONF_LLM_HASS_API: ["assist"]}
    )
    await hass.async_block_till_done()

    mock_tool = AsyncMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.parameters = vol.Schema({})
    mock_tool.async_call.return_value = {"result": "ok"}

    with patch(
        "homeassistant.helpers.llm.AssistAPI._async_get_tools",
        return_value=[mock_tool],
    ):
        await converse(hass, "turn on the light and check the weather")

    second = mock_client.beta.conversations.start_stream_async.await_args_list[1].kwargs
    assert second["instructions"].count(WEB_SEARCH_CITATIONS_PROMPT) == 1


async def test_no_citation_instruction_without_web_search(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """An agent that cannot search is not told to cite anything.

    It has nothing to cite, and the instruction would spend tokens on every
    request asking for attribution the model could only invent.
    """
    await converse(hass)

    messages = mock_client.chat.stream_async.await_args.kwargs["messages"]
    assert not any(
        WEB_SEARCH_CITATIONS_PROMPT in str(message.get("content", ""))
        for message in messages
    )


async def test_a_validation_error_reaches_the_user_readably(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The user is told which field is wrong, not shown the whole union.

    The end-to-end half of the parsing tests: a 422 has to travel through
    _convert_error and the translation to arrive as a sentence. Asserted on
    what the user actually sees, because that is the thing that was 2.5 kB of
    machinery and one useful line.
    """
    body = (Path(__file__).parent / "fixtures" / "conversations_422.json").read_text()
    mock_client.chat.stream_async = AsyncMock(
        side_effect=SDKError("boom", httpx.Response(422, text=body), body)
    )

    result = await converse(hass)

    spoken = speech(result)
    assert "temperature" in spoken
    assert "agent_id" not in spoken
    assert len(spoken) < 200


async def test_top_p_is_sent_on_the_chat_completions_path(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The configured value reaches the request rather than only being stored."""
    entry = next(
        subentry
        for subentry in init_integration.subentries.values()
        if subentry.subentry_type == "conversation"
    )
    hass.config_entries.async_update_subentry(
        init_integration, entry, data={**entry.data, CONF_TOP_P: 0.4}
    )
    await hass.async_block_till_done()

    await converse(hass)

    assert mock_client.chat.stream_async.await_args.kwargs["top_p"] == 0.4


async def test_top_p_is_sent_on_the_conversations_path(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """And on the other endpoint, where it lives inside completion_args.

    Worth its own test because the two paths take it in different places, and
    a setting that applies until web search is switched on is exactly the
    failure the temperature bounds already had to be fixed for.
    """
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_conversation_stream(
            _event(type="message.output.delta", content="ok")
        )
    )
    _enable_web_search(hass, init_integration, mock_client, **{CONF_TOP_P: 0.4})
    await hass.async_block_till_done()

    await converse(hass)

    request = mock_client.beta.conversations.start_stream_async.await_args.kwargs
    assert request["completion_args"]["top_p"] == 0.4


async def test_reasoning_effort_is_sent_on_the_chat_completions_path(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The configured effort reaches the request, top level on this endpoint."""
    entry = next(
        subentry
        for subentry in init_integration.subentries.values()
        if subentry.subentry_type == "conversation"
    )
    hass.config_entries.async_update_subentry(
        init_integration, entry, data={**entry.data, CONF_REASONING_EFFORT: "high"}
    )
    await hass.async_block_till_done()

    await converse(hass)

    assert mock_client.chat.stream_async.await_args.kwargs["reasoning_effort"] == "high"


async def test_reasoning_effort_is_sent_on_the_conversations_path(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """And inside completion_args on the other one.

    Both endpoints accept the same two values, so unlike temperature there is
    no per-endpoint ceiling -- but they take it in different places, which is
    the mistake this pair of tests exists to catch.
    """
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_conversation_stream(
            _event(type="message.output.delta", content="ok")
        )
    )
    _enable_web_search(
        hass, init_integration, mock_client, **{CONF_REASONING_EFFORT: "high"}
    )
    await hass.async_block_till_done()

    await converse(hass)

    request = mock_client.beta.conversations.start_stream_async.await_args.kwargs
    assert request["completion_args"]["reasoning_effort"] == "high"


async def test_reasoning_effort_is_absent_when_unset(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Nothing is sent unless it was chosen, and that is load-bearing.

    Every other generation parameter has a default sent on every request. This
    one must not: a model that does not advertise the capability rejects the
    field outright rather than ignoring it, so sending "none" to mean "off"
    would break every non-reasoning model. Absence is how those keep working.
    """
    await converse(hass)

    assert "reasoning_effort" not in mock_client.chat.stream_async.await_args.kwargs


async def test_web_search_reply_cut_off_before_answering_is_reported(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The web search path detects truncation too, without a finish reason.

    #131 fixed this on chat completions using finish_reason "length". No
    conversation event carries one -- the SDK has it only on the two
    chat-completions shapes -- so the same failure still arrived here as
    silence.

    What the endpoint does report is usage on the closing event, and a
    truncated response spends the cap exactly. Captured from a real request:
    max_tokens 12 with reasoning on returns completion_tokens 12 and a content
    list holding one thinking chunk.
    """
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_conversation_stream(
            _event(
                type="message.output.delta",
                content=_thinking_chunk("working through this"),
            ),
            _event(type="conversation.response.done", usage=_usage(12)),
        )
    )
    _enable_web_search(hass, init_integration, mock_client, **{CONF_MAX_TOKENS: 12})
    await hass.async_block_till_done()

    result = await converse(hass)

    assert result.response.response_type is intent.IntentResponseType.ERROR


async def test_web_search_keeps_a_truncated_partial_answer(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Half an answer is still worth more than an error here too.

    Spending the whole budget is only a failure when nothing was said. Cut off
    part way through a sentence, the user can see it was cut short and read
    what arrived.
    """
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_conversation_stream(
            _event(type="message.output.delta", content="The kitchen light is"),
            _event(type="conversation.response.done", usage=_usage(12)),
        )
    )
    _enable_web_search(hass, init_integration, mock_client, **{CONF_MAX_TOKENS: 12})
    await hass.async_block_till_done()

    result = await converse(hass)

    assert result.response.response_type is not intent.IntentResponseType.ERROR
    assert "kitchen light" in result.response.speech["plain"]["speech"]


async def test_web_search_does_not_report_a_reply_that_finished(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The control. A reply well inside the cap is not truncation.

    Without this, the test above would pass just as happily if every web
    search reply raised. Measured: an answer that finishes uses 13 tokens of
    a 500 or 3000 cap, so reaching the ceiling really does distinguish them.
    """
    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_conversation_stream(
            _event(type="message.output.delta", content="ok"),
            _event(type="conversation.response.done", usage=_usage(13)),
        )
    )
    _enable_web_search(hass, init_integration, mock_client, **{CONF_MAX_TOKENS: 3000})
    await hass.async_block_till_done()

    result = await converse(hass)

    assert result.response.response_type is not intent.IntentResponseType.ERROR


async def test_a_model_that_never_stops_calling_tools_is_reported(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Exhausting the tool loop says so instead of returning nothing.

    Both loops ran a fixed number of passes and broke when the model stopped
    asking for tools. Falling off the end was silent -- no log line, no error,
    and a chat log still holding tool results that were never sent back -- so
    the user got whatever fragmentary text came with the last tool call,
    usually nothing, indistinguishable from the model having nothing to say.

    A real API and a resolving tool are needed to reach it: without them the
    tool call is never dispatched, no result comes back, and the loop breaks
    after one pass rather than running out.
    """
    subentry = next(
        s
        for s in init_integration.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_CONVERSATION
    )
    hass.config_entries.async_update_subentry(
        init_integration,
        subentry,
        data={CONF_MODEL: "mistral-small-latest", CONF_LLM_HASS_API: ["assist"]},
    )
    await hass.async_block_till_done()

    # Never stops asking, so every pass produces another unresponded result.
    mock_client.chat.stream_async = AsyncMock(
        side_effect=stream_of(
            make_chunk(
                tool_calls=[
                    make_tool_call(
                        index=0, call_id="call_1", name="test_tool", arguments="{}"
                    )
                ]
            )
        )
    )

    mock_tool = AsyncMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.parameters = vol.Schema({})
    mock_tool.async_call.return_value = {"result": "ok"}

    with patch(
        "homeassistant.helpers.llm.AssistAPI._async_get_tools",
        return_value=[mock_tool],
    ):
        result = await converse(hass, "turn on the light")

    # The message, not just the fact of an error. Without the fix this is
    # still an error -- Home Assistant falls back to "Unable to get response"
    # once the chat log ends on a tool result -- so asserting the type alone
    # would pass either way and test nothing.
    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert (
        "did not finish within 10 rounds" in (result.response.speech["plain"]["speech"])
    )
    assert mock_client.chat.stream_async.await_count == MAX_TOOL_ITERATIONS


async def test_web_search_tool_loop_exhaustion_is_reported(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """And on the other loop, which had the same silence."""
    _enable_web_search(
        hass, init_integration, mock_client, **{CONF_LLM_HASS_API: ["assist"]}
    )
    await hass.async_block_till_done()

    mock_client.beta.conversations.start_stream_async = AsyncMock(
        side_effect=_conversation_stream(
            _event(
                type="function.call.delta",
                tool_call_id="call_1",
                name="test_tool",
                arguments="{}",
            ),
            _event(type="conversation.response.done", usage=_usage(5)),
        )
    )

    mock_tool = AsyncMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.parameters = vol.Schema({})
    mock_tool.async_call.return_value = {"result": "ok"}

    with patch(
        "homeassistant.helpers.llm.AssistAPI._async_get_tools",
        return_value=[mock_tool],
    ):
        result = await converse(hass, "turn on the light")

    assert result.response.response_type is intent.IntentResponseType.ERROR
    assert (
        "did not finish within 10 rounds" in (result.response.speech["plain"]["speech"])
    )
    assert (
        mock_client.beta.conversations.start_stream_async.await_count
        == MAX_TOOL_ITERATIONS
    )
