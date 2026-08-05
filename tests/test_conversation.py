"""Tests for the Mistral AI conversation entity."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mistral_ai.const import (
    CONF_MODEL,
    CONF_TEMPERATURE,
    CONF_WEB_SEARCH,
    SUBENTRY_TYPE_CONVERSATION,
)

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
    ("status_code", "expected"),
    [
        (401, "Invalid Mistral AI API key"),
        (429, "Rate limited by Mistral AI"),
        (500, "Error talking to Mistral AI"),
    ],
)
async def test_api_errors_are_reported(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_client: MagicMock,
    status_code: int,
    expected: str,
) -> None:
    """API failures surface as an error response rather than an exception."""
    mock_client.chat.stream_async = AsyncMock(side_effect=make_sdk_error(status_code))

    result = await converse(hass)

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert expected in speech(result)


async def test_auth_error_starts_reauth(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A rejected key mid-conversation raises a reauth flow."""
    mock_client.chat.stream_async = AsyncMock(side_effect=make_sdk_error(401))

    await converse(hass)
    await hass.async_block_till_done()

    flows = hass.config_entries.flow.async_progress()
    assert [flow["context"]["source"] for flow in flows] == ["reauth"]


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


async def _set_conversation_options(
    hass: HomeAssistant, entry: MockConfigEntry, **options: str | None
) -> None:
    """Replace options on the conversation subentry and reload."""
    subentry = next(
        s
        for s in entry.subentries.values()
        if s.subentry_type == SUBENTRY_TYPE_CONVERSATION
    )
    data = {**subentry.data, **options}
    hass.config_entries.async_update_subentry(
        entry, subentry, data={k: v for k, v in data.items() if v is not None}
    )
    await hass.async_block_till_done()


async def test_web_search_is_sent_as_a_tool(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Enabling web search adds the built-in connector to the request."""
    await _set_conversation_options(
        hass, init_integration, **{CONF_WEB_SEARCH: "web_search"}
    )

    await converse(hass, "what is the news")

    tools = mock_client.chat.stream_async.await_args.kwargs["tools"]
    assert {"type": "web_search"} in tools


async def test_web_search_works_without_the_llm_api(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """An agent with no Home Assistant control can still search the web.

    Tools were previously sent only when an LLM API was selected, so web
    search on its own would have been built and then silently dropped.
    """
    await _set_conversation_options(
        hass,
        init_integration,
        **{CONF_LLM_HASS_API: None, CONF_WEB_SEARCH: "web_search"},
    )

    await converse(hass, "what is the news")

    assert mock_client.chat.stream_async.await_args.kwargs["tools"] == [
        {"type": "web_search"}
    ]


async def test_web_search_off_by_default(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Nothing is searched, and nothing is billed, unless it is turned on."""
    await converse(hass, "what is the news")

    tools = mock_client.chat.stream_async.await_args.kwargs.get("tools", [])
    assert not any("type" in tool for tool in tools)
