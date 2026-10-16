"""Factories for fake Mistral AI API objects, and Home Assistant shims.

The tests run against both the oldest supported Home Assistant and the
current one, so where the two disagree on an API the difference is absorbed
here rather than in every test. Each shim checks for the API itself instead of
comparing version numbers.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import httpx
from homeassistant.components import conversation
from homeassistant.helpers import llm
from mistralai.client.errors import SDKError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractContextManager


def patch_assist_tools(tools: list[Any]) -> AbstractContextManager[Any]:
    """Patch the tools the Assist API offers to a conversation agent.

    Older Home Assistant builds the list in AssistAPI._async_get_tools, in
    homeassistant.helpers.llm. Newer releases moved the Assist API to the llm
    component, where it collects tools from each integration's platform
    through a module-level async_get_tools.
    """
    if hasattr(llm, "AssistAPI"):
        return patch(
            "homeassistant.helpers.llm.AssistAPI._async_get_tools",
            return_value=tools,
        )

    from homeassistant.components.llm import LLMTools  # noqa: PLC0415

    return patch(
        "homeassistant.components.llm.async_get_tools",
        return_value=LLMTools(tools=tools),
    )


def tool_output(data: dict[str, Any]) -> Any:
    """Return what a tool's async_call should return on this Home Assistant.

    Newer releases expect an llm.ToolResult, and treat a bare dict as a
    deprecated return that raises when the tool belongs to no integration --
    which a mock tool never does. Older releases have no ToolResult at all.
    """
    if hasattr(llm, "ToolResult"):
        return llm.ToolResult(data=data)
    return data


def make_tool_result(
    tool_call_id: str, tool_name: str, data: dict[str, Any]
) -> conversation.ToolResultContent:
    """Build a tool result, whichever shape this Home Assistant gives it.

    Older releases carry the result as a plain tool_result dict. Newer ones
    wrap it in llm.ToolResult as `result`, keeping tool_result only as a
    deprecated read-only property.
    """
    fields = {
        field.name for field in dataclasses.fields(conversation.ToolResultContent)
    }
    if "result" in fields:
        return conversation.ToolResultContent(
            agent_id="agent",
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=llm.ToolResult(data=data),  # type: ignore[call-arg]
        )
    return conversation.ToolResultContent(
        agent_id="agent",
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_result=data,  # type: ignore[call-arg]
    )


def make_sdk_error(status_code: int) -> SDKError:
    """Build an SDKError carrying the given HTTP status code.

    SDKError derives status_code from a real httpx.Response, so one has to be
    constructed rather than passed as a keyword argument.
    """
    return SDKError("boom", httpx.Response(status_code, text="error"))


def make_chunk(
    content: Any = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
) -> MagicMock:
    """Build a single Mistral streaming chunk.

    Mirrors CompletionEvent, whose payload hangs off ``.data``.

    finish_reason defaults to None rather than being left as an invented
    MagicMock attribute: every attribute of a MagicMock is truthy, so a chunk
    built without it would compare unequal to "length" but be truthy, and any
    future check written as `if chunk.finish_reason:` would fire on every
    chunk in every test.
    """
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason

    data = MagicMock()
    data.choices = [choice]

    chunk = MagicMock()
    chunk.data = data
    return chunk


def make_tool_call(
    index: int = 0,
    call_id: str | None = None,
    name: str | None = None,
    arguments: Any = "",
) -> MagicMock:
    """Build a single streamed tool call fragment."""
    function = MagicMock()
    # `name` cannot be passed to the constructor: MagicMock treats it as the
    # mock's own name rather than as an attribute.
    function.name = name
    function.arguments = arguments

    tool_call = MagicMock()
    tool_call.index = index
    tool_call.id = call_id
    tool_call.function = function
    return tool_call


def make_stream(chunks: list[MagicMock]) -> AsyncIterator[MagicMock]:
    """Return an async iterator over the given chunks."""

    async def _stream() -> AsyncIterator[MagicMock]:
        for chunk in chunks:
            yield chunk

    return _stream()


def stream_of(*chunks: MagicMock):
    """Return a side effect producing a fresh stream on each call.

    The API client is awaited once per tool-calling iteration, and an async
    generator cannot be consumed twice, so each call needs its own.
    """

    async def _side_effect(*args: Any, **kwargs: Any) -> AsyncIterator[MagicMock]:
        return make_stream(list(chunks))

    return _side_effect
