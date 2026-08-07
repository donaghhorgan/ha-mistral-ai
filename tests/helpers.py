"""Factories for building fake Mistral AI API objects in tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import httpx
from mistralai.client.errors import SDKError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


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
