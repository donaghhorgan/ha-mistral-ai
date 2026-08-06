"""Unit tests for the message conversion helpers."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

import pytest
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from custom_components.mistral_ai.entity import (
    _async_convert_messages,
    _convert_content,
    _format_tool,
    _parse_arguments,
    _transform_stream,
)

from .helpers import make_chunk, make_stream, make_tool_call

if TYPE_CHECKING:
    from pathlib import Path


def test_convert_user_content() -> None:
    """User content becomes a user message."""
    content = conversation.UserContent(content="hello")
    assert _convert_content(content) == {"role": "user", "content": "hello"}


def test_convert_system_content() -> None:
    """System content becomes a system message."""
    content = conversation.SystemContent(content="be helpful")
    assert _convert_content(content) == {"role": "system", "content": "be helpful"}


def test_convert_assistant_content() -> None:
    """Assistant content becomes an assistant message."""
    content = conversation.AssistantContent(agent_id="agent", content="hi")
    assert _convert_content(content) == {"role": "assistant", "content": "hi"}


def test_convert_assistant_content_with_tool_calls() -> None:
    """Tool calls are serialised with JSON string arguments."""
    content = conversation.AssistantContent(
        agent_id="agent",
        content=None,
        tool_calls=[
            llm.ToolInput(
                id="call_1", tool_name="turn_on", tool_args={"entity": "light.kitchen"}
            )
        ],
    )

    message = _convert_content(content)

    assert message["role"] == "assistant"
    assert message["content"] == ""
    assert message["tool_calls"][0]["id"] == "call_1"
    assert message["tool_calls"][0]["function"]["name"] == "turn_on"
    # The API expects a JSON string here, not a nested object.
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "entity": "light.kitchen"
    }


def test_convert_tool_result_content() -> None:
    """Tool results become tool messages carrying the call id."""
    content = conversation.ToolResultContent(
        agent_id="agent",
        tool_call_id="call_1",
        tool_name="turn_on",
        tool_result={"ok": True},
    )

    message = _convert_content(content)

    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call_1"
    assert message["name"] == "turn_on"
    assert json.loads(message["content"]) == {"ok": True}


def test_convert_unsupported_content() -> None:
    """An unknown content type is rejected rather than silently dropped."""
    with pytest.raises(HomeAssistantError, match="Unsupported content type"):
        _convert_content(object())


def test_format_tool_converts_parameter_types() -> None:
    """Tool parameters keep their real types.

    The previous implementation hard-coded every parameter to a string, so no
    numeric or boolean argument could ever be described correctly.
    """

    class FakeTool(llm.Tool):
        name = "set_temperature"
        description = "Set the temperature"
        parameters = vol.Schema(
            {vol.Required("degrees"): int, vol.Optional("boost"): bool}
        )

    spec = _format_tool(FakeTool(), None)

    assert spec["type"] == "function"
    assert spec["function"]["name"] == "set_temperature"
    properties = spec["function"]["parameters"]["properties"]
    assert properties["degrees"]["type"] == "integer"
    assert properties["boost"]["type"] == "boolean"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"a": 1}, {"a": 1}),
        ('{"a": 1}', {"a": 1}),
        ("", {}),
        (None, {}),
    ],
)
def test_parse_arguments(value: object, expected: dict) -> None:
    """Arguments are accepted as either a dict or a JSON string."""
    assert _parse_arguments(value) == expected


def test_parse_arguments_malformed() -> None:
    """Malformed JSON is reported rather than silently swallowed."""
    with pytest.raises(HomeAssistantError, match="malformed tool arguments"):
        _parse_arguments('{"a": ')


def test_parse_arguments_non_object() -> None:
    """A JSON scalar is rejected, since tool args must be an object."""
    with pytest.raises(HomeAssistantError, match="non-object tool arguments"):
        _parse_arguments("[1, 2, 3]")


async def collect(stream) -> list[dict]:
    """Drain an async generator into a list."""
    return [item async for item in stream]


async def test_transform_stream_content() -> None:
    """Content chunks are emitted after an opening role delta."""
    deltas = await collect(
        _transform_stream(
            make_stream([make_chunk(content="a"), make_chunk(content="b")])
        )
    )

    assert deltas == [{"role": "assistant"}, {"content": "a"}, {"content": "b"}]


async def test_transform_stream_skips_empty_chunks() -> None:
    """Chunks with no choices are ignored."""
    empty = make_chunk()
    empty.data.choices = []

    deltas = await collect(
        _transform_stream(make_stream([empty, make_chunk(content="a")]))
    )

    assert deltas == [{"role": "assistant"}, {"content": "a"}]


async def test_transform_stream_buffers_tool_arguments() -> None:
    """Fragmented tool arguments are only emitted once complete."""
    deltas = await collect(
        _transform_stream(
            make_stream(
                [
                    make_chunk(
                        tool_calls=[
                            make_tool_call(
                                index=0, call_id="c1", name="tool", arguments='{"x'
                            )
                        ]
                    ),
                    make_chunk(tool_calls=[make_tool_call(index=0, arguments='": 1}')]),
                ]
            )
        )
    )

    assert deltas[0] == {"role": "assistant"}
    # Exactly one tool call delta, emitted at the end, never a partial one.
    tool_deltas = [d for d in deltas if "tool_calls" in d]
    assert len(tool_deltas) == 1

    tool_call = tool_deltas[0]["tool_calls"][0]
    assert tool_call.id == "c1"
    assert tool_call.tool_name == "tool"
    assert tool_call.tool_args == {"x": 1}


async def test_transform_stream_parallel_tool_calls() -> None:
    """Two tool calls in one response do not overwrite each other."""
    deltas = await collect(
        _transform_stream(
            make_stream(
                [
                    make_chunk(
                        tool_calls=[
                            make_tool_call(
                                index=0, call_id="c1", name="first", arguments="{}"
                            ),
                            make_tool_call(
                                index=1, call_id="c2", name="second", arguments="{}"
                            ),
                        ]
                    )
                ]
            )
        )
    )

    tool_calls = next(d for d in deltas if "tool_calls" in d)["tool_calls"]
    assert [call.tool_name for call in tool_calls] == ["first", "second"]


async def test_transform_stream_empty() -> None:
    """An empty stream produces no deltas at all."""
    assert await collect(_transform_stream(make_stream([]))) == []


async def test_transform_stream_segmented_content() -> None:
    """Content delivered as a list of segments is flattened to text."""

    class Segment:
        def __init__(self, text: str) -> None:
            self.text = text

    deltas = await collect(
        _transform_stream(
            make_stream([make_chunk(content=[Segment("one "), Segment("two")])])
        )
    )

    assert deltas == [
        {"role": "assistant"},
        {"content": "one "},
        {"content": "two"},
    ]


async def test_transform_stream_dict_arguments() -> None:
    """Arguments already delivered as an object are accepted as-is."""
    deltas = await collect(
        _transform_stream(
            make_stream(
                [
                    make_chunk(
                        tool_calls=[
                            make_tool_call(
                                index=0,
                                call_id="c1",
                                name="tool",
                                arguments={"x": 1},
                            )
                        ]
                    )
                ]
            )
        )
    )

    tool_call = next(d for d in deltas if "tool_calls" in d)["tool_calls"][0]
    assert tool_call.tool_args == {"x": 1}


async def test_transform_stream_ignores_tool_call_without_function() -> None:
    """A tool call fragment carrying no function is skipped."""
    fragment = make_tool_call(index=0, call_id="c1")
    fragment.function = None

    deltas = await collect(
        _transform_stream(make_stream([make_chunk(tool_calls=[fragment])]))
    )

    assert not any("tool_calls" in delta for delta in deltas)


async def test_convert_messages_without_attachments(hass: HomeAssistant) -> None:
    """A plain user turn keeps its content as a string."""
    messages = await _async_convert_messages(
        hass, [conversation.UserContent(content="hello")]
    )

    assert messages == [{"role": "user", "content": "hello"}]


async def test_convert_messages_with_image_attachment(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """An image attachment is inlined as a base64 data URI chunk."""
    image = tmp_path / "snapshot.jpg"
    image.write_bytes(b"fake-jpeg-bytes")

    content = conversation.UserContent(
        content="what is this?",
        attachments=[
            conversation.Attachment(
                media_content_id="media", mime_type="image/jpeg", path=image
            )
        ],
    )

    messages = await _async_convert_messages(hass, [content])

    assert messages[0]["role"] == "user"
    text, chunk = messages[0]["content"]
    assert text == {"type": "text", "text": "what is this?"}
    assert chunk["type"] == "image_url"
    assert chunk["image_url"] == (
        "data:image/jpeg;base64," + base64.b64encode(b"fake-jpeg-bytes").decode()
    )


async def test_convert_messages_rejects_non_image(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """A non-image attachment is rejected rather than silently dropped."""
    doc = tmp_path / "notes.txt"
    doc.write_bytes(b"hello")

    content = conversation.UserContent(
        content="read this",
        attachments=[
            conversation.Attachment(
                media_content_id="media", mime_type="text/plain", path=doc
            )
        ],
    )

    with pytest.raises(HomeAssistantError) as raised:
        await _async_convert_messages(hass, [content])

    assert raised.value.translation_key == "unsupported_attachment_type"
    assert raised.value.translation_placeholders == {"path": str(doc)}


async def test_convert_messages_missing_file(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """A missing attachment file is reported clearly."""
    content = conversation.UserContent(
        content="look",
        attachments=[
            conversation.Attachment(
                media_content_id="media",
                mime_type="image/png",
                path=tmp_path / "gone.png",
            )
        ],
    )

    with pytest.raises(HomeAssistantError) as raised:
        await _async_convert_messages(hass, [content])

    assert raised.value.translation_key == "attachment_not_found"


async def test_transform_stream_thinking_content() -> None:
    """Reasoning chunks are surfaced as thinking content, not as the answer."""

    class Thought:
        def __init__(self, text: str) -> None:
            self.text = text

    class ThinkPart:
        type = "thinking"

        def __init__(self, *thoughts: Thought) -> None:
            self.thinking = list(thoughts)

    class TextPart:
        type = "text"

        def __init__(self, text: str) -> None:
            self.text = text

    deltas = await collect(
        _transform_stream(
            make_stream(
                [make_chunk(content=[ThinkPart(Thought("hmm")), TextPart("answer")])]
            )
        )
    )

    assert deltas == [
        {"role": "assistant"},
        {"thinking_content": "hmm"},
        {"content": "answer"},
    ]
