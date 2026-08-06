"""Tests for the Mistral AI task entity."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import voluptuous as vol
from homeassistant.components import ai_task, conversation
from homeassistant.components.conversation.chat_log import ChatLog
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import selector
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mistral_ai.ai_task import SUPPORTED_FEATURES
from custom_components.mistral_ai.const import DEFAULT_MODEL

from .helpers import make_chunk, make_sdk_error, stream_of

if TYPE_CHECKING:
    from pathlib import Path

ENTITY_ID = "ai_task.mistral_ai_task"


async def test_entity_created(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """The AI task subentry produces an entity with the features we support."""
    state = hass.states.get(ENTITY_ID)
    assert state is not None
    assert state.attributes["supported_features"] == SUPPORTED_FEATURES


async def test_generate_data(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """An unstructured task returns the generated text."""
    mock_client.chat.stream_async = AsyncMock(
        side_effect=stream_of(make_chunk(content="A generated haiku"))
    )

    result = await ai_task.async_generate_data(
        hass,
        task_name="test task",
        entity_id=ENTITY_ID,
        instructions="Write a haiku",
    )

    assert result.data == "A generated haiku"


async def test_generate_structured_data(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A structured task parses the JSON response and sets response_format."""
    mock_client.chat.stream_async = AsyncMock(
        side_effect=stream_of(make_chunk(content='{"name": "kitchen", "count": 2}'))
    )

    result = await ai_task.async_generate_data(
        hass,
        task_name="test task",
        entity_id=ENTITY_ID,
        instructions="Describe the room",
        structure=vol.Schema({vol.Required("name"): str, vol.Required("count"): int}),
    )

    assert result.data == {"name": "kitchen", "count": 2}

    # Structured output goes through the native API parameter rather than a
    # synthetic tool.
    kwargs = mock_client.chat.stream_async.await_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    schema = kwargs["response_format"]["json_schema"]
    assert schema["name"] == "test_task"
    assert "name" in schema["schema"]["properties"]
    assert "tools" not in kwargs


async def test_generate_structured_data_invalid_json(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A non-JSON response to a structured task raises a clear error."""
    mock_client.chat.stream_async = AsyncMock(
        side_effect=stream_of(make_chunk(content="not json at all"))
    )

    with pytest.raises(HomeAssistantError) as raised:
        await ai_task.async_generate_data(
            hass,
            task_name="test task",
            entity_id=ENTITY_ID,
            instructions="Describe the room",
            structure=vol.Schema({vol.Required("name"): str}),
        )

    assert raised.value.translation_key == "invalid_structured_response"


async def test_generate_data_api_error(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """An API failure during a task surfaces as a Home Assistant error."""
    mock_client.chat.stream_async = AsyncMock(side_effect=make_sdk_error(500))

    with pytest.raises(HomeAssistantError, match="Error talking to Mistral AI"):
        await ai_task.async_generate_data(
            hass,
            task_name="test task",
            entity_id=ENTITY_ID,
            instructions="Write a haiku",
        )


def _text_chunk(text: str) -> MagicMock:
    """Return a content chunk carrying text rather than a file."""
    chunk = MagicMock()
    chunk.file_id = None
    chunk.text = text
    return chunk


def _file_chunk(file_id: str = "file-123", file_type: str = "png") -> MagicMock:
    """Return a content chunk referencing a generated file.

    file_type is a real string: left as a MagicMock attribute it is truthy but
    not a str, so the fallback that reads it would be skipped and a test could
    pass on the final default without ever exercising it.
    """
    chunk = MagicMock()
    chunk.file_id = file_id
    chunk.file_type = file_type
    return chunk


def _tool_execution_entry() -> MagicMock:
    """Return a tool.execution entry, which carries no content at all.

    The real endpoint emits one of these alongside the message describing the
    connector's own work. It has no `content`, so anything walking the outputs
    has to tolerate that rather than assume every entry is a message.
    """
    entry = MagicMock(spec=["name", "type", "info"])
    entry.name = "image_generation"
    entry.type = "tool.execution"
    return entry


def _conversation(content: object, *, tool_execution: bool = True) -> MagicMock:
    """Return a conversation response whose message entry carries `content`."""
    message = MagicMock()
    message.content = content
    response = MagicMock()
    response.outputs = (
        [_tool_execution_entry(), message] if tool_execution else [message]
    )
    return response


PNG = b"\x89PNG\r\n\x1a\n" + b"fake png body"
JPEG = b"\xff\xd8\xff" + b"fake jpeg body"


def _download(data: bytes = PNG) -> MagicMock:
    """Return a files.download_async response.

    Deliberately carrying the octet-stream the live API actually responds
    with, so nothing here can quietly start depending on the header again.
    """
    response = MagicMock()
    response.content = data
    response.headers = {"content-type": "application/octet-stream"}
    return response


requires_image_support = pytest.mark.skipif(
    not hasattr(ai_task.AITaskEntityFeature, "GENERATE_IMAGE"),
    reason="Home Assistant predates AI Task image generation",
)


@pytest.fixture
def mock_image_client(mock_client: MagicMock) -> MagicMock:
    """Wire the client for a successful image generation."""
    mock_client.beta.conversations.start_async = AsyncMock(
        return_value=_conversation(
            [_text_chunk("Here is your bicycle."), _file_chunk()]
        )
    )
    mock_client.files.download_async = AsyncMock(return_value=_download())
    return mock_client


@requires_image_support
async def test_generate_image(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_image_client: MagicMock
) -> None:
    """An image is generated and downloaded."""
    result = await ai_task.async_generate_image(
        hass, task_name="poster", entity_id=ENTITY_ID, instructions="a red bicycle"
    )

    # The service saves the image and hands back a media source reference
    # rather than the bytes, so the mime type is what is assertable here.
    assert result["mime_type"] == "image/png"
    assert result["media_source_id"]

    request = mock_image_client.beta.conversations.start_async.await_args.kwargs
    assert request["tools"] == [{"type": "image_generation"}]

    # The system prompt goes to instructions: the conversations endpoint has no
    # system role, only user and assistant.
    assert request["instructions"]
    assert request["inputs"] == [{"role": "user", "content": "a red bicycle"}]
    assert not any(entry["role"] == "system" for entry in request["inputs"])

    # Explicit, because this endpoint retains conversations by default where
    # chat completions stores nothing.
    assert request["store"] is False

    # The file is fetched by id: generation returns a reference, not bytes.
    assert mock_image_client.files.download_async.await_args.kwargs["file_id"] == (
        "file-123"
    )


@requires_image_support
async def test_generate_image_reads_the_mime_type_from_the_bytes(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_image_client: MagicMock
) -> None:
    """The image itself decides the mime type, not what claims to describe it.

    Checked against the live API: the download responds
    application/octet-stream, and the chunk said file_type png for a file the
    connector reported at a URL ending .jpg. Here the chunk claims png and the
    bytes are JPEG, and the bytes win.
    """
    mock_image_client.files.download_async.return_value = _download(JPEG)

    result = await ai_task.async_generate_image(
        hass, task_name="poster", entity_id=ENTITY_ID, instructions="a red bicycle"
    )

    assert result["mime_type"] == "image/jpeg"


@requires_image_support
async def test_generate_image_falls_back_to_the_reported_file_type(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_image_client: MagicMock
) -> None:
    """An unrecognised format falls back to what the chunk reported.

    tiff rather than png, so this cannot pass on the final default.
    """
    mock_image_client.beta.conversations.start_async.return_value = _conversation(
        [_text_chunk("Here you go."), _file_chunk(file_type="tiff")]
    )
    mock_image_client.files.download_async.return_value = _download(b"not an image")

    result = await ai_task.async_generate_image(
        hass, task_name="poster", entity_id=ENTITY_ID, instructions="a red bicycle"
    )

    assert result["mime_type"] == "image/tiff"


@requires_image_support
async def test_generate_image_without_a_file_fails(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_image_client: MagicMock
) -> None:
    """A reply with no file reference is an error, not an empty image."""
    mock_image_client.beta.conversations.start_async.return_value = _conversation(
        "just some text"
    )

    with pytest.raises(HomeAssistantError, match="no image"):
        await ai_task.async_generate_image(
            hass, task_name="poster", entity_id=ENTITY_ID, instructions="a red bicycle"
        )

    mock_image_client.files.download_async.assert_not_awaited()


@requires_image_support
async def test_generate_image_with_empty_download_fails(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_image_client: MagicMock
) -> None:
    """An empty download is an error rather than a zero-byte image."""
    mock_image_client.files.download_async.return_value = _download(data=b"")

    with pytest.raises(HomeAssistantError, match="empty image"):
        await ai_task.async_generate_image(
            hass, task_name="poster", entity_id=ENTITY_ID, instructions="a red bicycle"
        )


@requires_image_support
async def test_generate_image_api_error(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_image_client: MagicMock
) -> None:
    """API failures surface as Home Assistant errors."""
    mock_image_client.beta.conversations.start_async.side_effect = make_sdk_error(500)

    with pytest.raises(HomeAssistantError):
        await ai_task.async_generate_image(
            hass, task_name="poster", entity_id=ENTITY_ID, instructions="a red bicycle"
        )


@requires_image_support
async def test_generate_image_sends_attachments(
    hass: HomeAssistant,
    init_integration: MockConfigEntry,
    mock_image_client: MagicMock,
    tmp_path: Path,
) -> None:
    """A reference image reaches the API.

    The entity advertises SUPPORT_ATTACHMENTS, and Home Assistant puts the
    attachments on the user turn of the chat log for us. Building the request
    from task.instructions alone dropped them on the floor while the entity
    went on claiming to support them.
    """
    image = tmp_path / "reference.png"
    image.write_bytes(b"\x89PNG reference")

    with patch(
        "homeassistant.components.ai_task.task._resolve_attachments",
        return_value=[
            conversation.Attachment(
                media_content_id="media-source://media_source/local/reference.png",
                mime_type="image/png",
                path=image,
            )
        ],
    ):
        await ai_task.async_generate_image(
            hass,
            task_name="poster",
            entity_id=ENTITY_ID,
            instructions="in this style",
            attachments=[
                {"media_content_id": "media-source://media_source/local/reference.png"}
            ],
        )

    request = mock_image_client.beta.conversations.start_async.await_args.kwargs
    user_message = request["inputs"][-1]
    encoded = base64.b64encode(b"\x89PNG reference").decode()
    assert user_message["content"] == [
        {"type": "text", "text": "in this style"},
        {"type": "image_url", "image_url": f"data:image/png;base64,{encoded}"},
    ]


@requires_image_support
async def test_generate_image_records_the_assistant_turn(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_image_client: MagicMock
) -> None:
    """What the model said is written back to the chat log.

    Home Assistant adds the user side before calling us. Leaving the assistant
    side out traced an image task as a question with no answer, and showed
    nothing in the conversation debug view.
    """
    logs: list[Any] = []

    original = ChatLog.async_add_assistant_content_without_tools

    def _record(self: ChatLog, content: Any) -> None:
        logs.append(content)
        original(self, content)

    with patch.object(ChatLog, "async_add_assistant_content_without_tools", _record):
        await ai_task.async_generate_image(
            hass, task_name="poster", entity_id=ENTITY_ID, instructions="a red bicycle"
        )

    assert [content.content for content in logs] == ["Here is your bicycle."]


@requires_image_support
async def test_image_generation_withdrawn_for_a_model_without_tools(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_client: MagicMock,
    mock_models_response: MagicMock,
) -> None:
    """A model that cannot call tools does not advertise image generation.

    Image generation is a built-in connector passed as a tool, so a model
    that cannot call tools cannot produce an image whatever is asked of it.
    Advertising it anyway pushed the failure out to automation run time.
    """
    for card in mock_models_response.data:
        if card.id == DEFAULT_MODEL:
            card.capabilities.function_calling = False

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    features = hass.states.get(ENTITY_ID).attributes["supported_features"]
    assert not features & ai_task.AITaskEntityFeature.GENERATE_IMAGE
    # The rest of the entity is untouched.
    assert features & ai_task.AITaskEntityFeature.GENERATE_DATA


@requires_image_support
async def test_image_generation_kept_when_capabilities_cannot_be_read(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A failed lookup leaves the feature alone.

    Withdrawing a feature that works, because a listing call did not, is worse
    than the error it would prevent.
    """
    mock_client.models.retrieve_async = AsyncMock(side_effect=make_sdk_error(500))

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    features = hass.states.get(ENTITY_ID).attributes["supported_features"]
    assert features & ai_task.AITaskEntityFeature.GENERATE_IMAGE


@requires_image_support
async def test_missing_image_names_the_model(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_image_client: MagicMock
) -> None:
    """A model that passes the check but generates nothing says which it was.

    function_calling is a precondition for the connector, not a guarantee, so
    this failure survives the capability gate and has to be diagnosable.
    """
    mock_image_client.beta.conversations.start_async = AsyncMock(
        return_value=_conversation("I cannot do that")
    )

    with pytest.raises(HomeAssistantError, match=DEFAULT_MODEL):
        await ai_task.async_generate_image(
            hass, task_name="poster", entity_id=ENTITY_ID, instructions="a bicycle"
        )


async def test_structure_with_selectors_converts(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """A structure built from Home Assistant selectors reaches the API.

    An AI task has no LLM API, so there is no custom_serializer from one.
    Passing None leaves voluptuous_openapi unable to convert a selector and
    every structured generation dies with:

        cannot use 'TextSelector' as a dict key (unhashable type)

    llm.selector_serializer is the fallback core uses for exactly this.
    """
    mock_client.chat.stream_async = AsyncMock(
        side_effect=stream_of(make_chunk(content='{"verdict": "warm", "degrees": 21}'))
    )

    structure = vol.Schema(
        {
            vol.Required("verdict"): selector.TextSelector(),
            vol.Optional("degrees"): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0, max=50)
            ),
        }
    )

    await ai_task.async_generate_data(
        hass,
        task_name="verdict",
        entity_id=ENTITY_ID,
        instructions="Is 21 degrees warm?",
        structure=structure,
    )

    schema = mock_client.chat.stream_async.await_args.kwargs["response_format"]
    properties = schema["json_schema"]["schema"]["properties"]
    assert properties["verdict"]["type"] == "string"
    assert properties["degrees"]["type"] == "number"


@requires_image_support
async def test_generate_image_tolerates_entries_without_content(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_image_client: MagicMock
) -> None:
    """A tool.execution entry alongside the message does not derail the walk.

    The live endpoint returns one of these describing the connector's own work,
    and it has no content at all. Assuming every entry is a message is an
    AttributeError on a response that is perfectly well formed.
    """
    result = await ai_task.async_generate_image(
        hass, task_name="poster", entity_id=ENTITY_ID, instructions="a red bicycle"
    )

    assert result["mime_type"] == "image/png"
    outputs = mock_image_client.beta.conversations.start_async.return_value.outputs
    assert any(getattr(entry, "type", None) == "tool.execution" for entry in outputs)
