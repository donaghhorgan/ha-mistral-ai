"""Tests for the Mistral AI task entity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import voluptuous as vol
from homeassistant.components import ai_task
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mistral_ai.ai_task import SUPPORTED_FEATURES

from .helpers import make_chunk, make_sdk_error, stream_of

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

    with pytest.raises(HomeAssistantError, match="structured response"):
        await ai_task.async_generate_data(
            hass,
            task_name="test task",
            entity_id=ENTITY_ID,
            instructions="Describe the room",
            structure=vol.Schema({vol.Required("name"): str}),
        )


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


def _file_chunk(file_id: str = "file-123") -> MagicMock:
    """Return a content chunk referencing a generated file."""
    chunk = MagicMock()
    chunk.file_id = file_id
    return chunk


def _completion(content: object) -> MagicMock:
    """Return a non-streaming chat completion carrying `content`."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _download(
    data: bytes = b"\x89PNG fake", content_type: str = "image/png"
) -> MagicMock:
    """Return a files.download_async response."""
    response = MagicMock()
    response.content = data
    response.headers = {"content-type": content_type}
    return response


requires_image_support = pytest.mark.skipif(
    not hasattr(ai_task.AITaskEntityFeature, "GENERATE_IMAGE"),
    reason="Home Assistant predates AI Task image generation",
)


@pytest.fixture
def mock_image_client(mock_client: MagicMock) -> MagicMock:
    """Wire the client for a successful image generation."""
    text = MagicMock()
    text.file_id = None
    mock_client.chat.complete_async = AsyncMock(
        return_value=_completion([text, _file_chunk()])
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

    request = mock_image_client.chat.complete_async.await_args.kwargs
    assert request["tools"] == [{"type": "image_generation"}]
    assert request["messages"] == [{"role": "user", "content": "a red bicycle"}]

    # The file is fetched by id: generation returns a reference, not bytes.
    assert mock_image_client.files.download_async.await_args.kwargs["file_id"] == (
        "file-123"
    )


@requires_image_support
async def test_generate_image_uses_the_downloads_mime_type(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_image_client: MagicMock
) -> None:
    """The mime type comes from the download, and parameters are stripped.

    Guessing from the chunk instead would mean assuming a format for whatever
    the model chose to produce.
    """
    mock_image_client.files.download_async.return_value = _download(
        content_type="image/webp; charset=binary"
    )

    result = await ai_task.async_generate_image(
        hass, task_name="poster", entity_id=ENTITY_ID, instructions="a red bicycle"
    )

    assert result["mime_type"] == "image/webp"


@requires_image_support
async def test_generate_image_without_a_file_fails(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_image_client: MagicMock
) -> None:
    """A reply with no file reference is an error, not an empty image."""
    mock_image_client.chat.complete_async.return_value = _completion("just some text")

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
    mock_image_client.chat.complete_async.side_effect = make_sdk_error(500)

    with pytest.raises(HomeAssistantError):
        await ai_task.async_generate_image(
            hass, task_name="poster", entity_id=ENTITY_ID, instructions="a red bicycle"
        )
