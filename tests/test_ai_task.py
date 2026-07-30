"""Tests for the Mistral AI task entity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import voluptuous as vol
from homeassistant.components import ai_task
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from .helpers import make_chunk, make_sdk_error, stream_of

ENTITY_ID = "ai_task.mistral_ai_task"


async def test_entity_created(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """The AI task subentry produces an entity that can generate data."""
    state = hass.states.get(ENTITY_ID)
    assert state is not None
    assert (
        state.attributes["supported_features"]
        == ai_task.AITaskEntityFeature.GENERATE_DATA
    )


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
