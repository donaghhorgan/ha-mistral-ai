"""Fixtures for the Mistral AI tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mistral_ai.const import (
    CONF_API_KEY,
    CONF_MODEL,
    DEFAULT_MODEL,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK_DATA,
    SUBENTRY_TYPE_CONVERSATION,
)

from .helpers import make_chunk, make_stream

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(
    enable_custom_integrations: None,
) -> Generator[None]:
    """Enable loading of this custom integration in every test."""
    yield


@pytest.fixture(autouse=True)
async def setup_homeassistant(hass: HomeAssistant) -> None:
    """Set up the core integration.

    The conversation component's default agent reads exposed-entity data on
    start, which only exists once `homeassistant` itself has been set up.
    """
    assert await async_setup_component(hass, "homeassistant", {})


@pytest.fixture
def mock_models_response() -> MagicMock:
    """Return a models.list_async() response listing two models."""
    first = MagicMock()
    first.id = DEFAULT_MODEL
    second = MagicMock()
    second.id = "mistral-large-latest"

    response = MagicMock()
    response.data = [first, second]
    return response


@pytest.fixture
def mock_client(mock_models_response: MagicMock) -> Generator[MagicMock]:
    """Patch the Mistral client everywhere it is constructed."""
    client = MagicMock()
    client.models.list_async = AsyncMock(return_value=mock_models_response)
    client.chat.stream_async = AsyncMock(
        return_value=make_stream([make_chunk(content="Hello there")])
    )

    with (
        patch("custom_components.mistral_ai.Mistral", return_value=client),
        patch(
            "custom_components.mistral_ai.config_flow.Mistral",
            return_value=client,
        ),
    ):
        yield client


@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Return a config entry with one conversation and one AI task subentry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Mistral AI",
        data={CONF_API_KEY: "test-api-key"},
        subentries_data=[
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_CONVERSATION,
                data={CONF_MODEL: DEFAULT_MODEL},
                title="Mistral AI conversation",
                unique_id=None,
            ),
            ConfigSubentryData(
                subentry_type=SUBENTRY_TYPE_AI_TASK_DATA,
                data={CONF_MODEL: DEFAULT_MODEL},
                title="Mistral AI task",
                unique_id=None,
            ),
        ],
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
async def init_integration(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_client: MagicMock,
) -> MockConfigEntry:
    """Set up the integration and return its config entry."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    return mock_config_entry
