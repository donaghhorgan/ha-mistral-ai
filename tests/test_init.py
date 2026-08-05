"""Tests for setting up and tearing down the integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers.httpx_client import get_async_client
from pytest_homeassistant_custom_component.common import MockConfigEntry

from .helpers import make_sdk_error


async def test_setup_entry(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The entry loads and stashes the client on runtime_data."""
    assert init_integration.state is ConfigEntryState.LOADED
    assert init_integration.runtime_data is mock_client
    mock_client.models.list_async.assert_awaited_once()


async def test_unload_entry(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """The entry unloads cleanly."""
    assert await hass.config_entries.async_unload(init_integration.entry_id)
    await hass.async_block_till_done()

    assert init_integration.state is ConfigEntryState.NOT_LOADED


@pytest.mark.parametrize("status_code", [401, 403])
async def test_setup_invalid_auth_starts_reauth(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_client: MagicMock,
    status_code: int,
) -> None:
    """A rejected API key fails setup and raises a reauth flow."""
    mock_client.models.list_async = AsyncMock(side_effect=make_sdk_error(status_code))

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.SETUP_ERROR

    flows = hass.config_entries.flow.async_progress()
    assert len(flows) == 1
    assert flows[0]["context"]["source"] == "reauth"


async def test_setup_server_error_retries(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_client: MagicMock,
) -> None:
    """A server-side failure schedules a retry rather than a reauth."""
    mock_client.models.list_async = AsyncMock(side_effect=make_sdk_error(500))

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY
    assert not hass.config_entries.flow.async_progress()


async def test_setup_connection_error_retries(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_client: MagicMock,
) -> None:
    """A transport failure schedules a retry."""
    mock_client.models.list_async = AsyncMock(
        side_effect=httpx.ConnectError("no route to host")
    )

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_setup_timeout_retries(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_client: MagicMock,
) -> None:
    """A timeout schedules a retry."""
    mock_client.models.list_async = AsyncMock(side_effect=TimeoutError)

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_setup_uses_home_assistants_http_client(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The SDK is handed Home Assistant's shared httpx client.

    Nothing closes the client, and an options change reloads the entry, so a
    client of our own would leave an abandoned connection pool behind every
    time the user touched a setting.
    """
    with patch(
        "custom_components.mistral_ai.Mistral", return_value=mock_client
    ) as constructor:
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    assert constructor.call_args.kwargs["async_client"] is get_async_client(hass)
