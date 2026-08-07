"""Tests for setting up and tearing down the integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers.httpx_client import get_async_client
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mistral_ai.client import LAZY_RESOURCES
from custom_components.mistral_ai.config_flow import async_fetch_model_cards
from custom_components.mistral_ai.data import MODEL_CACHE_SECONDS

from .helpers import make_sdk_error


async def test_setup_entry(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The entry loads and stashes the client on runtime_data."""
    assert init_integration.state is ConfigEntryState.LOADED
    assert init_integration.runtime_data.client is mock_client
    mock_client.models.list_async.assert_awaited_once()


async def test_unload_entry(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """The entry unloads cleanly."""
    assert await hass.config_entries.async_unload(init_integration.entry_id)
    await hass.async_block_till_done()

    assert init_integration.state is ConfigEntryState.NOT_LOADED


async def test_setup_invalid_auth_starts_reauth(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_client: MagicMock,
) -> None:
    """A rejected API key fails setup and raises a reauth flow."""
    mock_client.models.list_async = AsyncMock(side_effect=make_sdk_error(401))

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.SETUP_ERROR

    flows = hass.config_entries.flow.async_progress()
    assert len(flows) == 1
    assert flows[0]["context"]["source"] == "reauth"


async def test_setup_forbidden_does_not_start_reauth(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_client: MagicMock,
) -> None:
    """A 403 fails setup permanently and asks for no new key.

    403 used to be handled here as an authentication failure. It is not one --
    every way of getting the key wrong answers 401 -- so the reauth flow it
    raised asked the user to replace a key that was working.

    SETUP_ERROR rather than SETUP_RETRY matters too: the account is not going
    to start being permitted, so retrying the same call forever is noise.
    """
    mock_client.models.list_async = AsyncMock(side_effect=make_sdk_error(403))

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.SETUP_ERROR
    assert not hass.config_entries.flow.async_progress()


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
        "custom_components.mistral_ai.client.Mistral", return_value=mock_client
    ) as constructor:
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    assert constructor.call_args.kwargs["async_client"] is get_async_client(hass)


async def test_client_is_built_off_the_event_loop(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry, mock_client: MagicMock
) -> None:
    """The client is constructed in an executor, with its lazy imports warmed.

    Two blocking things happen on construction: httpx reads an SSL context
    from disk, and the SDK imports 89 modules the first time its resources
    are touched. Home Assistant warns about both in the event loop, and the
    warming has to happen in the same executor job or it simply moves the
    import back onto the loop at first use.
    """
    with patch(
        "custom_components.mistral_ai.client._build", return_value=mock_client
    ) as build:
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    build.assert_called_once()


def test_lazy_resources_cover_what_the_integration_uses() -> None:
    """Every resource the integration reaches for is warmed.

    A resource left out of the list is not an error -- it is an
    import_module on the event loop the first time it is used, which shows
    up as a Home Assistant warning and nothing else.
    """
    assert set(LAZY_RESOURCES) == {"models", "chat", "audio", "files"}


async def test_setup_seeds_the_model_cache(
    hass: HomeAssistant, mock_client: MagicMock, mock_config_entry: MockConfigEntry
) -> None:
    """The listing setup makes to validate the key is kept, not discarded.

    Setup has to fetch the model list anyway -- it is how a bad key becomes a
    reauth flow rather than a failure on the first sentence -- and it used to
    drop the response. The cache then started cold, so the first form opened
    after a restart paid for a listing the integration had held seconds
    earlier.
    """
    mock_config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    fetched = mock_client.models.list_async.await_count
    assert fetched == 1

    # A second reader gets the seeded list rather than a second round trip.
    cards = await mock_config_entry.runtime_data.async_models(async_fetch_model_cards)

    assert mock_client.models.list_async.await_count == fetched
    assert [card.id for card in cards] == [
        card.id for card in mock_client.models.list_async.return_value.data
    ]


async def test_seeding_does_not_extend_the_freshness_window(
    hass: HomeAssistant, mock_client: MagicMock, mock_config_entry: MockConfigEntry
) -> None:
    """A seeded list still goes stale on schedule.

    The seed is a head start, not a way to make an old list look new. An entry
    loaded for longer than the cache window must refetch on the next render,
    or a long-running Home Assistant would never see a newly released model.

    The clock is controlled across the setup as well as the reads, because the
    seed is stamped during setup: replacing a single later reading would leave
    it compared against the real system uptime, which is a much larger number,
    so the cache would look fresh no matter what value was chosen.
    """
    clock = [0.0]
    mock_config_entry.add_to_hass(hass)

    with patch(
        "custom_components.mistral_ai.data.monotonic", side_effect=lambda: clock[0]
    ):
        assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        fetched = mock_client.models.list_async.await_count

        # Inside the window: still the seeded list.
        await mock_config_entry.runtime_data.async_models(async_fetch_model_cards)
        assert mock_client.models.list_async.await_count == fetched

        clock[0] = MODEL_CACHE_SECONDS + 1
        await mock_config_entry.runtime_data.async_models(async_fetch_model_cards)

    assert mock_client.models.list_async.await_count == fetched + 1
