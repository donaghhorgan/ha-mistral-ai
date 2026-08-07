"""The Mistral AI integration."""

from __future__ import annotations

import asyncio

import httpx
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import (
    ConfigEntryAuthFailed,
    ConfigEntryError,
    ConfigEntryNotReady,
)
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType
from mistralai.client import Mistral
from mistralai.client.errors import SDKError

from .client import async_create_client
from .const import CONF_API_KEY, DOMAIN, TIMEOUT

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

PLATFORMS: tuple[Platform, ...] = (
    Platform.AI_TASK,
    Platform.CONVERSATION,
    Platform.STT,
    Platform.TTS,
)

type MistralConfigEntry = ConfigEntry[Mistral]


async def async_setup_entry(hass: HomeAssistant, entry: MistralConfigEntry) -> bool:
    """Set up Mistral AI from a config entry."""
    client = await async_create_client(hass, entry.data[CONF_API_KEY])

    # Verify credentials during setup so that a bad key surfaces as a reauth
    # flow, rather than as a failure on the user's first sentence.
    try:
        async with asyncio.timeout(TIMEOUT):
            await client.models.list_async()
    except SDKError as err:
        if err.status_code == 401:
            raise ConfigEntryAuthFailed("Invalid Mistral AI API key") from err
        if err.status_code == 403:
            # Permanent rather than not-ready: the key authenticated, so
            # retrying the same call on the same account will keep getting the
            # same answer. ConfigEntryError says so and stops, where
            # ConfigEntryNotReady would retry a failure that cannot resolve
            # itself, and ConfigEntryAuthFailed -- which is what this was --
            # would ask for a replacement key that is not the problem.
            raise ConfigEntryError(
                f"Mistral AI refused the request, and the API key is valid: {err}"
            ) from err
        raise ConfigEntryNotReady(f"Error talking to Mistral AI: {err}") from err
    except (TimeoutError, httpx.HTTPError) as err:
        raise ConfigEntryNotReady(f"Error talking to Mistral AI: {err}") from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: MistralConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_update_options(hass: HomeAssistant, entry: MistralConfigEntry) -> None:
    """Reload the config entry when its options change."""
    await hass.config_entries.async_reload(entry.entry_id)
