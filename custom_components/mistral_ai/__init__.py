"""The Mistral AI integration."""

from __future__ import annotations

import asyncio
from functools import partial

import httpx
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType
from mistralai.client import Mistral
from mistralai.client.errors import SDKError

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
    # Hand the SDK Home Assistant's shared httpx client rather than letting it
    # build its own. Nothing here closes the client, and an options change
    # reloads the entry, so a private pool would be abandoned every time the
    # user touched a setting. The SDK records that the client was supplied and
    # leaves it alone on teardown, so the shared one is never closed under us.
    #
    # The executor hop stays regardless: the constructor also builds a
    # synchronous httpx client when it is not given one, and that loads the SSL
    # context from disk. We never call the synchronous methods, but the client
    # is built either way.
    client = await hass.async_add_executor_job(
        partial(
            Mistral,
            api_key=entry.data[CONF_API_KEY],
            async_client=get_async_client(hass),
        )
    )

    # Verify credentials during setup so that a bad key surfaces as a reauth
    # flow, rather than as a failure on the user's first sentence.
    try:
        async with asyncio.timeout(TIMEOUT):
            await client.models.list_async()
    except SDKError as err:
        if err.status_code in (401, 403):
            raise ConfigEntryAuthFailed("Invalid Mistral AI API key") from err
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
