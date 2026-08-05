"""Construction of the Mistral AI client."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from homeassistant.helpers.httpx_client import get_async_client
from mistralai.client import Mistral

if TYPE_CHECKING:
    import httpx
    from homeassistant.core import HomeAssistant

# The SDK imports its resource modules on first attribute access -- touching
# these four pulls in 89 modules. Home Assistant warns about blocking calls in
# the event loop, and import_module is one, so they are touched here instead,
# inside the executor job that builds the client.
#
# The audio sub-resources (transcriptions, speech, voices) need no separate
# warming: they arrive with `audio`.
LAZY_RESOURCES = ("models", "chat", "audio", "files")


def _build(api_key: str, async_client: httpx.AsyncClient) -> Mistral:
    """Construct a client and import everything it will lazily reach for."""
    client = Mistral(api_key=api_key, async_client=async_client)
    for resource in LAZY_RESOURCES:
        getattr(client, resource)
    return client


async def async_create_client(hass: HomeAssistant, api_key: str) -> Mistral:
    """Return a Mistral AI client, built off the event loop.

    The SDK is handed Home Assistant's shared httpx client rather than being
    left to build its own. Nothing closes the client we build, and an options
    change reloads the entry, so a private pool would be abandoned every time
    the user touched a setting. The SDK records that the client was supplied
    and leaves it alone on teardown, so the shared one is never closed on us.

    Two reasons this cannot happen inline: the constructor still builds a
    *synchronous* httpx client when it is not given one, which reads an SSL
    context from disk, and the SDK's lazy imports run on first use. We never
    call the synchronous methods, but the client is built either way.
    """
    return await hass.async_add_executor_job(
        partial(_build, api_key, get_async_client(hass))
    )
