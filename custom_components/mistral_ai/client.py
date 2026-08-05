"""Construction of the Mistral AI client."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from mistralai.client import Mistral

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

# The SDK imports its resource modules on first attribute access -- touching
# these four pulls in 89 modules. Home Assistant warns about blocking calls in
# the event loop, and import_module is one, so they are touched here instead,
# inside the executor job that builds the client.
#
# The audio sub-resources (transcriptions, speech, voices) need no separate
# warming: they arrive with `audio`.
LAZY_RESOURCES = ("models", "chat", "audio", "files")


def _build(api_key: str) -> Mistral:
    """Construct a client and import everything it will lazily reach for."""
    client = Mistral(api_key=api_key)
    for resource in LAZY_RESOURCES:
        getattr(client, resource)
    return client


async def async_create_client(hass: HomeAssistant, api_key: str) -> Mistral:
    """Return a Mistral AI client, built off the event loop.

    Two reasons this cannot happen inline: constructing the client builds an
    httpx client, which reads an SSL context from disk, and the SDK's lazy
    imports run on first use.
    """
    return await hass.async_add_executor_job(partial(_build, api_key))
