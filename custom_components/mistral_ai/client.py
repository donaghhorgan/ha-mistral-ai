"""Construction of the Mistral AI client."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import httpx2
from mistralai.client import Mistral

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

# What a failed request can raise. The SDK moved from httpx to httpx2 in 3.0.0,
# and the two do not share a base class, so an `except httpx.HTTPError` no
# longer catches what the SDK raises. httpx stays in the tuple for the requests
# we make with it ourselves.
HTTP_ERRORS = (httpx.HTTPError, httpx2.HTTPError)

# The SDK imports its resource modules on first attribute access -- touching
# these five pulls in every module the integration ever reaches for. Home
# Assistant warns about blocking calls in the event loop, and import_module is
# one, so they are touched here instead, inside the executor job that builds
# the client.
#
# The audio sub-resources (transcriptions, speech, voices) need no separate
# warming: they arrive with `audio`. `beta` is the same story for
# `beta.conversations`, the web search path -- Beta.__init__ builds every beta
# sub-resource eagerly, so importing the `beta` module warms `conversations`
# along with prompts, agents, libraries, connectors, rag and users. Left out,
# the first web-search turn paid for it instead:
#
#   Detected blocking call to open inside the event loop by custom
#   integration 'mistral_ai' at entity.py, line 774: stream = await
#   client.beta.conversations.start_stream_async(
LAZY_RESOURCES = ("models", "chat", "audio", "files", "beta")


def _build(api_key: str) -> Mistral:
    """Construct a client and import everything it will lazily reach for."""
    client = Mistral(api_key=api_key)
    for resource in LAZY_RESOURCES:
        getattr(client, resource)
    return client


async def async_create_client(hass: HomeAssistant, api_key: str) -> Mistral:
    """Return a Mistral AI client, built off the event loop.

    The SDK builds its own HTTP client rather than being handed Home
    Assistant's shared one. Since 3.0.0 it speaks httpx2, whose requests and
    responses are different types from httpx's, so the shared client cannot be
    passed in -- the SDK would build an httpx2 request and ask an httpx client
    to send it. An options change reloads the entry, so this abandons a pool
    each time; the SDK closes clients it created itself when the Mistral
    object is collected.

    This has to happen off the event loop: the constructor builds synchronous
    and asynchronous clients, which read an SSL context from disk, and the
    SDK's lazy imports run on first use.
    """
    return await hass.async_add_executor_job(_build, api_key)
