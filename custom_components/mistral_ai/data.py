"""What a loaded config entry holds.

The client, and a cache of the model list beside it.

Not a DataUpdateCoordinator, which is the obvious Home Assistant answer and
brings two things this does not need. It polls on a schedule, and the model
list turns over a few times a year -- refreshing it every few hours would be
requests spent to learn nothing. And it drives entity availability, which this
integration does not do: a listing failure is not a reason to mark a
conversation agent unavailable, since the agent talks to a different endpoint
and works perfectly well while the list is stale.

What was actually wrong was re-fetching per form render, and a cache fixes
that without the rest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mistralai.client import Mistral

_LOGGER = logging.getLogger(__name__)

# How long a fetched model list is reused for.
#
# Long, because the thing it describes barely moves: models are added and
# retired a handful of times a year. The cost of being an hour stale is that a
# model released this morning is missing from a dropdown; the cost of being
# fresh is a network round trip before the form can render, every time it is
# opened.
#
# Reloading the integration clears it, which is the escape hatch for anyone who
# needs the list now.
MODEL_CACHE_SECONDS = 3600


@dataclass
class MistralData:
    """The client for a loaded entry, and its cached model list."""

    client: Mistral

    _models: list[Any] | None = field(default=None, repr=False)
    _fetched_at: float = field(default=0.0, repr=False)

    async def async_models(
        self, fetch: Callable[[Mistral], Awaitable[list[Any]]]
    ) -> list[Any]:
        """Return the model list, fetching it only when it is stale.

        The fetch is passed in rather than imported, because the function that
        knows how to ask lives in the config flow and importing it here would
        make a cycle. It is called with the client and is expected to raise
        rather than return an empty list on failure -- a failure must not be
        cached, or one blip would leave every form empty for the hour.
        """
        if self._models is not None and monotonic() - self._fetched_at < (
            MODEL_CACHE_SECONDS
        ):
            _LOGGER.debug("Using the cached Mistral AI model list")
            return self._models

        models = await fetch(self.client)
        self._models = models
        self._fetched_at = monotonic()
        return models

    def invalidate_models(self) -> None:
        """Forget the cached list.

        Used by the tests to reach the uncached path, which setup's seeding
        otherwise hides, and there for whatever adds a "refresh the model
        list" action or notices a model has been retired -- so that has
        somewhere obvious to go rather than reaching into the fields.
        """
        self._models = None
        self._fetched_at = 0.0
