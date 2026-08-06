"""Diagnostics for the Mistral AI integration.

Implementing this puts a "Download diagnostics" button on the integration's
page, producing a redacted dump the user can attach to an issue.

What it contains is chosen from what bug reports here have actually turned
on: which model, whether web search was enabled, and which version of the SDK
is installed. Without it every report starts with a round of questions, and
the answers arrive one at a time.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.helpers import entity_registry as er

from .const import CONF_API_KEY, CONF_PROMPT, CONF_VOICE

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from . import MistralConfigEntry

# The API key is obvious. The prompt is not, and matters more than it looks:
# a custom one routinely names the house, the rooms and the people in them,
# and this file exists to be pasted into public issues.
#
# The voice is redacted for a narrower reason. A preset voice ID says nothing,
# but a custom one is created against the account and names something the user
# made. Since the two are indistinguishable here, the identifying case decides
# it -- at the cost of some usefulness on exactly the text-to-speech reports
# where a voice would help, which is a trade worth knowing about rather than
# discovering.
TO_REDACT = {CONF_API_KEY, CONF_PROMPT, CONF_VOICE}


def _sdk_version() -> str:
    """Return the installed SDK version.

    Asked of the installed package rather than read from the manifest, which
    only pins a floor -- `mistralai>=2.1.0` says nothing about what resolved.
    Which version is actually present is the question worth answering when a
    request shape changes underneath us, and it has, twice.
    """
    try:
        return version("mistralai")
    except PackageNotFoundError:  # pragma: no cover - the package is a hard dep
        return "unknown"


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: MistralConfigEntry
) -> dict[str, Any]:
    """Return redacted diagnostics for a config entry."""
    registry = er.async_get(hass)

    entities = [
        {
            "entity_id": registry_entry.entity_id,
            "domain": registry_entry.domain,
            "disabled": registry_entry.disabled,
            "subentry_id": registry_entry.config_subentry_id,
        }
        for registry_entry in er.async_entries_for_config_entry(
            registry, entry.entry_id
        )
    ]

    return {
        "sdk_version": _sdk_version(),
        "entry": {
            "state": entry.state.value,
            "version": f"{entry.version}.{entry.minor_version}",
            "data": async_redact_data(dict(entry.data), TO_REDACT),
            "options": async_redact_data(dict(entry.options), TO_REDACT),
        },
        # Keyed by subentry id so an entity below can be tied to the settings
        # that produced it. The options are the point of the whole file: model,
        # temperature, maximum tokens and web search tier are what most
        # failures depend on.
        "subentries": {
            subentry_id: {
                "type": subentry.subentry_type,
                "title": subentry.title,
                "data": async_redact_data(dict(subentry.data), TO_REDACT),
            }
            for subentry_id, subentry in entry.subentries.items()
        },
        "entities": entities,
    }
