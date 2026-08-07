"""Repair flows for the Mistral AI integration.

One repair: an entity configured with a model Mistral has scheduled for
retirement. The dropdown already labels those when someone is choosing
(#121), which helps the next person to open the form and nobody who set
theirs up last year. This is the other half -- telling someone whose entity
already points at one, before it stops working rather than after.

The replacement is not guessed. Every retiring model card names its successor
in `deprecation_replacement_model`, so the flow offers that specific model,
and where the API names none the issue is raised as information rather than
as something with a button.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import voluptuous as vol
from homeassistant.components.repairs import RepairsFlow
from homeassistant.core import callback
from homeassistant.helpers import issue_registry as ir

from .const import CONF_MODEL, DOMAIN

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.data_entry_flow import FlowResult

ISSUE_PREFIX = "deprecated_model_"


def issue_id(subentry_id: str) -> str:
    """Return the issue id for a subentry.

    Per subentry rather than per model: two agents on the same retiring model
    are two things to fix, and fixing one should not clear the other's warning.
    """
    return f"{ISSUE_PREFIX}{subentry_id}"


class DeprecatedModelRepairFlow(RepairsFlow):
    """Move one subentry onto the replacement model."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Store what the issue was raised about."""
        self._entry_id = cast("str", data["entry_id"])
        self._subentry_id = cast("str", data["subentry_id"])
        self._replacement = cast("str", data["replacement"])

    async def async_step_init(
        self, user_input: dict[str, str] | None = None
    ) -> FlowResult:
        """Show what will change, and wait to be told to do it."""
        return await self.async_step_confirm()

    async def async_step_confirm(
        self, user_input: dict[str, str] | None = None
    ) -> FlowResult:
        """Switch the model, or show the confirmation form."""
        if user_input is None:
            return self.async_show_form(step_id="confirm", data_schema=vol.Schema({}))

        entry = self.hass.config_entries.async_get_entry(self._entry_id)
        subentry = entry.subentries.get(self._subentry_id) if entry else None

        # Both can be gone by now -- the issue outlives the thing it is about
        # if someone deletes the entity while the repair is open. Finishing
        # quietly is right: the problem really has been resolved, just not the
        # way this flow intended.
        if entry is None or subentry is None:
            return self.async_create_entry(data={})

        self.hass.config_entries.async_update_subentry(
            entry,
            subentry,
            data={**subentry.data, CONF_MODEL: self._replacement},
        )

        return self.async_create_entry(data={})


async def async_create_fix_flow(
    hass: HomeAssistant,
    issue_id: str,
    data: dict[str, str | int | float | None] | None,
) -> RepairsFlow:
    """Return the flow for one of this integration's repairs."""
    return DeprecatedModelRepairFlow(dict(data or {}))


@callback
def async_clear_issue(hass: HomeAssistant, subentry_id: str) -> None:
    """Withdraw the warning for a subentry."""
    ir.async_delete_issue(hass, DOMAIN, issue_id(subentry_id))
