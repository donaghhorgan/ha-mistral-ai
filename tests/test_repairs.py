"""Tests for the deprecated-model repair."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.helpers import issue_registry as ir

from custom_components.mistral_ai.const import CONF_MODEL, DOMAIN
from custom_components.mistral_ai.repairs import issue_id

from .conftest import DEPRECATED_MODEL, ORPHANED_MODEL

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from homeassistant.core import HomeAssistant
    from pytest_homeassistant_custom_component.common import MockConfigEntry


def _conversation(entry: MockConfigEntry) -> tuple[str, object]:
    """Return the conversation subentry's id and object."""
    return next(
        (subentry_id, subentry)
        for subentry_id, subentry in entry.subentries.items()
        if subentry.subentry_type == "conversation"
    )


async def test_no_issue_when_the_model_is_current(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """A healthy configuration raises nothing."""
    assert not [
        issue for issue in ir.async_get(hass).issues.values() if issue.domain == DOMAIN
    ]


async def test_a_retiring_model_raises_a_fixable_issue(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """An entity already pointing at a retiring model is warned about.

    The dropdown labels these for whoever is choosing one, which helps the
    next person to open the form and nobody who set theirs up a year ago.
    This is the half that reaches them.
    """
    subentry_id, subentry = _conversation(init_integration)
    hass.config_entries.async_update_subentry(
        init_integration, subentry, data={CONF_MODEL: DEPRECATED_MODEL}
    )
    await hass.config_entries.async_reload(init_integration.entry_id)
    await hass.async_block_till_done()

    issue = ir.async_get(hass).async_get_issue(DOMAIN, issue_id(subentry_id))

    assert issue is not None
    assert issue.is_fixable
    assert issue.severity is ir.IssueSeverity.WARNING
    # The successor is taken from the API rather than guessed, which is what
    # makes the fix offerable at all.
    assert issue.translation_placeholders["replacement"] == "mistral-medium-3-5"
    assert issue.translation_placeholders["date"] == "2026-08-31"


async def test_the_repair_switches_the_model_and_clears_the_issue(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Confirming the repair moves the subentry onto the replacement."""
    subentry_id, subentry = _conversation(init_integration)
    hass.config_entries.async_update_subentry(
        init_integration, subentry, data={CONF_MODEL: DEPRECATED_MODEL}
    )
    await hass.config_entries.async_reload(init_integration.entry_id)
    await hass.async_block_till_done()

    from custom_components.mistral_ai.repairs import async_create_fix_flow

    issue = ir.async_get(hass).async_get_issue(DOMAIN, issue_id(subentry_id))
    flow = await async_create_fix_flow(hass, issue_id(subentry_id), issue.data)
    flow.hass = hass

    await flow.async_step_confirm({})
    await hass.async_block_till_done()

    _, updated = _conversation(init_integration)
    assert updated.data[CONF_MODEL] == "mistral-medium-3-5"


async def test_the_issue_is_withdrawn_once_the_model_is_current(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Fixing it by hand clears the warning too.

    Issues are raised and withdrawn on every setup rather than tracked, so
    changing the model yourself -- or Mistral cancelling a retirement --
    clears this without anything having to notice.
    """
    subentry_id, subentry = _conversation(init_integration)
    hass.config_entries.async_update_subentry(
        init_integration, subentry, data={CONF_MODEL: DEPRECATED_MODEL}
    )
    await hass.config_entries.async_reload(init_integration.entry_id)
    await hass.async_block_till_done()
    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id(subentry_id))

    _, subentry = _conversation(init_integration)
    hass.config_entries.async_update_subentry(
        init_integration, subentry, data={CONF_MODEL: "mistral-small-latest"}
    )
    await hass.config_entries.async_reload(init_integration.entry_id)
    await hass.async_block_till_done()

    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id(subentry_id)) is None


async def test_a_retiring_model_with_no_successor_is_not_fixable(
    hass: HomeAssistant, init_integration: MockConfigEntry, mock_client: MagicMock
) -> None:
    """Where the API names no replacement, the issue has no button.

    Offering one would mean guessing where to move someone, which is worse
    than saying so and letting them choose. The unfixable form of the issue
    says exactly that.
    """
    subentry_id, subentry = _conversation(init_integration)
    hass.config_entries.async_update_subentry(
        init_integration, subentry, data={CONF_MODEL: ORPHANED_MODEL}
    )
    await hass.config_entries.async_reload(init_integration.entry_id)
    await hass.async_block_till_done()

    issue = ir.async_get(hass).async_get_issue(DOMAIN, issue_id(subentry_id))

    assert issue is not None
    assert not issue.is_fixable
    assert issue.translation_key == "deprecated_model_no_successor"
