"""Tests for the diagnostics dump."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.components.diagnostics import REDACTED

from custom_components.mistral_ai.const import (
    CONF_API_KEY,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_VOICE,
)
from custom_components.mistral_ai.diagnostics import (
    async_get_config_entry_diagnostics,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from pytest_homeassistant_custom_component.common import MockConfigEntry


async def test_diagnostics_carry_what_a_bug_report_needs(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """The dump answers the questions a report would otherwise have to ask.

    Which model, on which subentry type, against which SDK version. The SDK
    version comes from the installed package rather than the manifest, which
    only pins a floor and so says nothing about what actually resolved.
    """
    diagnostics = await async_get_config_entry_diagnostics(hass, init_integration)

    assert diagnostics["sdk_version"] not in ("", "unknown")
    assert diagnostics["entry"]["state"] == "loaded"

    types = {sub["type"] for sub in diagnostics["subentries"].values()}
    assert types == {"conversation", "ai_task_data", "stt", "tts"}

    models = {sub["data"].get(CONF_MODEL) for sub in diagnostics["subentries"].values()}
    assert None not in models

    assert diagnostics["entities"]
    assert all("entity_id" in entity for entity in diagnostics["entities"])


async def test_diagnostics_redact_the_key_the_prompt_and_the_voice(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """Nothing identifying survives into a file meant for public issues.

    The key is obvious. The prompt is the one worth a test: a custom one names
    the house, the rooms and the people in them, and this file exists to be
    pasted into a bug report.
    """
    entry = next(
        subentry
        for subentry in init_integration.subentries.values()
        if subentry.subentry_type == "conversation"
    )
    hass.config_entries.async_update_subentry(
        init_integration,
        entry,
        data={
            CONF_MODEL: "mistral-small-latest",
            CONF_PROMPT: "You are the assistant for the Smith family at 4 Elm Road.",
            CONF_VOICE: "a-custom-voice-id",
        },
    )
    await hass.async_block_till_done()

    diagnostics = await async_get_config_entry_diagnostics(hass, init_integration)

    assert diagnostics["entry"]["data"][CONF_API_KEY] == REDACTED

    conversation = next(
        sub
        for sub in diagnostics["subentries"].values()
        if sub["type"] == "conversation"
    )
    assert conversation["data"][CONF_PROMPT] == REDACTED
    assert conversation["data"][CONF_VOICE] == REDACTED
    assert "Elm Road" not in str(diagnostics)

    # The model is deliberately not redacted -- it is the single most useful
    # field here and identifies nobody.
    assert conversation["data"][CONF_MODEL] == "mistral-small-latest"
