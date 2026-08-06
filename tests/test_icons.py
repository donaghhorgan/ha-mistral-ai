"""Tests for the icon definitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import custom_components.mistral_ai as integration
from homeassistant.helpers import entity_registry as er

from custom_components.mistral_ai.const import DOMAIN

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from pytest_homeassistant_custom_component.common import MockConfigEntry


def _icons() -> dict:
    """Return the parsed icons.json."""
    path = Path(integration.__file__).parent / "icons.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _registered(hass: HomeAssistant) -> dict[str, str | None]:
    """Return the translation key of each of this integration's entities.

    Read from the entity registry rather than from the classes. Home Assistant
    rewrites `_attr_` class attributes into descriptors, so reading
    `SomeEntity._attr_translation_key` gives a property object rather than the
    string -- the registry holds what was actually resolved, which is also
    what the icon lookup uses.
    """
    return {
        entry.domain: entry.translation_key
        for entry in er.async_get(hass).entities.values()
        if entry.platform == DOMAIN
    }


async def test_every_entity_has_an_icon(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """Each entity's translation key resolves to an icon.

    Home Assistant looks an entity icon up by platform domain and translation
    key, so the two have to agree. They are declared in different files, and a
    mismatch is silent: the entity falls back to the platform's generic icon,
    which is what they all did before there was an icons.json at all.
    """
    icons = _icons()["entity"]
    registered = _registered(hass)

    assert set(registered) == {"ai_task", "conversation", "stt", "tts"}

    for domain, translation_key in registered.items():
        assert translation_key is not None, f"{domain} entity has no translation key"
        assert icons[domain][translation_key]["default"].startswith("mdi:")


async def test_no_icons_without_an_entity(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """Nothing is defined for a platform or key that does not exist.

    The other direction of the same mismatch, and the one that outlives a
    platform being renamed or removed.
    """
    registered = _registered(hass)

    for domain, keys in _icons()["entity"].items():
        assert domain in registered
        assert set(keys) == {registered[domain]}
