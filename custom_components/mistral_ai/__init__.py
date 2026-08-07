"""The Mistral AI integration."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import (
    ConfigEntryAuthFailed,
    ConfigEntryError,
    ConfigEntryNotReady,
)
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.typing import ConfigType

from .client import async_create_client
from .config_flow import (
    CannotConnect,
    Forbidden,
    InvalidAuth,
    async_fetch_model_cards,
)
from .const import CONF_API_KEY, CONF_MODEL, DOMAIN
from .data import MistralData
from .repairs import async_clear_issue, issue_id

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

PLATFORMS: tuple[Platform, ...] = (
    Platform.AI_TASK,
    Platform.CONVERSATION,
    Platform.STT,
    Platform.TTS,
)

type MistralConfigEntry = ConfigEntry[MistralData]


async def async_setup_entry(hass: HomeAssistant, entry: MistralConfigEntry) -> bool:
    """Set up Mistral AI from a config entry."""
    client = await async_create_client(hass, entry.data[CONF_API_KEY])
    data = MistralData(client=client)

    # Verify credentials during setup so that a bad key surfaces as a reauth
    # flow, rather than as a failure on the user's first sentence.
    #
    # Through the cache rather than around it. The listing this makes is the
    # same one the config flow wants, so going through async_models leaves it
    # cached and the first form opened after a restart renders without a round
    # trip. Setup used to call the endpoint directly and drop the response,
    # which meant paying for the same list twice.
    #
    # It is a head start, not an extension: async_models stamps the fetch time
    # now, so an entry loaded two hours ago still refetches on the next render.
    #
    # Reusing async_fetch_model_cards rather than filtering here keeps one
    # definition of what counts as a usable card. Two copies would drift, and
    # a seeded list that did not match an uncached one would show up only as a
    # model missing from a single dropdown.
    try:
        cards = await data.async_models(async_fetch_model_cards)
    except InvalidAuth as err:
        raise ConfigEntryAuthFailed("Invalid Mistral AI API key") from err
    except Forbidden as err:
        # Permanent rather than not-ready: the key authenticated, so retrying
        # the same call on the same account will keep getting the same answer.
        # ConfigEntryError says so and stops, where ConfigEntryNotReady would
        # retry a failure that cannot resolve itself, and ConfigEntryAuthFailed
        # -- which is what this was -- would ask for a replacement key that is
        # not the problem.
        raise ConfigEntryError(
            f"Mistral AI refused the request, and the API key is valid: {err}"
        ) from err
    except CannotConnect as err:
        raise ConfigEntryNotReady(f"Error talking to Mistral AI: {err}") from err

    entry.runtime_data = data

    # Free: the listing above is already in hand.
    _async_review_models(hass, entry, cards)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: MistralConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_update_options(hass: HomeAssistant, entry: MistralConfigEntry) -> None:
    """Reload the config entry when its options change."""
    await hass.config_entries.async_reload(entry.entry_id)


@callback
def _async_review_models(
    hass: HomeAssistant, entry: MistralConfigEntry, cards: list[Any]
) -> None:
    """Warn about any subentry pointing at a model that is being retired.

    The dropdown labels retiring models for whoever is choosing one. That
    helps the next person to open the form and nobody who set theirs up a year
    ago -- so this is the other half: telling someone before the model stops
    working rather than after.

    Raised and withdrawn on every setup rather than tracked, so that changing
    the model, or Mistral cancelling a retirement, clears the warning without
    anything having to notice.
    """
    # Already filtered to cards carrying an id, so only deprecation is checked.
    retiring = {card.id: card for card in cards if getattr(card, "deprecation", None)}

    for subentry_id, subentry in entry.subentries.items():
        card = retiring.get(subentry.data.get(CONF_MODEL))
        if card is None:
            async_clear_issue(hass, subentry_id)
            continue

        replacement = getattr(card, "deprecation_replacement_model", None)
        deprecation = card.deprecation

        ir.async_create_issue(
            hass,
            DOMAIN,
            issue_id(subentry_id),
            data={
                "entry_id": entry.entry_id,
                "subentry_id": subentry_id,
                "replacement": replacement,
            },
            # Fixable only when the API names a successor. Offering a button
            # that has to guess where to move someone would be worse than
            # telling them to choose, which is what the unfixable form of this
            # issue says.
            is_fixable=replacement is not None,
            issue_domain=DOMAIN,
            severity=ir.IssueSeverity.WARNING,
            translation_key=(
                "deprecated_model" if replacement else "deprecated_model_no_successor"
            ),
            translation_placeholders={
                "name": subentry.title,
                "model": subentry.data[CONF_MODEL],
                "replacement": replacement or "",
                "date": deprecation.date().isoformat()
                if hasattr(deprecation, "date")
                else str(deprecation),
            },
        )
