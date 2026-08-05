"""Shared helpers for the Mistral AI integration."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import httpx
from homeassistant.components.tts import Voice
from mistralai.client.errors import SDKError

from .const import TIMEOUT

if TYPE_CHECKING:
    from mistralai.client import Mistral

_LOGGER = logging.getLogger(__name__)


async def async_list_voices(client: Mistral) -> list[Voice]:
    """Return the voices available to a client, sorted by name.

    Voices are per account -- custom ones are created against it -- so the list
    has to come from the API rather than from a table here. That is the
    difference from google_generative_ai_conversation, which can hard-code its
    voices because they are the same for everyone.

    Shared by the config flow, which offers them as a dropdown, and the
    text-to-speech entity, which reports them to Home Assistant. Both want the
    same names, and a voice that reads differently in the two places would look
    like two different voices.

    A failure is not fatal and returns an empty list. The config form is still
    usable without it, and the speech endpoint picks a voice when none is
    given, so a listing failure should not take the platform down with it.
    """
    try:
        async with asyncio.timeout(TIMEOUT):
            response = await client.audio.voices.list_async()
    except (SDKError, TimeoutError, httpx.HTTPError) as err:
        _LOGGER.debug("Could not list Mistral AI voices: %s", err)
        return []

    voices = []
    for voice in getattr(response, "items", None) or []:
        # Languages are what make one voice distinguishable from twenty others
        # in a dropdown.
        name = voice.name
        if languages := getattr(voice, "languages", None):
            name = f"{name} ({', '.join(languages)})"
        voices.append(Voice(voice_id=voice.id, name=name))

    return sorted(voices, key=lambda voice: voice.name)
