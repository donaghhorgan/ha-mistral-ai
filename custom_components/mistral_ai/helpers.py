"""Shared helpers for the Mistral AI integration."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import httpx
from homeassistant.components.tts import Voice
from mistralai.client.errors import SDKError

from .const import TIMEOUT, VOICE_PAGE_SIZE

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
    voices = []
    offset = 0

    try:
        async with asyncio.timeout(TIMEOUT):
            while True:
                response = await client.audio.voices.list_async(
                    limit=VOICE_PAGE_SIZE, offset=offset
                )

                items = getattr(response, "items", None) or []
                for voice in items:
                    # Languages are what make one voice distinguishable from
                    # twenty others in a dropdown.
                    name = voice.name
                    if languages := getattr(voice, "languages", None):
                        name = f"{name} ({', '.join(languages)})"
                    voices.append(Voice(voice_id=voice.id, name=name))

                # A short page is the last page. Checked before the total,
                # because the total is only a hint and a wrong one would either
                # cut the list short or loop forever.
                if len(items) < VOICE_PAGE_SIZE:
                    break

                offset += len(items)

                total = getattr(response, "total", None)
                if isinstance(total, int) and offset >= total:
                    break
    except (SDKError, TimeoutError, httpx.HTTPError) as err:
        _LOGGER.debug("Could not list Mistral AI voices: %s", err)
        # Whatever arrived before the failure is still better than nothing,
        # and an empty list is what a first-page failure gives.
        return sorted(voices, key=lambda voice: voice.name)

    return sorted(voices, key=lambda voice: voice.name)


def entry_chunks(outputs: Any) -> Any:
    """Yield the content chunks of every message entry in a response.

    A conversation returns a list of entries rather than one message. Only
    message entries carry content; tool.execution and function.call entries
    describe work rather than saying anything, so they are skipped rather than
    special-cased.
    """
    for entry in outputs:
        content = getattr(entry, "content", None)
        if content is None or isinstance(content, str):
            continue
        yield from content


def outputs_text(outputs: Any) -> str | None:
    """Return the text a response carries, ignoring any non-text chunks."""
    parts = [
        content
        for entry in outputs
        if isinstance(content := getattr(entry, "content", None), str)
    ]
    # isinstance rather than a truth test: a chunk carrying a non-string in
    # `text` would otherwise fail the join with a TypeError, which is a poor
    # way to report that a response came back in an unexpected shape.
    parts.extend(
        text
        for chunk in entry_chunks(outputs)
        if isinstance(text := getattr(chunk, "text", None), str)
    )
    return "".join(parts) or None


def clamped_temperature(value: float, maximum: float) -> float:
    """Return a temperature the endpoint will accept.

    The form used to allow up to 2.0, and every endpoint rejects that, so
    settings above the real limit are already stored in the wild. Lowering the
    slider stops new ones being made and does nothing about those -- the value
    is only re-validated when someone opens the form again, and a conversation
    agent nobody has reconfigured since would 422 on every sentence.

    Clamping is the lesser of the two evils here: the stored value cannot
    produce a working request, so honouring it exactly means failing. Logged at
    warning rather than done quietly, because the form will still show the old
    number until it is next opened, and that gap is worth being able to find in
    the log.
    """
    if value <= maximum:
        return value

    _LOGGER.warning(
        "Temperature %s is above the maximum of %s this endpoint accepts, "
        "using %s instead -- reconfigure the entity to set it permanently",
        value,
        maximum,
        maximum,
    )
    return maximum
