"""The requests we build, and the limits we build them with, match the spec.

The payload half of this runs from conftest, against every call the whole
suite makes. What is here is the half a payload check cannot do: assert the
constants the integration limits itself with are the ones the API declares.

The difference matters. A payload check only sees a bad bound if some test
happens to send a value past it, and no test sends the maximum. These assert
the bound itself, so they also fail if Mistral tightens a limit under us --
which is the failure the weekly spec refresh exists to surface.
"""

from __future__ import annotations

import pytest

from custom_components.mistral_ai.const import (
    MAX_TEMPERATURE,
    SUBENTRY_TYPE_AI_TASK_DATA,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_STT,
    VOICE_PAGE_SIZE,
)

from .openapi import _constraints, _properties, spec


def _body_properties(path: str, method: str = "post") -> dict:
    """Return the request body properties an endpoint declares."""
    content = spec()["paths"][path][method]["requestBody"]["content"]
    # Transcription is multipart; everything else here is JSON.
    schema = next(iter(content.values()))["schema"]
    return _properties(schema)


def _maximum(properties: dict, field: str) -> float:
    """Return the maximum an endpoint declares for a field."""
    assert field in properties, f"no {field} in this request body"
    limits = _constraints(properties[field])
    assert "maximum" in limits, f"{field} declares no maximum"
    return limits["maximum"]


def test_ai_task_temperature_matches_chat_completions() -> None:
    """An AI task cannot be configured past what chat completions accepts."""
    assert MAX_TEMPERATURE[SUBENTRY_TYPE_AI_TASK_DATA] == _maximum(
        _body_properties("/v1/chat/completions"), "temperature"
    )


def test_transcription_declares_no_temperature_maximum() -> None:
    """A known gap, asserted so that closing it is noticed.

    Transcription rejects a temperature above 1.5 -- established with a real
    request -- and the spec says nothing about it: the property is a plain
    nullable number with no bound. So MAX_TEMPERATURE for speech-to-text
    cannot be checked against the spec the way the other two can, and the only
    thing holding it right is the live contract suite in #114.

    Asserted as an absence rather than left out, because if Mistral ever
    declares the bound this fails, and someone should then check it agrees
    with the 1.5 we measured and delete this test.
    """
    limits = _constraints(_body_properties("/v1/audio/transcriptions")["temperature"])

    assert "maximum" not in limits
    assert MAX_TEMPERATURE[SUBENTRY_TYPE_STT] == 1.5


def test_conversation_temperature_matches_completion_args() -> None:
    """Conversation agents are bound by the stricter of the two endpoints.

    Web search moves them from chat completions to the conversations endpoint,
    and it is a checkbox on the same form, so the limit that applies is
    whichever is lower -- not the one for the endpoint they happen to use
    while the box is unticked.
    """
    conversations = _properties(
        _body_properties("/v1/conversations")["completion_args"]
    )
    chat = _body_properties("/v1/chat/completions")

    limit = min(_maximum(conversations, "temperature"), _maximum(chat, "temperature"))

    assert MAX_TEMPERATURE[SUBENTRY_TYPE_CONVERSATION] == limit


def test_voice_page_size_is_the_documented_maximum() -> None:
    """Voices are read a page at a time, at the largest page allowed.

    The default is ten, which is how the dropdown came to show 10 of an
    account's 31 voices. Asking for the maximum keeps the number of requests
    down; asking for more than the maximum would be rejected.
    """
    parameters = spec()["paths"]["/v1/audio/voices"]["get"]["parameters"]
    limit = next(p for p in parameters if p["name"] == "limit")

    assert VOICE_PAGE_SIZE == limit["schema"]["maximum"]


def test_every_endpoint_we_call_is_in_the_fixture() -> None:
    """The trimmed fixture still covers every endpoint the integration uses.

    The fixture is generated from a list in scripts/update_openapi_fixture.py.
    Adding a call site without adding it there would leave that request
    unvalidated, and silently -- which is the state this whole check exists to
    get out of.
    """
    from scripts.update_openapi_fixture import USED_ENDPOINTS

    for method, path in USED_ENDPOINTS:
        assert path in spec()["paths"], f"{path} missing from the fixture"
        assert method in spec()["paths"][path], f"{method} {path} missing"
