"""Fixtures for the live contract tests.

These make real, billed requests. They are excluded from the default pytest
run by --ignore in pyproject.toml, so `uv run pytest` stays offline and free.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

BASE_URL = "https://api.mistral.ai/v1"

# The cheapest model that can do everything asked of it here. Nothing in this
# suite depends on the quality of an answer, only on the shape of a response,
# so there is no reason to pay for a larger one.
MODEL = "mistral-small-latest"

# Enough to prove a response has the shape we parse, and not a token more.
MAX_TOKENS = 8

# The speech model, which is a different family from the chat one.
TTS_MODEL = "voxtral-mini-tts-latest"


@pytest.fixture(scope="session")
def api_key() -> str:
    """Return the API key, or skip the whole suite.

    Skipped rather than failed when absent, because a pull request from a fork
    cannot see repository secrets. Failing there would paint every fork's pull
    request red for a reason its author cannot fix, and a check that is red for
    everyone is one people learn to scroll past.
    """
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        pytest.skip("MISTRAL_API_KEY is not set")
    return key


@pytest.fixture
async def client(
    api_key: str,
    socket_enabled: None,  # noqa: ARG001
) -> AsyncGenerator[httpx.AsyncClient]:
    """Return an HTTP client aimed at the API.

    Raw HTTP rather than the SDK, deliberately. The SDK is a thing that can be
    wrong -- it accepted `handoff_execution` for a request the endpoint
    refuses -- so a suite whose job is to check what the endpoint accepts
    should not ask the SDK what that is.

    socket_enabled is required because the Home Assistant test plugin blocks
    network access in tests, which is the right default for every other test
    here and has to be opted out of exactly once, in the one place that means
    to reach the network.
    """
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60.0,
    ) as http:
        yield http


@pytest.fixture
def post(client: httpx.AsyncClient) -> Callable:
    """Return a POST helper that retries a rate limit rather than failing.

    Two pull requests pushed together share an account quota and will limit
    each other. A false red on a shared limit is indistinguishable from a real
    failure to whoever reads it, and teaches them the suite is unreliable.
    """

    async def _post(path: str, payload: dict, **kwargs: object) -> httpx.Response:
        for attempt in range(4):
            response = await client.post(path, json=payload, **kwargs)
            if response.status_code != 429:
                return response
            await _sleep(2**attempt)
        return response

    return _post


async def _sleep(seconds: float) -> None:
    """Sleep between retries."""
    import asyncio

    await asyncio.sleep(seconds)
