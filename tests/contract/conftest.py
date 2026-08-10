"""Fixtures for the live contract tests.

These make real, billed requests. They are excluded from the default pytest
run by --ignore in pyproject.toml, so `uv run pytest` stays offline and free.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Generator

BASE_URL = "https://api.mistral.ai/v1"

# The cheapest model that can do everything asked of it here. Nothing in this
# suite depends on the quality of an answer, only on the shape of a response,
# so there is no reason to pay for a larger one.
MODEL = "mistral-small-latest"

# Enough to prove a response has the shape we parse, and not a token more.
MAX_TOKENS = 8

# The speech model, which is a different family from the chat one.
TTS_MODEL = "voxtral-mini-tts-latest"

# A chat model that reports capabilities.reasoning: false, used to prove the
# flag means what the config flow relies on it meaning. Pinned rather than
# -latest deliberately: mistral-medium-latest reasons and this pinned build of
# the same family does not, which is the whole reason the gate is read per
# model id instead of guessed from the name.
NON_REASONING_MODEL = "mistral-medium-2508"

# How many times a request that came back "not now" is sent again, and the
# backoff between tries: 1s, 2s, 4s, for about seven seconds in the worst case.
ATTEMPTS = 4

# The statuses that mean "not now" rather than "no".
#
# 429 is a shared quota. Two pull requests pushed together draw on one account
# and will limit each other.
#
# The 5xx family is the API having a moment, and is here because it happened:
# the weekly run of 2026-08-10 failed one test on a 503 while the other 33
# passed. The body was 167 bytes of text/plain and the gateway spent 2ms on it,
# which is an edge declining to route rather than a model declining a request.
#
# Retrying these does not hide a real outage. A sustained one exhausts the
# attempts and the suite still goes red, carrying the real status code. What it
# stops is a single blip painting the contract suite red, which is worse here
# than elsewhere: this suite going red is supposed to mean the API changed
# under us, and the value of that signal is in how rarely it fires.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class RetryTransport(httpx.AsyncBaseTransport):
    """Retry transient failures under the client, not per call site.

    Attached to the client rather than wrapped around a POST helper so that it
    covers the whole suite. The retry used to live in the `post` fixture, which
    left the tests that reach for `client.get` or `client.stream` -- the voice
    listing, the model listing, both streamed-speech tests -- with no cover at
    all, for no better reason than that they do not post.
    """

    def __init__(self, wrapped: httpx.AsyncBaseTransport) -> None:
        """Wrap the transport that does the real work."""
        self._wrapped = wrapped

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Send the request, trying again while it comes back retryable."""
        for attempt in range(ATTEMPTS):
            response = await self._wrapped.handle_async_request(request)
            if response.status_code not in RETRY_STATUSES or attempt == ATTEMPTS - 1:
                return response
            # Nothing has read the body yet, so this releases the connection
            # rather than discarding a response someone is waiting on.
            await response.aclose()
            await asyncio.sleep(2**attempt)
        return response

    async def aclose(self) -> None:
        """Close the wrapped transport along with this one."""
        await self._wrapped.aclose()


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
def unrestricted_network() -> Generator[None]:
    """Lift pytest-socket's restrictions for the duration of a test.

    The Home Assistant test plugin blocks outbound connections and allows
    127.0.0.1 only. That is the right default everywhere else in this suite,
    and has to be lifted in the one place that means to reach the network.

    `socket_enabled` alone is not enough, and how it fails is worth recording.
    It restores `socket.socket`, while the host allowlist is a separate patch
    over `socket.socket.connect` -- so sockets work and every connection to
    anything but 127.0.0.1 is still refused. That went unnoticed while these
    tests were written because the machine routed through a proxy on
    127.0.0.1: the requests were real and the assertions were real, and the
    allowlist was satisfied by accident. It failed the first time it ran
    somewhere without one.

    `socket_allow_hosts(["api.mistral.ai"])` is the public alternative and is
    worse here. It resolves the hostname to addresses once, and the API is
    behind a CDN that answers with different ones over time, so a connection
    to an address resolved later would be blocked. That is a flaky test rather
    than a safe one.
    """
    import pytest_socket

    pytest_socket._remove_restrictions()  # noqa: SLF001
    try:
        yield
    finally:
        # Put back, so nothing after these tests inherits an open network by
        # accident. They run in their own pytest invocation today, which makes
        # this belt and braces rather than load-bearing.
        pytest_socket.socket_allow_hosts(["127.0.0.1"], allow_unix_socket=True)


@pytest.fixture
async def client(
    api_key: str,
    unrestricted_network: None,  # noqa: ARG001
) -> AsyncGenerator[httpx.AsyncClient]:
    """Return an HTTP client aimed at the API.

    Raw HTTP rather than the SDK, deliberately. The SDK is a thing that can be
    wrong -- it accepted `handoff_execution` for a request the endpoint
    refuses -- so a suite whose job is to check what the endpoint accepts
    should not ask the SDK what that is.

    The network restrictions the rest of the suite runs under are lifted by
    the fixture above, in one place rather than per test.

    Transient failures are retried by the transport, so every request made
    through this client is covered however it was sent.
    """
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60.0,
        transport=RetryTransport(httpx.AsyncHTTPTransport()),
    ) as http:
        yield http


@pytest.fixture
def post(client: httpx.AsyncClient) -> Callable:
    """Return a POST helper, for the JSON body most of these tests send.

    Brevity only. Retries belong to the transport above, where `client.get`
    and `client.stream` get them too.
    """

    async def _post(path: str, payload: dict, **kwargs: object) -> httpx.Response:
        return await client.post(path, json=payload, **kwargs)

    return _post
