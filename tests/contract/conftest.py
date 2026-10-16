"""Fixtures for the live contract tests.

These make real, billed requests. They are excluded from the default pytest
run by --ignore in pyproject.toml, so `uv run pytest` stays offline and free.
"""

from __future__ import annotations

import os
import socket
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Generator

BASE_URL = "https://api.mistral.ai/v1"

# The resolver as it was before any test ran. Newer releases of the Home
# Assistant test plugin replace these before every test with versions that
# refuse any hostname, so this module captures the real ones while it is
# being imported -- collection happens before the first test's setup, so
# nothing has replaced them yet.
_REAL_RESOLVER = {
    name: getattr(socket, name)
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex")
}

# The cheapest model that can do everything asked of it here. Nothing in this
# suite depends on the quality of an answer, only on the shape of a response,
# so there is no reason to pay for a larger one.
MODEL = "mistral-small-latest"

# Enough to prove a response has the shape we parse, and not a token more.
MAX_TOKENS = 8

# The speech model, which is a different family from the chat one.
TTS_MODEL = "voxtral-mini-tts-latest"

# A chat model that reports capabilities.reasoning: false, used to prove the
# flag means what the config flow relies on it meaning.
#
# Previously pinned to mistral-medium-2508, a snapshot of the same "medium"
# family as mistral-medium-latest that did not reason -- which was the whole
# point: it showed the gate had to be read per model id rather than guessed
# from the name. Mistral retired that snapshot on 2026-08-31, and every
# current mistral-medium snapshot (dated or -latest) now reports
# reasoning: true, so that same-family contrast is no longer available to
# test against. mistral-large-2512 is the closest live substitute: a current,
# non-deprecated, general-purpose chat model that still reports
# reasoning: false. Pinned to a dated snapshot rather than -latest anyway, so
# a future Mistral release cannot silently flip this fixture's premise again
# without the pin itself going stale first.
NON_REASONING_MODEL = "mistral-large-2512"


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
def unrestricted_network(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Lift the test plugin's network restrictions for the duration of a test.

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

    Newer releases of the plugin also block DNS, by replacing the resolver
    functions in `socket` with ones that raise "DNS resolution disabled in
    tests" for any hostname. pytest-socket knows nothing of that, so the real
    resolver is put back too, and monkeypatch restores the blocking one after.
    The proxy hid this as well: through it, only the proxy's address is ever
    resolved, so every contract test passed locally and failed in CI.
    """
    import pytest_socket

    pytest_socket._remove_restrictions()  # noqa: SLF001
    for name, function in _REAL_RESOLVER.items():
        monkeypatch.setattr(socket, name, function)
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
