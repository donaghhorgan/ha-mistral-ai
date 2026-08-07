"""Tests for the cached model list on a loaded entry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.mistral_ai.data import MODEL_CACHE_SECONDS, MistralData


async def test_the_list_is_fetched_once_and_reused() -> None:
    """A second caller inside the window gets the cached list.

    The subentry options form is rendered often -- open it, change your mind,
    open it again -- and each render used to spend a network round trip before
    it could show anything.
    """
    calls = []

    async def _fetch(client: object) -> list[str]:
        calls.append(client)
        return ["mistral-small-latest"]

    data = MistralData(client=MagicMock())

    assert await data.async_models(_fetch) == ["mistral-small-latest"]
    assert await data.async_models(_fetch) == ["mistral-small-latest"]

    assert len(calls) == 1


async def test_a_stale_list_is_fetched_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache expires rather than lasting for the life of the entry.

    Models are added and retired a handful of times a year, so the window is
    long -- but not unbounded, or a model released this morning would stay
    missing until Home Assistant restarted.
    """
    now = 1000.0
    monkeypatch.setattr(
        "custom_components.mistral_ai.data.monotonic", lambda: now
    )

    calls = []

    async def _fetch(client: object) -> list[str]:
        calls.append(client)
        return ["mistral-small-latest"]

    data = MistralData(client=MagicMock())
    await data.async_models(_fetch)

    now += MODEL_CACHE_SECONDS + 1
    await data.async_models(_fetch)

    assert len(calls) == 2


async def test_a_failure_is_not_cached() -> None:
    """A blip must not leave every form empty for the hour.

    The reason async_fetch_model_cards raises rather than returning an empty
    list. An empty list is a plausible answer, so caching one would be
    indistinguishable from an account with no models, and the subentry flow
    would keep aborting long after the API had recovered.
    """
    attempts = []

    async def _fetch(client: object) -> list[str]:
        attempts.append(client)
        if len(attempts) == 1:
            raise TimeoutError
        return ["mistral-small-latest"]

    data = MistralData(client=MagicMock())

    with pytest.raises(TimeoutError):
        await data.async_models(_fetch)

    assert await data.async_models(_fetch) == ["mistral-small-latest"]
    assert len(attempts) == 2


async def test_invalidating_forces_a_refetch() -> None:
    """The escape hatch works, so whatever needs it later can rely on it."""
    calls = []

    async def _fetch(client: object) -> list[str]:
        calls.append(client)
        return ["mistral-small-latest"]

    data = MistralData(client=MagicMock())
    await data.async_models(_fetch)
    data.invalidate_models()
    await data.async_models(_fetch)

    assert len(calls) == 2
