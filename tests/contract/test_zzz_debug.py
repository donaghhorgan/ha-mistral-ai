"""Throwaway: dump full model catalog. Deleted before merge."""

from __future__ import annotations

import httpx


async def test_zzz_dump_model_capabilities(client: httpx.AsyncClient) -> None:
    response = await client.get("/models")
    cards = response.json()["data"]
    lines = [
        f"{c['id']}: caps={c.get('capabilities')} deprecation={c.get('deprecation')}"
        for c in cards
    ]
    raise AssertionError("\n" + "\n".join(sorted(lines)))
