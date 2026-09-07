"""Throwaway: dump model catalog capabilities. Deleted before merge."""

from __future__ import annotations

import httpx


async def test_zzz_dump_model_capabilities(client: httpx.AsyncClient) -> None:
    response = await client.get("/models")
    cards = response.json()["data"]
    lines = [
        f"{c['id']}: reasoning={c.get('capabilities', {}).get('reasoning')} "
        f"deprecation={c.get('deprecation')}"
        for c in cards
        if "medium" in c["id"] or "small" in c["id"]
    ]
    raise AssertionError("\n" + "\n".join(sorted(lines)))
