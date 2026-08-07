#!/usr/bin/env python3
"""Regenerate the trimmed OpenAPI fixture the request tests validate against.

The published spec is a megabyte covering 213 endpoints. This integration
calls eight of them, so the fixture keeps those and the schemas they reach,
and nothing else.

Trimming is not about disk. It is about the weekly drift check: a diff of the
whole spec is unreadable and would be approved without being read, where a
diff of only the parts we depend on is a few lines that mean something.

Usage:

    python scripts/update_openapi_fixture.py            # fetch the live spec
    python scripts/update_openapi_fixture.py spec.yaml  # use a local copy
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

SPEC_URL = "https://docs.mistral.ai/openapi.yaml"

FIXTURE = Path(__file__).parent.parent / "tests" / "fixtures" / "openapi.json"

# Every endpoint the integration calls. Adding a call site means adding it
# here, and the request tests fail loudly on an endpoint they have no schema
# for rather than skipping it -- a validator that silently ignores what it
# does not recognise is worse than no validator, because it reads as a pass.
USED_ENDPOINTS = (
    ("get", "/v1/models"),
    ("get", "/v1/models/{model_id}"),
    ("post", "/v1/chat/completions"),
    ("post", "/v1/conversations"),
    ("get", "/v1/files/{file_id}/content"),
    ("post", "/v1/audio/transcriptions"),
    ("post", "/v1/audio/speech"),
    ("get", "/v1/audio/voices"),
)


def _referenced_schemas(node: Any, found: set[str], schemas: dict) -> None:
    """Collect every schema name reachable from a node, following $ref."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            name = ref.rsplit("/", 1)[-1]
            if name not in found:
                found.add(name)
                # Recurse into the target: schemas reference other schemas,
                # and a partial closure would fail to resolve at test time.
                _referenced_schemas(schemas.get(name, {}), found, schemas)
        for value in node.values():
            _referenced_schemas(value, found, schemas)
    elif isinstance(node, list):
        for item in node:
            _referenced_schemas(item, found, schemas)


def trim(spec: dict) -> dict:
    """Return the spec reduced to the endpoints this integration calls."""
    schemas = spec["components"]["schemas"]

    paths: dict[str, dict] = {}
    for method, path in USED_ENDPOINTS:
        operation = spec["paths"].get(path, {}).get(method)
        if operation is None:
            raise SystemExit(f"{method.upper()} {path} is not in the spec")
        paths.setdefault(path, {})[method] = operation

    reachable: set[str] = set()
    _referenced_schemas(paths, reachable, schemas)

    return {
        "openapi": spec.get("openapi"),
        "info": spec.get("info"),
        "paths": paths,
        "components": {"schemas": {name: schemas[name] for name in sorted(reachable)}},
    }


def main() -> None:
    """Fetch or read the spec, trim it, and write the fixture."""
    if len(sys.argv) > 1:
        raw = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        # Checked rather than assumed, because urlopen will happily follow
        # file: and other schemes, and this reads whatever it is given into a
        # fixture the tests then trust.
        if not SPEC_URL.startswith("https://"):
            raise SystemExit(f"{SPEC_URL} is not https")
        with urllib.request.urlopen(SPEC_URL, timeout=60) as response:  # noqa: S310  # nosec B310
            raw = response.read().decode("utf-8")

    trimmed = trim(yaml.safe_load(raw))

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    # Sorted and indented so a drift diff is line-oriented and reviewable.
    FIXTURE.write_text(
        json.dumps(trimmed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"{FIXTURE}: {len(trimmed['paths'])} paths, "
        f"{len(trimmed['components']['schemas'])} schemas, "
        f"{FIXTURE.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
