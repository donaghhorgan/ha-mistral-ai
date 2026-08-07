"""Check request payloads against the vendored OpenAPI spec.

Deliberately shallow. It checks three things and nothing else:

- every key exists as a property of the schema
- numbers respect a declared minimum and maximum
- strings respect a declared enum

It does not validate list items, discriminated unions or required fields.
That is a choice rather than a shortcut. The spec describes `messages` and
`tools` as unions with discriminators, and this integration sends simplified
shapes the API is perfectly happy with; a full validator would reject working
requests, and a check that fails on correct code is worse than no check --
it gets marked xfail and takes the useful assertions with it.

The three rules above are exactly the shape of the bugs that shipped: a
parameter the endpoint does not have, and a value outside the range it
accepts. See #113.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

FIXTURE = Path(__file__).parent / "fixtures" / "openapi.json"

# Arguments the SDK takes that never reach the request body.
TRANSPORT_KWARGS = frozenset({"timeout_ms", "retries", "server_url", "http_headers"})


@lru_cache(maxsize=1)
def spec() -> dict[str, Any]:
    """Return the vendored spec."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _resolve(node: Any) -> Any:
    """Follow a $ref to the schema it names."""
    seen = 0
    while isinstance(node, dict) and "$ref" in node and seen < 10:
        node = spec()["components"]["schemas"][node["$ref"].rsplit("/", 1)[-1]]
        seen += 1
    return node


def _properties(schema: Any) -> dict[str, Any]:
    """Return a schema's properties, flattening allOf and anyOf.

    A conversation request is an allOf of a shared base and a variant, so the
    properties of interest live one level down rather than on the schema
    itself.
    """
    schema = _resolve(schema)
    if not isinstance(schema, dict):
        return {}

    properties = dict(schema.get("properties") or {})
    for key in ("allOf", "anyOf", "oneOf"):
        for part in schema.get(key) or []:
            properties.update(_properties(part))

    return properties


def _constraints(schema: Any) -> dict[str, Any]:
    """Return the constraints on a property, seeing through nullable unions.

    Optional fields are written as `anyOf: [{type: number, maximum: 1}, null]`,
    so the bound is on a branch rather than on the property.
    """
    schema = _resolve(schema)
    if not isinstance(schema, dict):
        return {}

    found = {
        key: schema[key] for key in ("minimum", "maximum", "enum") if key in schema
    }
    for branch in schema.get("anyOf") or []:
        for key, value in _constraints(branch).items():
            found.setdefault(key, value)

    return found


def problems(schema_name: str, payload: dict[str, Any], path: str = "") -> list[str]:
    """Return everything wrong with a payload, as readable sentences."""
    properties = _properties({"$ref": f"#/components/schemas/{schema_name}"})
    if not properties:
        return [f"{schema_name} has no properties in the vendored spec"]

    found: list[str] = []

    for key, value in payload.items():
        if key in TRANSPORT_KWARGS:
            continue

        where = f"{path}{key}"

        if key not in properties:
            found.append(
                f"{schema_name} has no field {where!r} -- the endpoint will reject it"
            )
            continue

        limits = _constraints(properties[key])

        if isinstance(value, bool):
            # bool is an int in Python, and no numeric bound applies to it.
            continue

        if isinstance(value, (int, float)):
            if "maximum" in limits and value > limits["maximum"]:
                found.append(
                    f"{where} is {value}, above the maximum of "
                    f"{limits['maximum']} the endpoint accepts"
                )
            if "minimum" in limits and value < limits["minimum"]:
                found.append(
                    f"{where} is {value}, below the minimum of "
                    f"{limits['minimum']} the endpoint accepts"
                )

        if isinstance(value, str) and "enum" in limits and value not in limits["enum"]:
            found.append(f"{where} is {value!r}, which is not one of {limits['enum']}")

        # Nested objects that are schemas in their own right -- completion_args
        # being the one that matters, since it carries its own bounds.
        if isinstance(value, dict):
            nested = _resolve(properties[key])
            for branch in [nested, *(nested.get("anyOf") or [])]:
                resolved = _resolve(branch)
                if isinstance(resolved, dict) and resolved.get("properties"):
                    name = properties[key].get("$ref") or branch.get("$ref")
                    if name:
                        found.extend(
                            problems(name.rsplit("/", 1)[-1], value, path=f"{where}.")
                        )
                    break

    return found
