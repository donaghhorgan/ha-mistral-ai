"""Tests for the translation definitions."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import custom_components.mistral_ai as integration


def _integration_path() -> Path:
    """Return the directory the integration lives in."""
    return Path(integration.__file__).parent


def _translations() -> dict:
    """Return the parsed English translations."""
    path = _integration_path() / "translations" / "en.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _raised_translation_keys() -> set[str]:
    """Return every translation_key passed to an exception in the source.

    Parsed rather than exercised, because reaching each of these at runtime
    would mean provoking ten different API failures. The keyword is only ever
    written as a literal, so reading it back is exact.
    """
    keys: set[str] = set()

    # translation_key is the direct form. transport_key is the indirection:
    # the shared error helper takes one and passes it on, so the literal
    # appears at the call site or as the parameter's default rather than
    # beside the raise.
    names = ("translation_key", "transport_key")

    for module in _integration_path().glob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                keys.update(
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg in names and isinstance(keyword.value, ast.Constant)
                )
            elif isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
                keys.update(
                    default.value
                    for argument, default in zip(
                        node.args.kwonlyargs, node.args.kw_defaults, strict=True
                    )
                    if argument.arg in names and isinstance(default, ast.Constant)
                )

    return keys


def test_every_raised_error_has_a_message() -> None:
    """Every translation key the code raises has a message to render.

    A missing one is not an error anywhere: Home Assistant shows the user the
    bare key instead of a sentence, which looks like a bug in the integration
    and says nothing about what went wrong.
    """
    messages = _translations()["exceptions"]

    # The config flow uses translation keys for its selector options too, and
    # those live under `selector` rather than `exceptions`.
    selectors = set(_translations().get("selector", {}))

    for key in _raised_translation_keys() - selectors:
        assert key in messages, f"no message for translation key {key!r}"
        assert messages[key]["message"]


def test_no_messages_without_an_error() -> None:
    """Nothing is translated for an error that is never raised.

    The other direction, and the one that outlives an error being reworded or
    removed.
    """
    raised = _raised_translation_keys()

    for key in _translations()["exceptions"]:
        assert key in raised, f"message {key!r} is never raised"
