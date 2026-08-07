"""Tests for the translation definitions."""

from __future__ import annotations

import ast
import json
import re
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


def _placeholders_in_translations() -> set[str]:
    """Return every {placeholder} named anywhere in the English strings."""
    names: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, str):
            names.update(re.findall(r"\{(\w+)\}", node))

    walk(_translations())
    return names


def _supplied_placeholders() -> set[str]:
    """Return every placeholder key the source passes to Home Assistant.

    Both keywords, because the two halves of this are separate APIs:
    description_placeholders fills a form's title and description, and
    translation_placeholders fills an exception message or a repair issue.
    Read statically for the same reason as the keys above -- reaching every
    one at runtime would mean provoking every failure the integration has.
    """
    names: set[str] = set()
    keywords = ("description_placeholders", "translation_placeholders")

    for module in _integration_path().glob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg not in keywords:
                    continue
                if isinstance(keyword.value, ast.Dict):
                    names.update(
                        key.value
                        for key in keyword.value.keys
                        if isinstance(key, ast.Constant)
                    )

    return names


def test_every_placeholder_is_supplied_by_the_code() -> None:
    """A message must not name something nothing fills in.

    Home Assistant substitutes only the placeholders it is handed and leaves
    the rest as literal text, so a string referring to {name} that no code
    path supplies renders the braces to the user. That happened twice: the
    reauth dialog (#137), and the deprecated-model repair's confirm step,
    which needs four and was given none.

    Checked as a set rather than per-string, which is looser than ideal --
    it would not catch the right key being supplied at the wrong call site --
    but it does catch a placeholder that nothing anywhere provides, which is
    the failure both of those were.
    """
    missing = _placeholders_in_translations() - _supplied_placeholders()

    assert not missing, f"never supplied by any code path: {sorted(missing)}"
