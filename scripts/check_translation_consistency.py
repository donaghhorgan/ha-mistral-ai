#!/usr/bin/env python3
"""
Check that every translation agrees with the English source.

`en.json` is the source of truth: it is written by hand alongside the code,
and every other file is a translation of it. Three things can drift, and none
of them fail loudly at runtime -- Home Assistant falls back to English for a
missing key and renders a broken placeholder as literal braces.

    Missing keys      a string added to en.json and not to the others, which
                      shows in English and looks like an oversight rather
                      than a fallback.
    Extra keys        a string removed from en.json and left behind, which
                      does nothing and survives forever because nothing reads
                      it.
    Placeholder drift the important one. "{model}" translated, dropped or
                      mistyped renders as the literal text to the user, which
                      is the fault #137 and #159 arrived at by two other
                      routes.

Translations here are best-effort and not reviewed by native speakers, so
this cannot check whether the language is any good. It checks the parts that
are mechanical, which are also the parts that break silently.
"""

import json
import re
import sys
from pathlib import Path

PLACEHOLDER = re.compile(r"\{(\w+)\}")

# Terms that must survive translation verbatim, because they name something
# outside this integration: a brand, a hostname the user has to type, or a
# sampling parameter Mistral's own documentation calls by that name. A
# translated "Top-p" is a setting nobody can look up.
#
# These are matched against the *text*, not the keys. Naming a key here
# achieves nothing -- "top_p" is what the setting is called in en.json's
# structure, and never appears in a string a user reads.
# "Mistral" rather than "Mistral AI", because German writes the brand as a
# hyphenated compound -- "Mistral-AI-API-Schlüssel" -- and that is correct
# orthography rather than drift. The token is the part that must not become
# "Мистраль".
DO_NOT_TRANSLATE = ("console.mistral.ai", "Mistral", "Top-p")


def flatten(node: object, path: str = "") -> dict[str, str]:
    """Return every string in a translation file, keyed by its full path."""
    found: dict[str, str] = {}

    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            found.update(flatten(value, child))
    elif isinstance(node, str):
        found[path] = node

    return found


def check(english: dict[str, str], other: dict[str, str]) -> list[str]:
    """Return every disagreement between a translation and the English."""
    problems = []

    if missing := sorted(set(english) - set(other)):
        problems.append(f"missing {len(missing)} key(s): {', '.join(missing[:5])}")
        if len(missing) > 5:
            problems[-1] += f", and {len(missing) - 5} more"

    if extra := sorted(set(other) - set(english)):
        problems.append(f"{len(extra)} key(s) not in en.json: {', '.join(extra[:5])}")

    for key in sorted(set(english) & set(other)):
        expected = set(PLACEHOLDER.findall(english[key]))
        actual = set(PLACEHOLDER.findall(other[key]))
        if expected != actual:
            problems.append(
                f"{key}: placeholders {sorted(expected)} became {sorted(actual)}"
            )

        for term in DO_NOT_TRANSLATE:
            if term in english[key] and term not in other[key]:
                problems.append(f"{key}: '{term}' must not be translated")

    return problems


def main() -> int:
    """Compare every translation against en.json."""
    root = Path(__file__).parent.parent
    directory = root / "custom_components" / "mistral_ai" / "translations"

    print("Checking translations against en.json...")
    print()

    source = directory / "en.json"
    if not source.exists():
        print(f"❌ No English source at {source}")
        return 1

    english = flatten(json.loads(source.read_text(encoding="utf-8")))
    print(f"en.json: {len(english)} strings")

    others = sorted(p for p in directory.glob("*.json") if p.name != "en.json")
    if not others:
        print()
        print("✅ No translations to check")
        return 0

    failed = False
    for path in others:
        try:
            other = flatten(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as err:
            print(f"❌ {path.name}: not valid JSON -- {err}")
            failed = True
            continue

        if problems := check(english, other):
            failed = True
            print(f"❌ {path.name}")
            for problem in problems:
                print(f"     {problem}")
        else:
            print(f"✅ {path.name}: {len(other)} strings")

    if failed:
        print()
        print("Recommendation:")
        print("Translations mirror en.json exactly -- same keys, same")
        print("placeholders. A missing key falls back to English silently, and")
        print("a dropped placeholder renders as literal braces to the user.")
        return 1

    print()
    print("✅ Every translation agrees with en.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
