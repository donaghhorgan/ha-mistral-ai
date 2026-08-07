#!/usr/bin/env python3
"""
Check that every declared dependency is a decision Dependabot can act on.

`.github/dependabot.yml` uses an allowlist, because the blocklist can never
be complete: Home Assistant pins some 46 packages exactly and
pytest-homeassistant-custom-component another 27, and that set changes with
every Home Assistant release.

The cost of an allowlist is silence. A package added to pyproject.toml and
not added to the allowlist is never updated, and nothing says so -- no failed
check, no stale-version warning, just a floor that quietly drifts. That is
not hypothetical: ha-ffmpeg and mutagen were added to the dev group and
missed, and mutagen's floor was a release behind before anyone noticed.

So this asks a narrower question than "should this move?", which needs
judgement: it asks whether somebody has *decided*. A package is covered if it
is allowed, ignored, or named below as deliberately neither. Adding a
dependency without touching any of those three is the mistake this catches.
"""

import re
import sys
import tomllib
from pathlib import Path

# Packages that are deliberately neither allowed nor ignored, with the reason
# they need no entry. Kept here rather than inferred, because "Dependabot must
# not touch this" is a decision and decisions should be written down.
EXPECTED_ABSENT = {
    "homeassistant": (
        "the supported floor, checked against hacs.json and README.md. "
        "Raising it is a decision rather than an upgrade."
    ),
    "pytest-homeassistant-custom-component": (
        "chooses the Home Assistant version everything is tested against. "
        "New releases are noticed by the test-latest job in ci.yml."
    ),
    "pillow": (
        "Home Assistant pins it exactly, so allowing it makes the project "
        "unresolvable rather than merely outdated."
    ),
}


def declared_packages(pyproject: Path) -> set[str]:
    """Return every package named in the project or its dependency groups."""
    data = tomllib.loads(pyproject.read_text())

    entries: list[str] = list(data.get("project", {}).get("dependencies", []))
    for group in data.get("dependency-groups", {}).values():
        entries.extend(entry for entry in group if isinstance(entry, str))

    names = set()
    for entry in entries:
        # Strip any version specifier, extras or marker: the name is all that
        # Dependabot matches on.
        name = re.split(r"[<>=!~;\[]", entry, maxsplit=1)[0].strip()
        if name:
            names.add(name.lower())

    return names


def configured_packages(dependabot: Path) -> tuple[set[str], set[str]]:
    """Return the allowed and ignored package names.

    Read with a regex rather than a YAML parser, because pyyaml is not a
    direct dependency of this project and adding one for two lists would be a
    worse trade than a pattern over a file whose shape is fixed by
    Dependabot's own schema.
    """
    text = dependabot.read_text()

    def names_under(section: str) -> set[str]:
        # Everything from the section header to the next line at the same
        # indentation that is not part of the list.
        match = re.search(
            rf"^(\s*){section}:\s*$(.*?)(?=^\1\S)", text, re.MULTILINE | re.DOTALL
        )
        if not match:
            return set()
        return {
            found.lower()
            for found in re.findall(
                r"^\s*-\s*dependency-name:\s*(\S+)", match.group(2), re.MULTILINE
            )
        }

    return names_under("allow"), names_under("ignore")


def main() -> int:
    """Report any declared package Dependabot has no decision for."""
    root = Path(__file__).parent.parent

    print("Checking Dependabot covers every declared dependency...")
    print()

    declared = declared_packages(root / "pyproject.toml")
    allowed, ignored = configured_packages(root / ".github" / "dependabot.yml")

    if not allowed:
        print("❌ Found no allowlist in .github/dependabot.yml")
        print("   The regex above may no longer match the file's shape.")
        return 1

    print(f"Declared: {len(declared)}   allowed: {len(allowed)}   ", end="")
    print(f"ignored: {len(ignored)}")
    print()

    uncovered = declared - allowed - ignored - set(EXPECTED_ABSENT)
    if uncovered:
        print("❌ No Dependabot decision for:")
        print()
        for name in sorted(uncovered):
            print(f"  {name}")
        print()
        print("Recommendation:")
        print("Add each to the `allow` list in .github/dependabot.yml if it is")
        print("free to move, to `ignore` if it resolves and then breaks at")
        print("runtime, or to EXPECTED_ABSENT in this script -- with the reason")
        print("-- if Dependabot should leave it alone entirely.")
        print()
        print("An allowlist fails quietly: an omitted package is simply never")
        print("updated, which is why this check exists.")
        return 1

    # A name that no longer exists is worth reporting too, though not as an
    # error: an entry left behind after a dependency is dropped is clutter
    # rather than a fault, and failing on it would block the commit that
    # removed the dependency.
    stale = (allowed | set(EXPECTED_ABSENT)) - declared
    if stale:
        print(f"⚠️  Configured but no longer declared: {', '.join(sorted(stale))}")
        print()

    print("✅ Every declared dependency is allowed, ignored, or explicitly not")
    print("   Dependabot's to touch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
