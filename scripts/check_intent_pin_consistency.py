#!/usr/bin/env python3
"""
Check that the intent package pins match the installed Home Assistant.

Home Assistant pins hassil and home-assistant-intents to exact versions in
the conversation component's manifest, and the pins differ between releases.
This project therefore pins them per Home Assistant dependency group in
pyproject.toml rather than sharing one constraint.

Nothing else keeps those two in step: bumping
pytest-homeassistant-custom-component changes the Home Assistant version
without touching the pins beside it. This script compares the pins for the
currently synced group against what that Home Assistant release actually
declares.
"""

import json
import re
import sys
import tomllib
from pathlib import Path

# The packages Home Assistant pins exactly and we therefore mirror.
PACKAGES = ("hassil", "home-assistant-intents")

# Maps the pytest-homeassistant-custom-component pin in a group to the group
# name, so the script can tell which group is currently installed.
PHCC = "pytest-homeassistant-custom-component"


def installed_home_assistant() -> tuple[str, dict[str, str]] | None:
    """Return the installed HA version and what it pins, or None."""
    try:
        import homeassistant.components.conversation as conversation
        from homeassistant.const import __version__ as ha_version
    except ImportError as err:
        print(f"❌ Could not import Home Assistant: {err}")
        return None

    manifest = Path(conversation.__file__).parent / "manifest.json"
    try:
        requirements = json.loads(manifest.read_text())["requirements"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as err:
        print(f"❌ Could not read {manifest}: {err}")
        return None

    pinned = {}
    for requirement in requirements:
        match = re.match(r"^([A-Za-z0-9_.-]+)==(.+)$", requirement.strip())
        if match and match.group(1) in PACKAGES:
            pinned[match.group(1)] = match.group(2)

    return ha_version, pinned


def group_pins(pyproject: Path) -> dict[str, dict[str, str]]:
    """Return the pinned versions declared by each dependency group."""
    data = tomllib.loads(pyproject.read_text())
    groups = data.get("dependency-groups", {})

    pins: dict[str, dict[str, str]] = {}
    for name, entries in groups.items():
        found = {}
        for entry in entries:
            if not isinstance(entry, str):
                continue
            match = re.match(r"^([A-Za-z0-9_.-]+)==(.+?)\s*(?:#.*)?$", entry.strip())
            if match and match.group(1) in (*PACKAGES, PHCC):
                found[match.group(1)] = match.group(2)
        if PHCC in found:
            pins[name] = found

    return pins


def main() -> int:
    """Compare the installed Home Assistant against the group pins."""
    workspace_root = Path(__file__).parent.parent

    print("Checking intent package pins against Home Assistant...")
    print()

    installed = installed_home_assistant()
    if installed is None:
        return 1

    ha_version, declared = installed
    if not declared:
        print(f"❌ Home Assistant {ha_version} pins none of {', '.join(PACKAGES)}")
        print("   The conversation component's requirements may have changed.")
        return 1

    pins = group_pins(workspace_root / "pyproject.toml")
    if not pins:
        print("❌ Found no dependency group pinning Home Assistant")
        return 1

    print(f"Installed Home Assistant: {ha_version}")
    for package, version in sorted(declared.items()):
        print(f"  declares {package}=={version}")
    print()

    # Only the group matching the installed Home Assistant can be checked; the
    # others describe a version that is not present in this environment.
    matching = [
        name
        for name, found in pins.items()
        if all(found.get(package) == version for package, version in declared.items())
    ]
    if matching:
        print(f"✅ Group '{matching[0]}' matches Home Assistant {ha_version}")
        return 0

    print(f"❌ No dependency group matches Home Assistant {ha_version}")
    print()
    for name, found in sorted(pins.items()):
        pinned = ", ".join(f"{p}=={found.get(p, '(unpinned)')}" for p in PACKAGES)
        print(f"  {name}: {pinned}")
    print()
    print("Recommendation:")
    print("Update the group whose Home Assistant version is installed so that")
    print("its hassil and home-assistant-intents pins match the values above.")
    print("Home Assistant pins them exactly, and they are not interchangeable")
    print("across releases.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
