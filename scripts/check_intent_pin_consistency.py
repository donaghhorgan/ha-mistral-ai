#!/usr/bin/env python3
"""
Check that the intent package pins match the installed Home Assistant.

Home Assistant pins hassil and home-assistant-intents to exact versions in
the conversation component's manifest, and the pins differ between releases.
This project therefore pins them beside each Home Assistant version rather
than sharing one constraint.

There are two places that happens. The current version is a dependency group
in pyproject.toml; the oldest supported version is resolved on the fly by the
Test (ha-minimum) job, so its pins live in the env block of ci.yml. Both are
read here, because the failure is the same either way.

Nothing else keeps those two in step: bumping
pytest-homeassistant-custom-component changes the Home Assistant version
without touching the pins beside it. This script compares the pins for the
currently installed Home Assistant against what that release actually
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


def workflow_pins(workflow: Path) -> dict[str, dict[str, str]]:
    """Return the pins the minimum-version CI job resolves with.

    Those pins used to be a dependency group and are now environment
    variables, because carrying a year-old resolution in pyproject.toml made
    the project unresolvable for Dependabot. Moving them must not lose the
    check, so they are read back out of the workflow.

    Matched by variable name rather than by parsing YAML, to avoid a
    dependency for four strings. A renamed variable drops the job from the
    report rather than silently passing it -- main() fails when nothing
    matches the installed Home Assistant, so a rename shows up as a failure
    in that environment.
    """
    try:
        text = workflow.read_text()
    except FileNotFoundError:
        return {}

    names = {
        "PHCC_VERSION": PHCC,
        "HASSIL_VERSION": "hassil",
        "INTENTS_VERSION": "home-assistant-intents",
    }

    found = {}
    for variable, package in names.items():
        match = re.search(rf"^\s*{variable}:\s*([^\s#]+)", text, re.MULTILINE)
        if match:
            found[package] = match.group(1)

    if PHCC not in found:
        return {}

    return {"ci.yml Test (ha-minimum)": found}


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
    pins.update(workflow_pins(workspace_root / ".github" / "workflows" / "ci.yml"))
    if not pins:
        print("❌ Found nothing pinning Home Assistant")
        return 1

    print(f"Installed Home Assistant: {ha_version}")
    for package, version in sorted(declared.items()):
        print(f"  declares {package}=={version}")
    print()

    # Only the source matching the installed Home Assistant can be checked;
    # the others describe a version not present in this environment.
    matching = [
        name
        for name, found in pins.items()
        if all(found.get(package) == version for package, version in declared.items())
    ]
    if matching:
        print(f"✅ '{matching[0]}' matches Home Assistant {ha_version}")
        return 0

    print(f"❌ Nothing matches Home Assistant {ha_version}")
    print()
    for name, found in sorted(pins.items()):
        pinned = ", ".join(f"{p}=={found.get(p, '(unpinned)')}" for p in PACKAGES)
        print(f"  {name}: {pinned}")
    print()
    print("Recommendation:")
    print("Update whichever of pyproject.toml or ci.yml names the installed")
    print("Home Assistant, so its hassil and home-assistant-intents pins")
    print("match the values above.")
    print("Home Assistant pins them exactly, and they are not interchangeable")
    print("across releases.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
