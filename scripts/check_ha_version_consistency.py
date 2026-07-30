#!/usr/bin/env python3
"""
Check Home Assistant version consistency between hacs.json and pyproject.toml.

This script verifies that the minimum Home Assistant version specified in hacs.json
is compatible with the homeassistant dependency version in pyproject.toml.
"""

import json
import re
import sys
import tomllib
from pathlib import Path


def parse_version(version_str: str) -> tuple[int, int, int]:
    """Parse a version string into a tuple of integers (major, minor, patch)."""
    # Remove any 'v' prefix if present
    version_str = version_str.lstrip("v")

    # Handle different version formats
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")

    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def extract_ha_version_from_hacs_json(file_path: Path) -> str | None:
    """Extract the Home Assistant version from hacs.json."""
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        return data.get("homeassistant")
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error reading hacs.json: {e}")
        return None


def extract_ha_version_from_pyproject(file_path: Path) -> str | None:
    """Extract the homeassistant dependency version from pyproject.toml."""
    try:
        with open(file_path, "rb") as f:
            data = tomllib.load(f)

        dependencies = data.get("project", {}).get("dependencies", [])

        for dep in dependencies:
            if dep.startswith("homeassistant"):
                # Extract version specifier (e.g., ">=2024.8.0" -> "2024.8.0")
                match = re.search(r"homeassistant\s*([><=!]+)\s*([0-9.]+)", dep)
                if match:
                    operator = match.group(1)
                    version = match.group(2)
                    return f"{operator}{version}"

                # Handle case where it's just "homeassistant" without version
                if dep.strip() == "homeassistant":
                    return None

        return None
    except (FileNotFoundError, Exception) as e:
        print(f"Error reading pyproject.toml: {e}")
        return None


def extract_ha_version_from_readme(file_path: Path) -> str | None:
    """Extract the stated minimum Home Assistant version from README.md."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        print(f"Error reading README.md: {e}")
        return None

    match = re.search(
        r"Home Assistant\s+(\d+\.\d+\.\d+)\s+or newer", content, re.IGNORECASE
    )
    return match.group(1) if match else None


def compare_versions(hacs_version: str, pyproject_version: str) -> bool:
    """
    Compare versions to ensure they are compatible.

    The HACS version should be >= the minimum version specified in pyproject.toml.
    """
    # Parse the pyproject version (remove operator)
    pyproject_match = re.search(r"([><=!]+)\s*([0-9.]+)", pyproject_version)
    if not pyproject_match:
        print(f"Could not parse pyproject version: {pyproject_version}")
        return False

    operator = pyproject_match.group(1)
    pyproject_ver = pyproject_match.group(2)

    try:
        hacs_parsed = parse_version(hacs_version)
        pyproject_parsed = parse_version(pyproject_ver)
    except ValueError as e:
        print(f"Version parsing error: {e}")
        return False

    # For >=, the HACS version should be >= the pyproject minimum version
    if operator == ">=":
        if hacs_parsed < pyproject_parsed:
            print(
                f"Version mismatch: HACS version {hacs_version} is less than "
                f"pyproject minimum {pyproject_ver}"
            )
            return False
    elif operator == "==":
        if hacs_parsed != pyproject_parsed:
            print(
                f"Version mismatch: HACS version {hacs_version} does not match "
                f"pyproject exact version {pyproject_ver}"
            )
            return False
    elif operator == ">":
        if hacs_parsed <= pyproject_parsed:
            print(
                f"Version mismatch: HACS version {hacs_version} is not greater than "
                f"pyproject version {pyproject_ver}"
            )
            return False
    else:
        print(f"Unsupported version operator: {operator}")
        return False

    return True


def main() -> int:
    """Main function to check HA version consistency."""
    workspace_root = Path(__file__).parent.parent
    hacs_json_path = workspace_root / "hacs.json"
    pyproject_toml_path = workspace_root / "pyproject.toml"
    readme_path = workspace_root / "README.md"

    print("Checking Home Assistant version consistency...")
    print(f"HACS config: {hacs_json_path}")
    print(f"Project config: {pyproject_toml_path}")
    print(f"README: {readme_path}")
    print()

    # Extract versions
    hacs_version = extract_ha_version_from_hacs_json(hacs_json_path)
    if not hacs_version:
        print("❌ Could not find Home Assistant version in hacs.json")
        return 1

    pyproject_version = extract_ha_version_from_pyproject(pyproject_toml_path)
    if not pyproject_version:
        print("❌ Could not find homeassistant dependency in pyproject.toml")
        return 1

    readme_version = extract_ha_version_from_readme(readme_path)
    if not readme_version:
        print("❌ Could not find a stated Home Assistant version in README.md")
        print("   Expected a line like: 'Home Assistant 2025.7.0 or newer'")
        return 1

    print(f"HACS version: {hacs_version}")
    print(f"PyProject version: {pyproject_version}")
    print(f"README version: {readme_version}")
    print()

    # Compare versions
    if not compare_versions(hacs_version, pyproject_version):
        print("❌ Home Assistant versions are inconsistent!")
        print()
        print("Recommendation:")
        print(
            "Update hacs.json to match or exceed the minimum version in pyproject.toml"
        )
        return 1

    # The README is user-facing documentation, so it must state exactly the
    # version advertised to HACS rather than merely a compatible one.
    if parse_version(readme_version) != parse_version(hacs_version):
        print(
            f"❌ README states Home Assistant {readme_version} but hacs.json "
            f"advertises {hacs_version}"
        )
        print()
        print("Recommendation:")
        print("Update the Requirements section of README.md to match hacs.json")
        return 1

    print("✅ Home Assistant versions are consistent!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
