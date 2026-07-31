#!/usr/bin/env python3
"""
Python Version Consistency Checker

This script checks that Python version configurations are consistent across
all project files including pyproject.toml, .python-version, .devcontainer.json,
GitHub Actions workflows, and tool configurations.
"""

import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any


class VersionChecker:
    """Checks Python version consistency across project files."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(f"❌ ERROR: {message}")

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(f"⚠️  WARNING: {message}")

    def parse_version_requirement(self, version_spec: str) -> tuple[str, str]:
        """Parse a version requirement like '>=3.12' into operator and version."""
        match = re.match(r"([><=!]+)(.+)", version_spec.strip())
        if match:
            return match.group(1), match.group(2)
        return "==", version_spec.strip()

    def extract_major_minor(self, version: str) -> str:
        """Extract major.minor version from a version string."""
        # Handle versions like "3.12.1" -> "3.12"
        parts = version.split(".")
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
        return version

    def parse_version(self, version: str) -> tuple[int, ...]:
        """Parse a version string into a comparable tuple of integers."""
        return tuple(int(part) for part in version.split(".") if part.isdigit())

    def check_pyproject_toml(self) -> dict[str, Any]:
        """Check pyproject.toml for Python version configurations."""
        pyproject_path = self.project_root / "pyproject.toml"
        if not pyproject_path.exists():
            self.add_error("pyproject.toml not found")
            return {}

        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            self.add_error(f"Failed to parse pyproject.toml: {e}")
            return {}

        versions = {}

        # Check project.requires-python
        if "project" in data and "requires-python" in data["project"]:
            requires_python = data["project"]["requires-python"]
            versions["requires-python"] = requires_python

        # Check tool.mypy.python_version
        if (
            "tool" in data
            and "mypy" in data["tool"]
            and "python_version" in data["tool"]["mypy"]
        ):
            mypy_version = data["tool"]["mypy"]["python_version"]
            versions["mypy"] = mypy_version

        # Check tool.ruff.target-version
        if (
            "tool" in data
            and "ruff" in data["tool"]
            and "target-version" in data["tool"]["ruff"]
        ):
            ruff_version = data["tool"]["ruff"]["target-version"]
            versions["ruff"] = ruff_version

        return versions

    def check_python_version_file(self) -> str | None:
        """Check .python-version file."""
        python_version_path = self.project_root / ".python-version"
        if not python_version_path.exists():
            self.add_warning(".python-version file not found")
            return None

        try:
            with open(python_version_path) as f:
                version = f.read().strip()
                return version
        except Exception as e:
            self.add_error(f"Failed to read .python-version: {e}")
            return None

    def check_devcontainer_json(self) -> str | None:
        """Check .devcontainer.json for Python version."""
        devcontainer_path = self.project_root / ".devcontainer.json"
        if not devcontainer_path.exists():
            self.add_warning(".devcontainer.json not found")
            return None

        try:
            with open(devcontainer_path) as f:
                # Remove comments from JSON
                content = f.read()
                # Simple comment removal (not perfect but works for most cases)
                content = re.sub(r"//.*", "", content)
                data = json.loads(content)

            # Check image field for Python version
            if "image" in data:
                image = data["image"]
                # Look for python:3.x pattern
                match = re.search(r"python:(\d+\.\d+)", image)
                if match:
                    return match.group(1)

        except Exception as e:
            self.add_error(f"Failed to parse .devcontainer.json: {e}")

        return None

    def check_github_workflows(self) -> list[str]:
        """Check GitHub Actions workflows for Python versions."""
        workflows_dir = self.project_root / ".github" / "workflows"
        if not workflows_dir.exists():
            self.add_warning(".github/workflows directory not found")
            return []

        versions = []

        for workflow_file in workflows_dir.glob("*.yml"):
            try:
                with open(workflow_file) as f:
                    content = f.read()

                # Look for python-version patterns
                patterns = [
                    r'python-version:\s*"([^"]+)"',
                    r"python-version:\s*'([^']+)'",
                    r'python-version:\s*\["([^"]+)"\]',
                    r"python-version:\s*\['([^']+)'\]",
                ]

                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        # Handle matrix versions like "3.12" or ["3.11", "3.12"]
                        if "," in match:
                            # Multiple versions in matrix
                            for v in match.split(","):
                                v = v.strip().strip("\"'")
                                if v:
                                    versions.append(v)
                        else:
                            versions.append(match.strip().strip("\"'"))

            except Exception as e:
                self.add_error(f"Failed to parse {workflow_file}: {e}")

        return versions

    def check_consistency(self) -> bool:
        """Perform all consistency checks."""
        print("🔍 Checking Python version consistency across project files...\n")

        # Gather all version information
        pyproject_versions = self.check_pyproject_toml()
        python_version_file = self.check_python_version_file()
        devcontainer_version = self.check_devcontainer_json()
        github_versions = self.check_github_workflows()

        # Display found versions
        print("📋 Found Python versions:")
        if pyproject_versions:
            for key, value in pyproject_versions.items():
                print(f"  pyproject.toml [{key}]: {value}")
        if python_version_file:
            print(f"  .python-version: {python_version_file}")
        if devcontainer_version:
            print(f"  .devcontainer.json: {devcontainer_version}")
        if github_versions:
            print(f"  GitHub Actions: {', '.join(set(github_versions))}")
        print()

        # Determine the canonical version from requires-python
        canonical_version = None
        if "requires-python" in pyproject_versions:
            requires_python = pyproject_versions["requires-python"]
            _, version = self.parse_version_requirement(requires_python)
            canonical_version = self.extract_major_minor(version)
            print(f"🎯 Canonical version (from requires-python): {canonical_version}")
        else:
            self.add_error("No requires-python found in pyproject.toml")
            print()
            return False

        print()

        # Check consistency
        if canonical_version:
            # Check mypy version
            if "mypy" in pyproject_versions:
                mypy_version = self.extract_major_minor(pyproject_versions["mypy"])
                if mypy_version != canonical_version:
                    self.add_error(
                        f"mypy python_version ({mypy_version}) doesn't match "
                        f"requires-python ({canonical_version})"
                    )

            # Check ruff target-version
            if "ruff" in pyproject_versions:
                ruff_version = pyproject_versions["ruff"]
                # Ruff uses format like "py312"
                if ruff_version.startswith("py"):
                    ruff_numeric = f"{ruff_version[2]}.{ruff_version[3:]}"
                    if ruff_numeric != canonical_version:
                        self.add_error(
                            f"ruff target-version ({ruff_version} = {ruff_numeric}) "
                            f"doesn't match requires-python ({canonical_version})"
                        )

            # Check .python-version
            if python_version_file:
                file_version = self.extract_major_minor(python_version_file)
                if file_version != canonical_version:
                    self.add_error(
                        f".python-version ({file_version}) doesn't match "
                        f"requires-python ({canonical_version})"
                    )

            # Check devcontainer
            if devcontainer_version:
                container_version = self.extract_major_minor(devcontainer_version)
                if container_version != canonical_version:
                    self.add_error(
                        f".devcontainer.json Python version ({container_version}) "
                        f"doesn't match requires-python ({canonical_version})"
                    )

            # Check GitHub Actions.
            #
            # CI must exercise the declared minimum, but is allowed to test
            # newer versions as well: the test-latest job deliberately runs on
            # a newer Python, because recent Home Assistant releases require
            # one. Anything *older* than requires-python is still an error,
            # since the project could not run there at all.
            gh_versions_clean = {
                self.extract_major_minor(v) for v in set(github_versions)
            }

            if canonical_version not in gh_versions_clean:
                self.add_error(
                    f"No GitHub Actions job runs on the declared Python "
                    f"version ({canonical_version}); found "
                    f"{', '.join(sorted(gh_versions_clean))}"
                )

            for gh_version_clean in sorted(gh_versions_clean):
                if self.parse_version(gh_version_clean) < self.parse_version(
                    canonical_version
                ):
                    self.add_error(
                        f"GitHub Actions Python version ({gh_version_clean}) "
                        f"is older than requires-python ({canonical_version})"
                    )

        # Print results
        if self.warnings:
            for warning in self.warnings:
                print(warning)
            print()

        if self.errors:
            for error in self.errors:
                print(error)
            print()
            print("❌ Version consistency check FAILED")
            return False
        else:
            print("✅ All Python versions are consistent!")
            return True


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    checker = VersionChecker(project_root)

    success = checker.check_consistency()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
