"""The release workflow's checks refuse anything but an exact, on-main, documented version.

``.github/scripts/release.py`` runs in ``release.yml`` before anything is built or published;
these tests hold it to what that workflow relies on.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import frames2py

ROOT = Path(__file__).resolve().parents[1]


def load_release() -> ModuleType:
    spec = importlib.util.spec_from_file_location("release_checks", ROOT / ".github" / "scripts" / "release.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = load_release()


def project(tmp_path: Path, version: str, source_version: str | None = None) -> Path:
    (tmp_path / "src" / "frames2py").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "frames2py"\nversion = "{version}"\n')
    (tmp_path / "src" / "frames2py" / "__init__.py").write_text(f'__version__ = "{source_version or version}"\n')
    return tmp_path


# ---------------------------------------------------------------- tag and version


@pytest.mark.parametrize("tag, prerelease", [("v1.0.0", False), ("v1.0.0rc1", True), ("v1.10.0rc12", True),
                                             ("v2.0.3", False)])
def test_a_release_tag_names_the_declared_version(tmp_path: Path, tag: str, prerelease: bool) -> None:
    assert release.check_tag(tag, project(tmp_path, tag[1:])) == (tag[1:], prerelease)


@pytest.mark.parametrize("tag", ["1.0.0rc1", "v1.0", "v1.0.0.0", "v01.0.0", "v1.0.0-rc1", "v1.0.0rc", "v1.0.0rc01",
                                 "v1.0.0a1", "v1.0.0b1", "v1.0.0.post1", "v1.0.0.dev1", "v1.0.0+local", "V1.0.0",
                                 "v1.0.0rc1 ", "release-1.0.0"])
def test_a_tag_outside_the_release_scheme_is_refused(tmp_path: Path, tag: str) -> None:
    with pytest.raises(release.ReleaseError, match="is not v"):
        release.check_tag(tag, project(tmp_path, "1.0.0rc1"))


@pytest.mark.parametrize("tag, declared", [("v1.0.0", "1.0.0rc1"), ("v1.0.0rc1", "1.0.0"), ("v1.0.0rc2", "1.0.0rc1"),
                                           ("v1.0.1", "1.0.0")])
def test_a_tag_that_differs_from_pyproject_is_refused(tmp_path: Path, tag: str, declared: str) -> None:
    with pytest.raises(release.ReleaseError, match="pyproject.toml declares"):
        release.check_tag(tag, project(tmp_path, declared))


def test_a_malformed_declared_version_is_refused(tmp_path: Path) -> None:
    with pytest.raises(release.ReleaseError, match="is not X.Y.Z"):
        release.check_tag("v1.0.0", project(tmp_path, "1.0.0.0"))


def test_a_version_attribute_that_differs_from_pyproject_is_refused(tmp_path: Path) -> None:
    with pytest.raises(release.ReleaseError, match="__version__"):
        release.check_tag("v1.0.0", project(tmp_path, "1.0.0", source_version="1.0.0rc1"))


def test_the_repository_version_is_releasable_and_is_the_installed_one() -> None:
    version, _ = release.check_tag(f"v{frames2py.__version__}")
    assert version == frames2py.__version__


# ---------------------------------------------------------------- the tagged commit is on main


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    def git(*args: str) -> str:
        done = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                               "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
                              cwd=tmp_path, capture_output=True, text=True, check=True)
        return done.stdout.strip()

    git("init", "-q", "-b", "main")
    git("commit", "-q", "--allow-empty", "-m", "first")
    first = git("rev-parse", "HEAD")
    git("commit", "-q", "--allow-empty", "-m", "second")
    main = git("rev-parse", "HEAD")
    git("switch", "-q", "-c", "side", first)
    git("commit", "-q", "--allow-empty", "-m", "off main")
    side = git("rev-parse", "HEAD")
    git("tag", "-a", "v9.9.9", "-m", "annotated, off main")
    git("tag", "-a", "v1.0.0", main, "-m", "annotated, on main")
    git("switch", "-q", "main")
    return tmp_path, {"first": first, "main": main, "side": side}


def test_a_commit_on_main_passes(repository: tuple[Path, dict[str, str]]) -> None:
    path, commits = repository
    assert release.check_on_main(commits["main"], "main", path) == commits["main"]
    assert release.check_on_main(commits["first"], "main", path) == commits["first"]


def test_an_annotated_tag_on_main_is_checked_as_its_commit(repository: tuple[Path, dict[str, str]]) -> None:
    path, commits = repository
    assert release.check_on_main("v1.0.0", "main", path) == commits["main"]


@pytest.mark.parametrize("commit", ["side", "v9.9.9"])
def test_a_commit_off_main_is_refused(repository: tuple[Path, dict[str, str]], commit: str) -> None:
    path, commits = repository
    with pytest.raises(release.ReleaseError, match="is not on main"):
        release.check_on_main(commits.get(commit, commit), "main", path)


def test_an_unknown_commit_or_missing_main_is_refused(repository: tuple[Path, dict[str, str]]) -> None:
    path, commits = repository
    with pytest.raises(release.ReleaseError, match="is not a commit"):
        release.check_on_main("0" * 40, "main", path)
    with pytest.raises(release.ReleaseError, match="fetch main first"):
        release.check_on_main(commits["main"], "origin/main", path)


# ---------------------------------------------------------------- release notes


def test_the_changelog_has_notes_for_the_current_version() -> None:
    notes = release.release_notes(frames2py.__version__)
    assert notes.strip() and "\n## " not in notes


def test_notes_are_the_one_section_headed_by_the_exact_version(tmp_path: Path) -> None:
    changelog = tmp_path / "changelog.md"
    changelog.write_text("# Changelog\n\n## 1.0.0\n\nFinal.\n\n### Details\n\nMore.\n\n"
                         "## 1.0.0rc1\n\nCandidate.\n")
    assert release.release_notes("1.0.0", changelog) == "Final.\n\n### Details\n\nMore.\n"
    assert release.release_notes("1.0.0rc1", changelog) == "Candidate.\n"
    with pytest.raises(release.ReleaseError, match="0 sections"):
        release.release_notes("1.0", changelog)


@pytest.mark.parametrize("text, message", [("## 1.0.0\n\n## 0.9.0\n\nOld.\n", "is empty"),
                                           ("## 1.0.0\n\nA.\n\n## 1.0.0\n\nB.\n", "2 sections")])
def test_an_empty_or_duplicated_section_is_refused(tmp_path: Path, text: str, message: str) -> None:
    changelog = tmp_path / "changelog.md"
    changelog.write_text(text)
    with pytest.raises(release.ReleaseError, match=message):
        release.release_notes("1.0.0", changelog)


# ---------------------------------------------------------------- the pypi environment


def environment(*rules: dict[str, Any]) -> dict[str, Any]:
    return {"name": "pypi", "protection_rules": list(rules), "can_admins_bypass": False,
            "deployment_branch_policy": {"protected_branches": False, "custom_branch_policies": True}}


def reviewers(*logins: str, kind: str = "User") -> dict[str, Any]:
    return {"type": "required_reviewers", "prevent_self_review": False,
            "reviewers": [{"type": kind, "reviewer": {"login": login}} for login in logins]}


def test_an_environment_that_requires_the_owner_passes() -> None:
    report = release.check_environment(environment(reviewers("siddiquifaras"), {"type": "branch_policy"}),
                                       "siddiquifaras")
    assert any("siddiquifaras" in line for line in report)


@pytest.mark.parametrize("env", [environment(), environment({"type": "branch_policy"}),
                                 environment({"type": "wait_timer", "wait_timer": 5}),
                                 environment(reviewers()), environment(reviewers("someone-else")),
                                 environment(reviewers("siddiquifaras", kind="Team"))],
                         ids=["no rules", "branch policy only", "wait timer only", "no reviewers", "another reviewer",
                              "a team of that name"])
def test_an_environment_without_the_owners_approval_is_refused(env: dict[str, Any]) -> None:
    with pytest.raises(release.ReleaseError, match="no required-reviewers rule"):
        release.check_environment(env, "siddiquifaras")
