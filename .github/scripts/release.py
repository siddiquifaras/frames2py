"""Checks the release workflow runs before anything is built or published.

    python .github/scripts/release.py check-tag v1.0.0rc1
    python .github/scripts/release.py check-on-main "$GITHUB_SHA" origin/main
    python .github/scripts/release.py notes 1.0.0rc1 --out notes.md
    python .github/scripts/release.py check-environment --repository OWNER/REPO --reviewer OWNER

- ``check-tag``: the tag is ``v`` plus a release version (``X.Y.Z`` or ``X.Y.ZrcN``, no leading
  zeros), and that version is exactly the one in ``pyproject.toml`` and ``__version__``. In
  GitHub Actions it writes ``version`` and ``prerelease`` (``true`` for an ``rc``) to
  ``$GITHUB_OUTPUT``.
- ``check-on-main``: the tagged commit is an ancestor of (or equal to) the fetched ``main``,
  by ``git merge-base --is-ancestor``.
- ``notes``: the changelog section headed exactly ``## <version>`` (up to the next ``## ``
  heading), written to ``--out``. Fails if there is no such section, or it is empty.
- ``check-environment``: the GitHub environment exists and has a required-reviewers rule
  naming ``--reviewer``, so no deployment to it can proceed without that person's approval.
  Reads the REST API with ``$GITHUB_TOKEN`` if set; the token needs ``actions: read``.

Every failure exits non-zero with the reason on stderr.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CHANGELOG = ROOT / "docs" / "content" / "changelog.md"
NUMBER = r"(?:0|[1-9][0-9]*)"
VERSION = re.compile(rf"{NUMBER}\.{NUMBER}\.{NUMBER}(?:rc{NUMBER})?")
API = "https://api.github.com"


class ReleaseError(Exception):
    pass


def project_version(root: Path = ROOT) -> str:
    return str(tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"])


def source_version(root: Path = ROOT) -> str:
    tree = ast.parse((root / "src" / "frames2py" / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__version__" for t in node.targets):
            return str(ast.literal_eval(node.value))
    raise ReleaseError("no __version__ in src/frames2py/__init__.py")


def check_tag(tag: str, root: Path = ROOT) -> tuple[str, bool]:
    """The version a release tag names, and whether it is a pre-release."""
    if not tag.startswith("v") or not VERSION.fullmatch(tag[1:]):
        raise ReleaseError(f"tag {tag!r} is not v<X.Y.Z> or v<X.Y.Z>rc<N>")
    version = tag[1:]
    declared = project_version(root)
    if not VERSION.fullmatch(declared):
        raise ReleaseError(f"pyproject.toml version {declared!r} is not X.Y.Z or X.Y.ZrcN")
    if version != declared:
        raise ReleaseError(f"tag {tag!r} names {version}, but pyproject.toml declares {declared}")
    if source_version(root) != declared:
        raise ReleaseError(f"__version__ {source_version(root)!r} differs from pyproject.toml {declared!r}")
    return version, "rc" in version


def git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def check_on_main(commit: str, main: str, cwd: Path = ROOT) -> str:
    """The commit *commit* names, if it is on *main*."""
    peeled = git("rev-parse", "--verify", f"{commit}^{{commit}}", cwd=cwd)
    if peeled.returncode != 0:
        raise ReleaseError(f"{commit} is not a commit: {peeled.stderr.strip()}")
    sha = peeled.stdout.strip()
    if git("rev-parse", "--verify", f"{main}^{{commit}}", cwd=cwd).returncode != 0:
        raise ReleaseError(f"{main} is not a commit here; fetch main first")
    ancestor = git("merge-base", "--is-ancestor", sha, main, cwd=cwd)
    if ancestor.returncode == 1:
        raise ReleaseError(f"commit {sha} is not on {main}")
    if ancestor.returncode != 0:
        raise ReleaseError(f"git merge-base failed: {ancestor.stderr.strip()}")
    return sha


def release_notes(version: str, changelog: Path = CHANGELOG) -> str:
    """The body of the changelog section headed exactly ``## <version>``."""
    lines = changelog.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.rstrip() == f"## {version}"]
    if len(starts) != 1:
        raise ReleaseError(f"{changelog.name} has {len(starts)} sections headed '## {version}', expected 1")
    body = []
    for line in lines[starts[0] + 1:]:
        if line.startswith("## "):
            break
        body.append(line)
    text = "\n".join(body).strip()
    if not text:
        raise ReleaseError(f"the '## {version}' section of {changelog.name} is empty")
    return text + "\n"


def get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                   "X-GitHub-Api-Version": "2022-11-28"})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def check_environment(environment: dict[str, Any], reviewer: str) -> list[str]:
    """What the environment's protection rules say, if they require *reviewer*'s approval."""
    rules = environment.get("protection_rules") or []
    reviewing = [r for r in rules if r.get("type") == "required_reviewers"]
    logins = {str(entry.get("reviewer", {}).get("login", "")).lower()
              for rule in reviewing for entry in rule.get("reviewers") or [] if entry.get("type") == "User"}
    if reviewer.lower() not in logins:
        raise ReleaseError(f"environment {environment.get('name')!r} has no required-reviewers rule naming "
                           f"{reviewer!r} (reviewers: {sorted(logins) or 'none'})")
    report = [f"required reviewers: {sorted(logins)}"]
    for rule in reviewing:
        report.append(f"prevent self-review: {rule.get('prevent_self_review')}")
    report.append(f"administrators may bypass: {environment.get('can_admins_bypass')}")
    report.append(f"deployment branch policy: {environment.get('deployment_branch_policy')}")
    return report


def fetch_environment(repository: str, name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = f"{API}/repos/{repository}/environments/{name}"
    try:
        environment = get_json(base)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise ReleaseError(f"{repository} has no environment {name!r}") from error
        raise ReleaseError(f"reading environment {name!r} failed: HTTP {error.code}") from error
    policies: list[dict[str, Any]] = []
    if (environment.get("deployment_branch_policy") or {}).get("custom_branch_policies"):
        policies = get_json(base + "/deployment-branch-policies").get("branch_policies", [])
    return environment, policies


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    tag = commands.add_parser("check-tag")
    tag.add_argument("tag")
    on_main = commands.add_parser("check-on-main")
    on_main.add_argument("commit")
    on_main.add_argument("main")
    notes = commands.add_parser("notes")
    notes.add_argument("version")
    notes.add_argument("--out", type=Path, required=True)
    environment = commands.add_parser("check-environment")
    environment.add_argument("--repository", required=True)
    environment.add_argument("--reviewer", required=True)
    environment.add_argument("--environment", default="pypi")
    args = parser.parse_args()

    try:
        if args.command == "check-tag":
            version, prerelease = check_tag(args.tag)
            print(f"tag {args.tag}: version {version}, pre-release {prerelease}")
            output = os.environ.get("GITHUB_OUTPUT")
            if output:
                with open(output, "a") as file:
                    file.write(f"version={version}\nprerelease={str(prerelease).lower()}\n")
        elif args.command == "check-on-main":
            print(f"commit {check_on_main(args.commit, args.main)} is on {args.main}")
        elif args.command == "notes":
            text = release_notes(args.version)
            args.out.write_text(text)
            print(f"release notes for {args.version}: {len(text.splitlines())} lines written to {args.out}")
        else:
            env, policies = fetch_environment(args.repository, args.environment)
            for line in check_environment(env, args.reviewer):
                print(f"{args.environment}: {line}")
            for policy in policies:
                print(f"{args.environment}: deployment {policy.get('type')} pattern {policy.get('name')!r}")
    except ReleaseError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
