"""Check the built documentation site the way GitHub Pages will serve it.

    uv run --only-group docs zensical build -f docs/mkdocs.yml --strict
    uv run --only-group docs python docs/check_site.py

Fails unless:
- every page in the navigation exists in ``docs/content`` and every page there is in the
  navigation;
- served over HTTP under ``/frames2py/``, every page reachable from the home page answers,
  and every internal link, stylesheet, script and image it references answers too, with any
  ``#fragment`` present as an id on the target page;
- no page links to a ``.md`` file or to an absolute path outside ``/frames2py/``;
- every link from the site or the README to this repository's files on GitHub
  (``github.com/.../blob/main/...``, ``raw.githubusercontent.com/.../main/...``) names a file
  that exists in the checkout, and every README link into the documentation site names a
  page (and anchor) of the built site.
"""

from __future__ import annotations

import functools
import http.server
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SITE = DOCS / "site"
BASE = "/frames2py/"
PAGES_URL = "https://siddiquifaras.github.io/frames2py/"
REPO_FILE = re.compile(r"https://(?:github\.com/siddiquifaras/frames2py/(?:blob|tree)/main|"
                       r"raw\.githubusercontent\.com/siddiquifaras/frames2py/main)/([^#?\s)\"'>]+)")


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for key in ("id", "name"):
            if values.get(key):
                self.ids.add(str(values[key]))
        for key in ("href", "src"):
            if values.get(key):
                self.refs.append(str(values[key]))
        if values.get("srcset"):
            self.refs.extend(part.split()[0] for part in str(values["srcset"]).split(","))


def nav_files(nav: object) -> list[str]:
    if isinstance(nav, str):
        return [nav]
    if isinstance(nav, list):
        return [f for item in nav for f in nav_files(item)]
    if isinstance(nav, dict):
        return [f for value in nav.values() for f in nav_files(value)]
    return []


class Loader(yaml.SafeLoader):
    pass


Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)


def check_nav(errors: list[str]) -> None:
    config = yaml.load((DOCS / "mkdocs.yml").read_text(), Loader=Loader)
    content = DOCS / config["docs_dir"]
    listed = nav_files(config["nav"])
    for name in listed:
        if not (content / name).is_file():
            errors.append(f"nav names {name}, which is not in {content.relative_to(ROOT)}")
    pages = {str(p.relative_to(content)) for p in content.rglob("*.md")}
    for name in sorted(pages - set(listed)):
        errors.append(f"{name} is not in the navigation")


def serve() -> tuple[http.server.ThreadingHTTPServer, str]:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path: str) -> str:
            path = urllib.parse.urlsplit(path).path
            if not path.startswith(BASE):
                return str(SITE / "__outside_base__")
            return super().translate_path("/" + path[len(BASE):])

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(SITE)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def fetch(url: str) -> tuple[int, bytes, str]:
    try:
        with urllib.request.urlopen(url) as response:
            return response.status, response.read(), response.headers.get_content_type()
    except urllib.error.HTTPError as error:
        return error.code, b"", ""


def crawl(origin: str, errors: list[str]) -> dict[str, set[str]]:
    """Every HTML page reachable from the home page, with its ids."""
    pages: dict[str, set[str]] = {}
    fragments: list[tuple[str, str, str]] = []
    todo = [origin + BASE]
    seen = set(todo)
    assets: set[str] = set()
    while todo:
        url = todo.pop()
        status, body, kind = fetch(url)
        if status != 200:
            errors.append(f"{url[len(origin):]} answered {status}")
            continue
        if kind != "text/html":
            continue
        page = Page()
        page.feed(body.decode("utf-8"))
        pages[url] = page.ids
        for ref in page.refs:
            if ref.startswith(("mailto:", "data:", "javascript:")):
                continue
            target = urllib.parse.urljoin(url, ref)
            split = urllib.parse.urlsplit(target)
            if split.netloc and not target.startswith(origin):
                continue  # external: repository links are checked separately
            where = url[len(origin):]
            if not split.path.startswith(BASE):
                errors.append(f"{where}: {ref} resolves outside {BASE}")
                continue
            if split.path.endswith(".md"):
                errors.append(f"{where}: {ref} links to a Markdown source, not a page")
                continue
            clean = urllib.parse.urlunsplit(split._replace(fragment=""))
            if split.fragment:
                fragments.append((where, ref, clean + "#" + split.fragment))
            if clean in seen:
                continue
            seen.add(clean)
            if clean.endswith("/") or clean.endswith(".html"):
                todo.append(clean)
            else:
                assets.add(clean)
    for asset in sorted(assets):
        status, _, _ = fetch(asset)
        if status != 200:
            errors.append(f"{asset[len(origin):]} answered {status}")
    for where, ref, target in fragments:
        page_url, fragment = target.split("#", 1)
        ids = pages.get(page_url)
        if ids is not None and urllib.parse.unquote(fragment) not in ids:
            errors.append(f"{where}: {ref} names an anchor that isn't on the page")
    return pages


def check_repo_links(errors: list[str], where: str, text: str) -> None:
    for path in REPO_FILE.findall(text):
        if not (ROOT / urllib.parse.unquote(path)).exists():
            errors.append(f"{where}: links to {path}, which is not in the repository")


def check_readme(errors: list[str], origin: str, pages: dict[str, set[str]]) -> None:
    readme = (ROOT / "README.md").read_text()
    check_repo_links(errors, "README.md", readme)
    links = re.findall(re.escape(PAGES_URL) + r"[^\s)>\"']*", readme)
    if len(links) < 2:
        errors.append("README.md: fewer than two links to the documentation site")
    for link in links:
        target = origin + BASE + link[len(PAGES_URL):]
        page_url, _, fragment = target.partition("#")
        if page_url not in pages:
            errors.append(f"README.md: {link} is not a page of the built site")
        elif fragment and fragment not in pages[page_url]:
            errors.append(f"README.md: {link} names an anchor that isn't on the page")


def main() -> int:
    errors: list[str] = []
    if not (SITE / "index.html").is_file():
        print(f"ERROR: no built site in {SITE.relative_to(ROOT)}; build it first", file=sys.stderr)
        return 1
    check_nav(errors)
    server, origin = serve()
    try:
        pages = crawl(origin, errors)
        for html in sorted(SITE.rglob("*.html")):
            check_repo_links(errors, str(html.relative_to(SITE)), html.read_text())
        check_readme(errors, origin, pages)
    finally:
        server.shutdown()
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print(f"{len(pages)} pages checked under {BASE}, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
