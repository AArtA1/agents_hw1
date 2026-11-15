"""Parse and validate `path:start-end` citations in answers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .index import RepoIndex

_EXTLESS = r"(?:Dockerfile|Makefile|Procfile|Gemfile|Pipfile|Rakefile|Jenkinsfile|Vagrantfile|Containerfile)"
# path/to/file.ext:12 | path/to/file.ext:12-40 | path/to/file.ext:L12-L40 | path/to/file.ext#L12-L40 | Dockerfile:3
_SEG = r"(?:[\w.\-@+]|\[[\w.\-]+\])+"   # dir segment: also Next.js [id], @types, c++
CITE_RE = re.compile(
    r"(?<![\w/.\-@+])(?P<path>(?:\.\./)*(?:" + _SEG + r"/)*(?:[\w.\-@+]*[\w\-]\.[A-Za-z0-9]{1,8}|" + _EXTLESS + r"))"
    r"(?![\w\-])(?:(?::|#)L?(?P<start>\d+)(?:\s*[-–]\s*L?(?P<end>\d+))?)?"
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


@dataclass
class Citation:
    raw: str
    path: str               # as written
    resolved: str | None    # repository path (None if not found)
    start: int | None
    end: int | None
    valid: bool             # file exists and line range is inside the file

    @property
    def label(self) -> str:
        p = self.resolved or self.path
        if self.start:
            return f"{p}:{self.start}" + (f"-{self.end}" if self.end and self.end != self.start else "")
        return p


def _blank_urls(text: str) -> str:
    """URLs are not citations (their path parts would be mis-parsed); keep offsets stable."""
    return _URL_RE.sub(lambda m: " " * len(m.group(0)), text)


def extract(text: str, idx: RepoIndex, require_lines: bool = False) -> list[Citation]:
    clean = _blank_urls(text)
    raw = [(m.group(0), m.group("path"), m.group("start"), m.group("end")) for m in CITE_RE.finditer(clean)]
    mentioned = [p for _, p, _, _ in raw if "/" in p]
    out: list[Citation] = []
    seen = set()
    for whole, path, s, e in raw:
        start = int(s) if s else None
        end = int(e) if e else start
        if require_lines and start is None:
            continue
        if path.replace(".", "").isdigit():  # version numbers like 1.2.3
            continue
        resolved = idx.resolve_path(path, context=mentioned)
        if resolved is None and (start is None or not _looks_like_source(path)):
            continue  # "e.g", "api.github.com:443", "v1.2.3:4" — not citations of repository files
        valid = False
        if resolved:
            _, n = idx.read_lines(resolved, 1, 1)
            valid = start is None or (1 <= start <= n and 1 <= (end or start) <= n and (end or start) >= start)
        key = (resolved or path, start, end)
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(whole, path, resolved, start, end, valid))
    return out


def _looks_like_source(path: str) -> bool:
    from . import config
    name = path.rsplit("/", 1)[-1]
    return config.language_of(name) is not None


def cited_files(cites: list[Citation]) -> list[str]:
    files: list[str] = []
    for c in cites:
        if c.resolved and c.resolved not in files:
            files.append(c.resolved)
    return files


def linkify(text: str, idx: RepoIndex) -> str:
    """Turn `path:lines` citations into GitHub blob links (markdown) when the repo is on GitHub.

    Never touches fenced code, existing markdown links, bare URLs, or text inside inline code
    (except an inline code span that is exactly one citation — that one is unwrapped and linked).
    """
    if not idx.meta.get("web_url"):
        return text

    def link(m: re.Match) -> str:
        resolved = idx.resolve_path(m.group("path"))
        if not resolved or not m.group("start"):
            return m.group(0)
        start = int(m.group("start"))
        end = int(m.group("end")) if m.group("end") else None
        return f"[{m.group(0)}]({idx.blob_url(resolved, start, end)})"

    protected = re.compile(r"((?:```|~~~).*?(?:```|~~~|\Z)|\[[^\]]*\]\([^)]*\)|https?://\S+|``.+?``|`[^`\n]+`)",
                           re.S)
    parts = protected.split(text)
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 0:
            out.append(CITE_RE.sub(link, p))
        elif p.startswith("`") and not p.startswith("``"):
            inner = p[1:-1]
            m = CITE_RE.fullmatch(inner)
            out.append(link(m) if (m and m.group("start") and idx.resolve_path(m.group("path"))) else p)
        else:
            out.append(p)
    return "".join(out)
