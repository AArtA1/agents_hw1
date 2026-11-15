"""Fetching a repository and reading its files / git history."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .textutil import is_translation_path


@dataclass
class Commit:
    sha: str
    author: str
    date: str
    subject: str
    body: str = ""
    files: list[str] = field(default_factory=list)


def parse_repo_spec(spec: str) -> tuple[str, str]:
    """Return (clone_url, slug) for 'owner/name', a GitHub URL, or a local path."""
    p = Path(spec).expanduser()
    if p.exists():
        return str(p.resolve()), p.resolve().name
    m = re.match(r"^(?:https?://github\.com/|git@github\.com:)?([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", spec)
    if not m:
        raise ValueError(f"Не понимаю репозиторий: {spec!r}. Ожидается owner/name, URL GitHub или локальный путь.")
    owner, name = m.group(1), m.group(2)
    return f"https://github.com/{owner}/{name}.git", f"{owner}__{name}"


class GitTimeout(RuntimeError):
    pass


def git_run(repo: Path, *args: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-c", "core.quotepath=off", "-C", str(repo), *args],
                              capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise GitTimeout(f"git {args[0] if args else ''} не уложился в {timeout:.0f} с") from e


def git(repo: Path, *args: str, check: bool = True, timeout: float | None = None) -> str:
    res = git_run(repo, *args, timeout=timeout)
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout


def fetch(spec: str, depth: int | None = None) -> tuple[Path, str]:
    """Clone (or reuse) a repository. Returns (path to checkout, slug)."""
    url, slug = parse_repo_spec(spec)
    if Path(url).exists():  # local repository: use in place
        return Path(url), slug
    dest = config.WORKSPACE / slug / "repo"
    if (dest / ".git").exists():
        return dest, slug
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--quiet"]
    if depth:
        cmd += ["--depth", str(depth)]
    cmd += [url, str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"git clone failed: {res.stderr.strip()}")
    return dest, slug


def remote_web_url(repo: Path) -> str | None:
    """https://github.com/owner/name for building blob links (None for non-GitHub)."""
    url = git(repo, "config", "--get", "remote.origin.url", check=False).strip()
    m = re.match(r"^(?:https?://github\.com/|git@github\.com:)([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", url)
    return f"https://github.com/{m.group(1)}/{m.group(2)}" if m else None


def head_sha(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD", check=False).strip() or "HEAD"


def list_files(repo: Path) -> list[str]:
    """Repository-relative paths of text files worth indexing."""
    # -z: raw (unquoted) names, so non-ASCII paths survive
    tracked = [p for p in git(repo, "ls-files", "-z", check=False).split("\0") if p]
    if not tracked:  # not a git repo: walk the tree
        tracked = [str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file()]
    out = []
    for rel in tracked:
        parts = Path(rel).parts
        if any(part in config.SKIP_DIRS for part in parts[:-1]) or is_translation_path(rel):
            continue
        if config.language_of(rel) is None:
            continue
        full = repo / rel
        try:
            if not full.is_file() or full.stat().st_size > config.MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        if rel.endswith((".min.js", "package-lock.json", "yarn.lock", "poetry.lock")):
            continue
        out.append(rel)
    return sorted(out)


def read_text(repo: Path, rel: str) -> str | None:
    try:
        data = (repo / rel).read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:4096]:
        return None
    return data.decode("utf-8", errors="replace")


def commits(repo: Path, limit: int = config.MAX_COMMITS, path: str | None = None,
            timeout: float | None = None) -> list[Commit]:
    sep, end = "\x1f", "\x1e"
    args = ["log", f"-n{limit}", f"--pretty=format:{end}%H{sep}%an{sep}%ad{sep}%s{sep}%b{sep}",
            "--date=short", "--name-only"]
    if path:
        args += ["--follow", "--", path]  # keep history across renames
    raw = git(repo, *args, check=False, timeout=timeout)
    result = []
    for rec in raw.split(end):
        if not rec.strip():
            continue
        parts = rec.split(sep)
        if len(parts) < 6:
            continue
        files = [f for f in parts[5].strip().splitlines() if f.strip()]
        result.append(Commit(parts[0], parts[1], parts[2], parts[3], parts[4].strip(), files))
    return result


def show_commit(repo: Path, sha: str, max_chars: int = 60_000, timeout: float | None = None) -> str:
    out = git(repo, "show", "--stat", "--patch", "--format=%H %an %ad%n%s%n%n%b", "--date=short",
              sha, check=False, timeout=timeout)
    return out[:max_chars]


def blame(repo: Path, rel: str, start: int, end: int, timeout: float | None = None) -> str:
    """Commit, author, date per line (author/date are what 'who changed this' questions need)."""
    if start < 1 or end < start:
        return f"Ошибка: некорректный диапазон строк {start}-{end}."
    res = git_run(repo, "blame", "--date=short", "-L", f"{start},{end}", "--", rel, timeout=timeout)
    if res.returncode != 0:
        return f"Ошибка git blame: {res.stderr.strip()}"
    return res.stdout or "(пустой результат blame)"
