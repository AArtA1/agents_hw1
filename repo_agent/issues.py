"""GitHub issues of the indexed repository (public REST API; GITHUB_TOKEN raises the rate limit).

Used by the agent (search_issues / get_issue tools: "why was this changed?", known bugs) and by
commit review (issues referenced as #N in the commit message are given to the reviewer).
"""

from __future__ import annotations

import http.client
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from . import config

API = "https://api.github.com"


class IssuesError(RuntimeError):
    pass


def owner_name(web_url: str | None) -> tuple[str, str] | None:
    m = re.match(r"https://github\.com/([\w.-]+)/([\w.-]+)$", web_url or "")
    return (m.group(1), m.group(2)) if m else None


def _get(url: str, timeout: int = 20):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "repo-agent",
               "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (fixed https API host)
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            raise IssuesError("Лимит GitHub API исчерпан (задайте GITHUB_TOKEN).") from e
        if e.code == 404:
            raise IssuesError("Не найдено на GitHub.") from e
        raise IssuesError(f"GitHub API: HTTP {e.code}") from e
    except (OSError, ValueError, http.client.HTTPException) as e:  # URLError, timeouts, resets, bad JSON
        raise IssuesError(f"GitHub API недоступен или вернул некорректный ответ: {e}") from e


def _slim(it: dict) -> dict:
    return {
        "number": it["number"],
        "title": it.get("title") or "",
        "body": (it.get("body") or "")[:4000],
        "state": it.get("state"),
        "labels": [l["name"] if isinstance(l, dict) else str(l) for l in it.get("labels") or []],
        "is_pr": "pull_request" in it,
        "created_at": (it.get("created_at") or "")[:10],
        "closed_at": (it.get("closed_at") or "")[:10] or None,
        "comments": it.get("comments", 0),
        "url": it.get("html_url"),
    }


def issues_path(slug: str) -> Path:
    return config.WORKSPACE / slug / "index" / "issues.json"


def fetch_issues(slug: str, web_url: str | None, limit: int = 200) -> list[dict]:
    """Most recently updated issues and PRs (title + body), saved next to the index."""
    on = owner_name(web_url)
    if not on:
        return []
    out: list[dict] = []
    page = 1
    while len(out) < limit:
        batch = _get(f"{API}/repos/{on[0]}/{on[1]}/issues?state=all&sort=updated&per_page=100&page={page}")
        if not batch:
            break
        if not isinstance(batch, list):
            raise IssuesError("GitHub API вернул неожиданный ответ.")
        try:
            out += [_slim(x) for x in batch]
        except (KeyError, TypeError, AttributeError) as e:
            raise IssuesError(f"Неожиданный формат issues: {e}") from e
        if len(batch) < 100:
            break
        page += 1
    out = out[:limit]
    p = issues_path(slug)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False))
    return out


def load_issues(slug: str) -> list[dict]:
    p = issues_path(slug)
    return json.loads(p.read_text()) if p.exists() else []


def get_issue(slug: str, web_url: str | None, number: int, max_comments: int = 20, timeout: float = 20) -> dict:
    """One issue/PR with its discussion (cached)."""
    cache = config.WORKSPACE / slug / "issues" / f"{number}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    on = owner_name(web_url)
    if not on:
        raise IssuesError("Issues доступны только для репозиториев GitHub.")
    try:
        it = _slim(_get(f"{API}/repos/{on[0]}/{on[1]}/issues/{number}", timeout=timeout))
        it["discussion"] = []
        if it["comments"]:
            for c in _get(f"{API}/repos/{on[0]}/{on[1]}/issues/{number}/comments?per_page={max_comments}",
                          timeout=timeout):
                it["discussion"].append({"author": (c.get("user") or {}).get("login"),
                                         "role": c.get("author_association"),
                                         "body": (c.get("body") or "")[:2000]})
    except (KeyError, TypeError, AttributeError) as e:
        raise IssuesError(f"Неожиданный формат issue: {e}") from e
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(it, ensure_ascii=False))
    return it


def create_issue(owner: str, name: str, title: str, body: str, labels: list[str], token: str,
                 timeout: float = 20) -> str:
    """POST a new issue (only called after human confirmation). Returns its URL."""
    data = json.dumps({"title": title, "body": body, "labels": labels}).encode()
    req = urllib.request.Request(f"{API}/repos/{owner}/{name}/issues", data=data, method="POST", headers={
        "Accept": "application/vnd.github+json", "User-Agent": "repo-agent", "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (fixed https API host)
            return json.loads(r.read().decode())["html_url"]
    except urllib.error.HTTPError as e:
        raise IssuesError(f"GitHub API: HTTP {e.code}") from e
    except (OSError, ValueError, KeyError, http.client.HTTPException) as e:
        raise IssuesError(f"GitHub API недоступен: {e}") from e


_REF = re.compile(r"(?<![\w/])#(\d{1,6})\b")


def referenced_numbers(text: str) -> list[int]:
    seen: list[int] = []
    for m in _REF.finditer(text or ""):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def format_issue(it: dict, max_chars: int = 6000) -> str:
    kind = "PR" if it.get("is_pr") else "Issue"
    head = f"{kind} #{it['number']} [{it.get('state')}] {it['title']}"
    if it.get("labels"):
        head += f"  labels: {', '.join(it['labels'])}"
    parts = [head, f"{it.get('url')}  создано {it.get('created_at')}", "", it.get("body") or "(без описания)"]
    for c in it.get("discussion", []):
        parts.append(f"\n— {c['author']} ({c['role']}): {c['body']}")
    return "\n".join(parts)[:max_chars]
