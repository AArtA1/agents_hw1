"""Pull-request review: fetch a PR diff from GitHub, review it, and (only after human confirmation)
post the findings as line comments of one PR review.

Read: public API (GITHUB_TOKEN optional). Write: needs GITHUB_TOKEN and an explicit "y" from a human.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from . import config
from .issues import API, IssuesError, owner_name
from .review.llm_review import LLMReviewer, make_report, numbered_git_diff
from .review.static import check_text

MAX_COMMENTS = 30


def _req(url: str, accept: str = "application/vnd.github+json", data: dict | None = None, token: str | None = None):
    headers = {"Accept": accept, "User-Agent": "repo-agent", "X-GitHub-Api-Version": "2022-11-28"}
    token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(data).encode() if data is not None else None
    if body:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(req, timeout=config.TOOL_TIMEOUT) as r:  # noqa: S310 (GitHub API)
            raw = r.read().decode("utf-8", "replace")
            return raw if accept.endswith("diff") else json.loads(raw)
    except urllib.error.HTTPError as e:
        raise IssuesError(f"GitHub API: HTTP {e.code} {e.read()[:200].decode('utf-8', 'replace')}") from e
    except (OSError, ValueError) as e:
        raise IssuesError(f"GitHub API недоступен: {e}") from e


def fetch_pr(web_url: str | None, number: int) -> dict:
    on = owner_name(web_url)
    if not on:
        raise IssuesError("Ревью PR доступно только для репозиториев GitHub.")
    base = f"{API}/repos/{on[0]}/{on[1]}/pulls/{number}"
    meta = _req(base)
    diff = _req(base, accept="application/vnd.github.v3.diff")
    return {"owner": on[0], "name": on[1], "number": number, "title": meta.get("title", ""),
            "body": (meta.get("body") or "")[:3000], "head_sha": (meta.get("head") or {}).get("sha"),
            "url": meta.get("html_url"), "diff": diff}


def _static_on_added(path: str, view: str):
    """Static rules on added lines only (other lines blanked, numbering preserved)."""
    added = {int(m.group(1)): m.group(2) for m in re.finditer(r"^\+\s+(\d+) \| (.*)$", view, re.M)}
    if not added:
        return []
    n = max(added)
    text = "\n".join(added.get(i, "") for i in range(1, n + 1)) + "\n"
    return [f for f in check_text(path, text) if f.line in added and f.rule != "py-syntax"]


def review_pr(idx, llm, number: int) -> dict:
    pr = fetch_pr(idx.meta.get("web_url"), number)
    views = {p: v for p, v in numbered_git_diff(pr["diff"]).items() if config.is_code(p)}
    static = [f for p, v in views.items() for f in _static_on_added(p, v)]
    llm_findings = []
    if llm is not None and views:
        items = [{"path": p, "kind": "diff", "view": v, "hints": [f for f in static if f.path == p]}
                 for p, v in views.items()]
        llm_findings = LLMReviewer(llm).review_items(
            items, context=f"Pull request #{number}: {pr['title']}\n\n{pr['body']}")
    rep = make_report(idx, static, llm_findings, list(views), scope=f"PR #{number}")
    rep["llm_used"] = llm is not None
    rep["pr"] = {k: pr[k] for k in ("owner", "name", "number", "title", "head_sha", "url")}
    # GitHub accepts line comments only on lines that are part of the diff (new side)
    diff_lines = {p: {int(x) for x in re.findall(r"^[+ ]\s+(\d+) \|", v, re.M)} for p, v in views.items()}
    rep["comments"] = [
        {"path": f["path"], "line": f["line"], "side": "RIGHT",
         "body": f"**[{f['severity']}/{f['category']}] {f['title']}**\n\n{f['explanation']}"
                 + (f"\n\n**Исправление:** {f['suggestion']}" if f.get("suggestion") else "")
                 + "\n\n_repo-agent_"}
        for f in rep["findings"] if f["line"] in diff_lines.get(f["path"], set())
    ][:MAX_COMMENTS]
    return rep


def post_review(rep: dict, token: str | None = None) -> str:
    """Publish the review on GitHub. Called only after a human confirmed it."""
    token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise IssuesError("Для публикации нужен GITHUB_TOKEN.")
    pr = rep["pr"]
    payload = {"event": "COMMENT", "commit_id": pr["head_sha"], "comments": rep["comments"],
               "body": f"Автоматическое ревью repo-agent: {len(rep['comments'])} замечаний."}
    res = _req(f"{API}/repos/{pr['owner']}/{pr['name']}/pulls/{pr['number']}/reviews", data=payload, token=token)
    return res.get("html_url") or pr["url"]
