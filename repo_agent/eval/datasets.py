"""Loaders that turn public repo-QA benchmarks into eval items {question, answer, gold_*}.

* SWE-QA (qa.load_sweqa) — 15 Python repos, gold files = paths named in the reference answer.
* StackRepoQA — Stack Overflow questions + accepted answers mapped to 134 Java GitHub repos
  (one CSV). No gold files in the data: gold = repository files named after classes that appear
  in the accepted answer's code (`gold_symbols`), resolved against the index.
* CodeRepoQA — multi-turn Q&A mined from GitHub issues of 30 repos. The data is distributed via
  Google Drive (no stable direct URL), so it is loaded from a local JSON/JSONL export of GitHub
  issue objects (fields title, body, comments_details[], repository_url, number).
"""

from __future__ import annotations

import csv
import html
import json
import re
import urllib.request
from pathlib import Path

from .. import config

STACKREPOQA_URL = "https://code-world-no-blanket.github.io/StackRepoQA/data/questions.csv"
_MAINTAINER = {"OWNER", "MEMBER", "COLLABORATOR"}


def html_to_text(s: str) -> str:
    s = re.sub(r"<pre><code>(.*?)</code></pre>", lambda m: "\n```\n" + m.group(1) + "\n```\n", s or "", flags=re.S)
    s = re.sub(r"<code>(.*?)</code>", r"`\1`", s, flags=re.S)
    s = re.sub(r"<br\s*/?>|</p>|</li>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(re.sub(r"\n{3,}", "\n\n", s)).strip()


def _class_names(answer_html: str) -> list[str]:
    """CamelCase identifiers used in the answer's code — candidate classes of the repository."""
    code = " ".join(re.findall(r"<code>(.*?)</code>", answer_html or "", flags=re.S))
    names: list[str] = []
    for n in re.findall(r"\b([A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*)\b", html.unescape(code)):
        if len(n) >= 4 and n not in names and n not in {"String", "Integer", "Object", "List", "Map", "Set", "Exception", "System",
                                        "Override", "Long", "Boolean", "Double", "Class", "Thread", "Arrays",
                                        "HashMap", "ArrayList", "Optional", "Collections", "Runtime", "Math"}:
            names.append(n)
    return names


def load_stackrepoqa(repo: str | None = None, related_only: bool = True) -> list[dict]:
    cache = config.WORKSPACE / "datasets" / "stackrepoqa" / "questions.csv"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(STACKREPOQA_URL, timeout=120) as r:  # noqa: S310 (fixed https URL)
            cache.write_bytes(r.read())
    csv.field_size_limit(10 ** 9)
    items = []
    with cache.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            row = {k.strip(): v for k, v in row.items()}
            if related_only and row.get("Question Related?", "").strip().lower() != "yes":
                continue
            if repo and row["Repo Full Name"].lower() != repo.lower():
                continue
            items.append({
                "question": f"{row['SO Title']}\n\n{html_to_text(row['SO Question Body'])[:4000]}",
                "answer": html_to_text(row["SO Accepted Answer Body"]),
                "gold_symbols": _class_names(row["SO Accepted Answer Body"]),
                "repo": row["Repo Full Name"],
                "source": row.get("SO Link"),
            })
    return items


def stackrepoqa_repos() -> dict[str, int]:
    from collections import Counter
    return dict(Counter(it["repo"] for it in load_stackrepoqa()).most_common())


def load_coderepoqa(path: str, repo: str | None = None, max_items: int | None = None) -> list[dict]:
    """GitHub-issue records -> QA items. Answer = maintainers' replies (all replies if none)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    records = json.loads(text) if text.lstrip().startswith("[") else \
        [json.loads(l) for l in text.splitlines() if l.strip()]
    items = []
    for r in records:
        m = re.search(r"/repos/([\w.-]+/[\w.-]+)", r.get("repository_url") or r.get("url") or "")
        rname = m.group(1) if m else None
        if repo and (rname or "").lower() != repo.lower():
            continue
        turns = r.get("comments_details") or []
        maint = [t for t in turns if (t.get("author_association") or "").upper() in _MAINTAINER]
        answer = "\n\n".join((t.get("body") or "") for t in (maint or turns))
        if not answer.strip():
            continue
        items.append({"question": f"{r.get('title', '')}\n\n{(r.get('body') or '')[:4000]}", "answer": answer,
                      "repo": rname, "source": r.get("html_url")})
        if max_items and len(items) >= max_items:
            break
    return items
