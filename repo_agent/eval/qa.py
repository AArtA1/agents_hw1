"""QA evaluation: "доля ответов с верным файлом".

Datasets (see eval/datasets.py): SWE-QA (gold = source files named in the reference answer),
StackRepoQA (gold = repo files named after classes used in the accepted answer), CodeRepoQA
(local export of GitHub issues), or any JSONL {"question", "answer"?, "gold_files"?}.

Metrics per answer (cited files are taken in the order the answer cites them):
* primary_hit    — the FIRST cited file is a gold file. Headline metric: citing many files can't game it.
* file_hit3      — a gold file is among the first three cited files.
* file_hit_any / main_file_hit / cite_precision — diagnostics.
* cite_valid     — share of citations whose file exists and line range lies inside the file.
* retrieval hit@1/3/5 — BM25 file ranking alone (no LLM).
* baseline_popular_hit1/3 — question-blind reference: always name the most frequent gold file(s).
* judge (optional) — LLM judge 1..5 of the answer vs the reference answer.
95% Wilson intervals are reported for the headline numbers.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath

from .. import config
from ..agent import RepoAgent
from ..citations import CITE_RE
from ..index import RepoIndex
from ..llm import LLM
from ..textutil import is_aux_path, is_test_path
from .stats import rate

SWEQA_URL = "https://huggingface.co/datasets/swe-qa/SWE-QA-Benchmark/resolve/main/{name}.jsonl"
SWEQA_REPOS = {
    "requests": "psf/requests", "flask": "pallets/flask", "pytest": "pytest-dev/pytest",
    "django": "django/django", "sphinx": "sphinx-doc/sphinx", "sympy": "sympy/sympy",
    "xarray": "pydata/xarray", "astropy": "astropy/astropy", "matplotlib": "matplotlib/matplotlib",
    "pylint": "pylint-dev/pylint", "scikit-learn": "scikit-learn/scikit-learn", "sqlfluff": "sqlfluff/sqlfluff",
    "conan": "conan-io/conan", "reflex": "reflex-dev/reflex", "streamlink": "streamlink/streamlink",
}


def sweqa_name_for(slug: str) -> str | None:
    """'psf__requests' -> 'requests'."""
    for name, gh in SWEQA_REPOS.items():
        if gh.replace("/", "__").lower() == slug.lower():
            return name
    return None


def load_sweqa(name: str) -> list[dict]:
    cache = config.WORKSPACE / "datasets" / "swe-qa" / f"{name}.jsonl"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SWEQA_URL.format(name=name), timeout=60) as r:  # noqa: S310 (fixed https URL)
            cache.write_bytes(r.read())
    return [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def gold_files(item: dict, idx: RepoIndex) -> list[str]:
    """Gold = source files the reference names.

    * explicit `gold_files` win;
    * `gold_symbols` (class names, e.g. from StackRepoQA answers) map to files named after them;
    * otherwise paths mentioned in the reference answer: only source-code files (docs, changelogs,
      configs are not "the file that answers the question"); ambiguous bare names are resolved
      by the directories of other mentioned paths; a bare name that only resolves to a test/docs
      file is treated as a convention mention (e.g. "a wsgi.py file"), not a gold file.
    """
    if item.get("gold_files"):
        return [g for g in (idx.resolve_path(p) for p in item["gold_files"]) if g]
    out: list[str] = []
    if item.get("gold_symbols"):
        for sym in item["gold_symbols"]:
            cands = [f for f in idx.files if config.is_code(f) and PurePosixPath(f).stem == sym]
            prod = [f for f in cands if not is_test_path(f) and not is_aux_path(f)]
            if len(prod) == 1 and prod[0] not in out:
                out.append(prod[0])
        return out
    text = _URLS.sub(" ", item.get("answer", ""))
    found = [(m.group("path")) for m in CITE_RE.finditer(text)]
    context = [p for p in found if "/" in p]
    for written in found:
        p = idx.resolve_path(written, context=context)
        if p is None and "/" not in written:
            # ambiguous bare name (flask: app.py in src/flask and src/flask/sansio): every production candidate
            # counts as gold — the reference does not say which one, so any is a correct "file"
            cands = [f for f in idx.files if f.endswith("/" + written) and config.is_code(f)
                     and not is_test_path(f) and not is_aux_path(f)]
            for c in cands[:3]:
                if c not in out:
                    out.append(c)
            continue
        if not p or p in out or not config.is_code(p):
            continue
        if "/" not in written and (is_test_path(p) or is_aux_path(p)):
            continue
        out.append(p)
    return out


_URLS = re.compile(r"https?://\S+")


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer", "enum": [1, 2, 3, 4, 5]}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
    "additionalProperties": False,
}
JUDGE_PROMPT = """Оцени ответ ассистента на вопрос о коде по сравнению с эталонным ответом.
5 — полностью верно и по существу; 4 — верно с мелкими упущениями; 3 — частично верно;
2 — в основном неверно; 1 — неверно/не по теме. Оценивай суть, а не стиль и длину.

Вопрос: {q}

Эталонный ответ: {ref}

Ответ ассистента: {ans}"""


def evaluate(idx: RepoIndex, items: list[dict], mode: str = "agent", llm: LLM | None = None,
             limit: int | None = None, judge: bool = False, progress=None, dataset: str = "custom") -> dict:
    """Headline metric "доля ответов с верным файлом" = primary_hit: the FIRST file the answer cites is
    a gold file. Citing many files cannot game it; file_hit@3 is the lenient variant (any of the first 3)."""
    agent = RepoAgent(idx, llm if mode == "agent" else None)
    rows = []
    t0 = time.time()
    with_gold = [(it, gold_files(it, idx)) for it in items]
    with_gold = [(it, g) for it, g in with_gold if g]
    usable = with_gold[:limit] if limit else with_gold
    for i, (it, gold) in enumerate(usable):
        if progress:
            progress(i, len(usable), it["question"])
        q = it["question"]
        ranked = [f for f, _ in idx.search_files(q, k=5)]
        # leakage: a gold file's name (stem) is written in the question itself -> retrieval is easier
        leak = any(re.search(rf"\b{re.escape(PurePosixPath(g).stem)}\b", q) for g in gold
                   if len(PurePosixPath(g).stem) > 3 and PurePosixPath(g).stem not in ("__init__", "main", "index"))
        row = {"question": q, "gold": gold, "gold_in_question": leak, "retrieval_top5": ranked,
               "retrieval_hit1": bool(ranked[:1]) and ranked[0] in gold,
               "retrieval_hit3": any(f in gold for f in ranked[:3]),
               "retrieval_hit5": any(f in gold for f in ranked)}
        agent.reset()
        try:
            ans = agent.ask(q)
            cited = ans.files
            row.update({
                "answer": ans.text, "cited_files": cited, "files_read": ans.files_read,
                "n_steps": len(ans.steps), "n_cited": len(cited),
                "primary_hit": bool(cited) and cited[0] in gold,
                "file_hit3": any(f in gold for f in cited[:3]),
                "file_hit_any": any(f in gold for f in cited),
                "main_file_hit": gold[0] in cited[:3],
                "cite_precision": (sum(f in gold for f in cited) / len(cited)) if cited else 0.0,
                "cite_valid": (sum(c.valid for c in ans.citations) / len(ans.citations)) if ans.citations else 0.0,
            })
            if judge and llm is not None and it.get("answer"):
                j = llm.json(JUDGE_PROMPT.format(q=q, ref=it["answer"][:6000], ans=ans.text[:6000]),
                             JUDGE_SCHEMA, effort="low", max_tokens=2000)
                row["judge"] = j["score"]
                row["judge_reason"] = j["reason"]
        except Exception as e:  # keep evaluating; the error is recorded
            row.update({"error": str(e), "primary_hit": False, "file_hit3": False, "file_hit_any": False,
                        "main_file_hit": False, "cite_precision": 0.0, "cite_valid": 0.0, "cited_files": [],
                        "n_steps": 0, "n_cited": 0})
        rows.append(row)

    def mean(key):
        vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    # question-blind reference: always cite the most frequent gold file(s) of this dataset slice
    freq = Counter(f for r in rows for f in r["gold"])
    top = [f for f, _ in freq.most_common(3)]
    base1 = rate([bool(top) and top[0] in r["gold"] for r in rows])
    base3 = rate([any(f in r["gold"] for f in top) for r in rows])
    prim = rate([r["primary_hit"] for r in rows])
    clean_rows = [r for r in rows if not r["gold_in_question"]]
    prim_clean = rate([r["primary_hit"] for r in clean_rows])
    hit3 = rate([r["file_hit3"] for r in rows])
    return {
        "slug": idx.slug, "dataset": dataset, "mode": mode,
        "model": llm.model if (llm and mode == "agent") else None,
        "n_questions": len(rows), "n_dataset": len(items), "n_without_gold": len(items) - len(with_gold),
        "metrics": {
            "primary_hit": prim["rate"], "primary_hit_ci95": prim["ci95"],
            "file_hit3": hit3["rate"], "file_hit3_ci95": hit3["ci95"],
            # same metric on questions that do not name a gold file (no leakage)
            "primary_hit_no_leak": prim_clean["rate"], "n_no_leak": prim_clean["n"],
            "file_hit_any": mean("file_hit_any"), "main_file_hit": mean("main_file_hit"),
            "cite_precision": mean("cite_precision"), "cite_valid": mean("cite_valid"),
            "avg_cited_files": mean("n_cited"),
            "retrieval_hit1": mean("retrieval_hit1"), "retrieval_hit3": mean("retrieval_hit3"),
            "retrieval_hit5": mean("retrieval_hit5"),
            "baseline_popular_hit1": base1["rate"], "baseline_popular_hit3": base3["rate"],
            "avg_steps": mean("n_steps"), "judge": mean("judge"),
            "errors": sum(1 for r in rows if "error" in r),
        },
        "baseline_popular_files": top,
        "seconds": round(time.time() - t0, 1),
        "usage": llm.usage if llm else None,
        "rows": rows,
    }


def save(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = re.sub(r"[^\w-]+", "-", str(result.get("dataset", "custom")))[:40]
    p = out_dir / f"qa_{result['mode']}_{ds}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return p
