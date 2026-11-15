"""Bug-detection evaluation: "доля найденных внесённых багов".

Bugs:
* synthetic — one single-line mutation per trial in a Python function (flipped comparison, off-by-one,
  inverted None check, `and`/`or`, disabled TLS verification, `eval`, ...); the mutation must change the
  AST and never touches strings/comments/asserts; at most one trial per function;
* history   — real bugs: the parent version of code lines changed by small, specific "fix ..." commits
  (lint/doc/typing/test commits and module-level edits are filtered out).

The reviewer sees the whole enclosing function with line numbers and no hint where the bug is.
A trial is *detected* if a bug/security finding is within ±tol lines of a bug line; a finding that spans
more than MAX_SPAN lines counts only by its first line (no "flag the whole function" wins).
Every trial has a *control*: the same function without the bug (original code / the fixed version);
a finding at the same place there is a false alarm. Reported: detection rate, false-alarm rate,
discrimination = detection − false alarms, precision of findings, chance level of random flagging,
and 95% Wilson intervals.
"""

from __future__ import annotations

import ast
import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import config, repo as repo_mod
from ..llm import LLM
from ..review.llm_review import LLMReviewer, numbered, numbered_diff
from ..chunking import safe_parse
from ..review.static import Finding, check_text
from ..textutil import is_aux_path, is_test_path, line_count, mask_python, split_lines
from .stats import rate, wilson

# (op id, kind, regex applied to code-only text of the line, replacement, description)
MUTATIONS = [
    ("flip_eq", "bug", r"(?<=[^=!<>])==(?==?[^=])", "!=", "== заменено на !="),
    ("flip_neq", "bug", r"!=", "==", "!= заменено на =="),
    ("lt_to_le", "bug", r"(?<![<>=])<(?![<=])", "<=", "< заменено на <= (off-by-one)"),
    ("le_to_lt", "bug", r"<=", "<", "<= заменено на < (off-by-one)"),
    ("gt_to_ge", "bug", r"(?<![<>=-])>(?![>=])", ">=", "> заменено на >= (off-by-one)"),
    ("ge_to_gt", "bug", r">=", ">", ">= заменено на > (off-by-one)"),
    ("drop_not", "bug", r"\b(if|elif|while|and|or|return)\s+not\s+", r"\1 ", "удалено отрицание not"),
    ("is_none_flip", "bug", r"\bis None\b", "is not None", "is None заменено на is not None"),
    ("is_not_none_flip", "bug", r"\bis not None\b", "is None", "is not None заменено на is None"),
    ("and_to_or", "bug", r"(?<=\s)and(?=\s)", "or", "and заменено на or"),
    ("or_to_and", "bug", r"(?<=\s)or(?=\s)", "and", "or заменено на and"),
    ("return_true_false", "bug", r"\breturn True\b", "return False", "return True заменено на return False"),
    ("return_false_true", "bug", r"\breturn False\b", "return True", "return False заменено на return True"),
    ("plus_one_drop", "bug", r"\s*\+\s*1\b(?!\d|\.)", "", "убрано + 1 (off-by-one)"),
    ("minus_one_drop", "bug", r"\s*-\s*1\b(?!\d|\.)(?=\s*[\]\):,])", "", "убрано - 1 (off-by-one)"),
    ("index_shift", "bug", r"\[0\]", "[1]", "индекс [0] заменён на [1]"),
    ("last_index_shift", "bug", r"\[-1\]", "[-2]", "индекс [-1] заменён на [-2]"),
    ("tls_verify_off", "security", r"\b(verify\s*=\s*)True\b", r"\1False", "verify=True заменено на verify=False"),
    ("weak_hash", "security", r"\bsha256\b", "md5", "sha256 заменено на md5"),
    ("json_to_eval", "security", r"\b(?:json|complexjson)\.loads\(", "eval(", "json.loads заменено на eval"),
    ("yaml_unsafe", "security", r"\byaml\.safe_load\(", "yaml.load(", "yaml.safe_load заменено на yaml.load"),
    ("shell_true", "security", r"\bshell\s*=\s*False\b", "shell=True", "shell=False заменено на shell=True"),
    ("basename_drop", "security", r"\bos\.path\.basename\(([^()]*)\)", r"\1", "убран os.path.basename (path traversal)"),
]
# only mutate comparisons/boolean ops in control-flow-ish lines
_CONTROL_ONLY = {"flip_eq", "flip_neq", "lt_to_le", "le_to_lt", "gt_to_ge", "ge_to_gt", "and_to_or", "or_to_and",
                 "is_none_flip", "is_not_none_flip"}
_CONTROL_RE = re.compile(r"^\s*(if|elif|while|return)\b|\bif\b.*\belse\b")


@dataclass
class Trial:
    id: str
    source: str              # synthetic | history
    path: str
    lines: list[int]         # bug location(s) in the shown (buggy) version
    op: str
    kind: str                # bug | security
    description: str
    view_start: int
    view_end: int
    buggy_text: str = field(repr=False, default="")
    clean_text: str = field(repr=False, default="")
    clean_lines: list[int] = field(default_factory=list)   # corresponding location in the clean version
    clean_view_start: int = 0
    clean_view_end: int = 0
    commit: str | None = None


# ------------------------------------------------------------ helpers

def _function_span(tree: ast.AST, line: int) -> tuple[int, int] | None:
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s, e = node.lineno, getattr(node, "end_lineno", node.lineno)
            if node.decorator_list:
                s = min(d.lineno for d in node.decorator_list)
            if s <= line <= e and (best is None or e - s < best[1] - best[0]):
                best = (s, e)
    return best


def _enclosing_span(tree: ast.AST, line: int, n_lines: int, max_span: int = 120) -> tuple[int, int]:
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s, e = node.lineno, getattr(node, "end_lineno", node.lineno)
            if node.decorator_list:
                s = min(d.lineno for d in node.decorator_list)
            if s <= line <= e and (best is None or e - s < best[1] - best[0]):
                best = (s, e)
    if best is None or best[1] - best[0] > max_span:
        return max(1, line - 30), min(n_lines, line + 30)
    return best


def find_sites(path: str, text: str) -> list[tuple[str, int, str]]:
    """All (op, line, mutated_text) candidates in a file. Lines are '\n'-separated (git numbering)."""
    code = mask_python(text)
    if code is None:
        return []
    orig_tree = safe_parse(text)
    if orig_tree is None:
        return []
    orig_dump = ast.dump(orig_tree)
    lines = text.split("\n")
    sites = []
    for ln, masked in enumerate(code, 1):
        stripped = masked.lstrip()
        if not stripped or stripped.startswith(("def ", "class ", "@", "import ", "from ", "assert ")):
            continue
        for op, _kind, rx, repl, _desc in MUTATIONS:
            if op in _CONTROL_ONLY and not _CONTROL_RE.search(masked):
                continue
            m = re.search(rx, masked)
            if not m or (op == "weak_hash" and "usedforsecurity" in masked):
                continue
            orig_line = lines[ln - 1]
            new_seg = m.expand(repl) if "\\" in repl else repl
            new_line = orig_line[:m.start()] + new_seg + orig_line[m.end():]
            if new_line == orig_line:
                continue
            new_text = "\n".join(lines[:ln - 1] + [new_line] + lines[ln:])
            new_tree = safe_parse(new_text)
            if new_tree is None or ast.dump(new_tree) == orig_dump:
                continue
            sites.append((op, ln, new_text))
    return sites


def synthetic_trials(idx, n: int = 20, seed: int = 13, path_prefix: str | None = None) -> list[Trial]:
    rng = random.Random(seed)
    by_op: dict[str, list[tuple[str, int, str, str]]] = {}
    for f in idx.files:
        if not f.endswith(".py") or is_test_path(f) or is_aux_path(f) or (path_prefix and not f.startswith(path_prefix)):
            continue
        if f.split("/")[0] in ("scripts", "tools"):
            continue
        text = repo_mod.read_text(idx.root, f) or ""
        for op, ln, new_text in find_sites(f, text):
            by_op.setdefault(op, []).append((f, ln, new_text, text))
    for v in by_op.values():
        rng.shuffle(v)
    ops = sorted(by_op)
    rng.shuffle(ops)
    meta = {m[0]: m for m in MUTATIONS}
    trials: list[Trial] = []
    used_funcs: set[tuple[str, int]] = set()  # at most one trial per function -> independent trials
    while len(trials) < n and any(by_op.values()):
        for op in ops:  # round-robin across mutation types for diversity
            if len(trials) >= n:
                break
            while by_op[op]:
                f, ln, new_text, text = by_op[op].pop()
                tree = safe_parse(new_text)
                s, e = _enclosing_span(tree, ln, line_count(new_text))
                if (f, s) in used_funcs:
                    continue
                used_funcs.add((f, s))
                trials.append(Trial(f"syn-{len(trials):03d}", "synthetic", f, [ln], op, meta[op][1], meta[op][4],
                                    s, e, new_text, text, [ln], s, e))
                break
    return trials


_FIX_RE = re.compile(r"\b(fix(e[sd])?|bug|regression|crash|incorrect|wrong|broken|handle)\b", re.I)
_NOT_BUG_RE = re.compile(r"typo|\bdocs?\b|documentation|docstring|lint|style|format|spelling|grammar|comment|"
                         r"readme|link|cosmetic|tests?\b|\bci\b|changelog|version bump|message|flake8|pep ?8|"
                         r"indent|whitespace|mypy|typing|type hints?|annotation|pylint|ruff|isort|\bblack\b|"
                         r"warning|deprecat|python ?[23]\b|py[23]\b|python ?2\.\d|compat|based on review|"
                         r"review comments|address(?:ed)? review", re.I)


def _hunks(diff: str) -> list[tuple[int, int, int, int]]:
    return [(int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1))
            for m in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", diff, re.M)]


def _changed(hunks, side: int) -> list[int]:
    out: list[int] = []
    for h in hunks:
        a, b = (h[0], h[1]) if side == 0 else (h[2], h[3])
        out += list(range(a, a + b)) if b else [max(1, a + (0 if side == 0 else 1))]
    return out


def history_trials(idx, n: int = 15, max_changed: int = 8, max_coverage: float = 0.4, tol: int = 2) -> list[Trial]:
    """Real bugs: the parent version of code changed by small 'fix' commits (needs full history).

    Bug lines are the *code* lines the fix changed (docstring/comment lines are dropped); trials whose
    ±tol band would cover more than `max_coverage` of the shown function are skipped (too easy to hit).
    The control is the fixed version of the same function.
    """
    root = idx.root
    trials: list[Trial] = []
    used_funcs: set[tuple[str, int]] = set()
    for c in repo_mod.commits(root, limit=3000):
        if len(trials) >= n:
            break
        subj = c.subject
        if not _FIX_RE.search(subj) or _NOT_BUG_RE.search(subj) or "merge" in subj.lower() or \
                subj.lower().startswith("revert") or len(re.findall(r"[A-Za-z]{2,}", subj)) < 3:
            continue  # vague subjects like "fix" / "fix models.py" are not reliable bug labels
        code_files = [f for f in c.files if f.endswith(".py") and not is_test_path(f) and not is_aux_path(f)
                      and f not in ("setup.py", "noxfile.py") and not f.startswith(("scripts/", "tools/"))]
        if len(code_files) != 1:
            continue
        path = code_files[0]
        hunks = _hunks(repo_mod.git(root, "diff", "-U0", f"{c.sha}^", c.sha, "--", path, check=False))
        if not hunks:
            continue
        old_text = repo_mod.git(root, "show", f"{c.sha}^:{path}", check=False)
        new_text = repo_mod.git(root, "show", f"{c.sha}:{path}", check=False)
        if not old_text or not new_text:
            continue
        old_tree, new_tree = safe_parse(old_text), safe_parse(new_text)
        if old_tree is None or new_tree is None:
            continue
        old_code, new_code = mask_python(old_text) or [], mask_python(new_text) or []
        is_code = lambda code, l: 0 < l <= len(code) and code[l - 1].strip() != ""
        # only code lines inside functions: an added import at module level is context, not the bug
        old_lines = [l for l in _changed(hunks, 0) if is_code(old_code, l) and _function_span(old_tree, l)]
        new_lines = [l for l in _changed(hunks, 1) if is_code(new_code, l) and _function_span(new_tree, l)]
        if not old_lines or len(old_lines) > max_changed or max(old_lines) - min(old_lines) > 30:
            continue
        span = _function_span(old_tree, old_lines[0])  # module-level edits (imports etc.) are not bug trials
        if span is None or span[1] - span[0] > 120 or not all(span[0] <= l <= span[1] for l in old_lines):
            continue
        s, e = span
        if (path, s) in used_funcs:  # one trial per function keeps trials independent
            continue
        band = {x for l in old_lines for x in range(l - tol, l + tol + 1)}
        view_code = [l for l in range(s, e + 1) if is_code(old_code, l)]
        if not view_code or len(band & set(view_code)) / len(view_code) > max_coverage:
            continue
        # control = the same function in the fixed version (matched by name, not by line numbers)
        fname = _function_name(old_tree, s)
        cspan = _named_function_span(new_tree, fname, near=s) if fname else None
        if cspan is None:
            continue
        cs, ce = cspan
        clean_lines = [l for l in new_lines if cs <= l <= ce]
        if not clean_lines:  # pure deletion fix: the same offset inside the fixed function
            clean_lines = [min(ce, cs + (old_lines[0] - s))]
        used_funcs.add((path, s))
        trials.append(Trial(f"hist-{len(trials):03d}", "history", path, old_lines, "history-fix", "bug",
                            subj, s, e, old_text, new_text, clean_lines, cs, ce, c.sha))
    return trials


def _function_name(tree: ast.AST, start: int) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s = min([d.lineno for d in node.decorator_list] + [node.lineno])
            if s == start:
                return node.name
    return None


def _named_function_span(tree: ast.AST, name: str, near: int) -> tuple[int, int] | None:
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            s = min([d.lineno for d in node.decorator_list] + [node.lineno])
            e = getattr(node, "end_lineno", node.lineno)
            if best is None or abs(s - near) < abs(best[0] - near):
                best = (s, e)
    return best


# ------------------------------------------------------------ scoring

MAX_SPAN = 3  # a finding spanning more lines than this only counts by its first line


def _localizes(f: Finding, path: str, lines: list[int], tol: int) -> bool:
    if f.path != path or f.category not in ("bug", "security"):
        return False
    lo = f.line
    hi = f.end_line if (f.end_line and 0 <= f.end_line - f.line <= MAX_SPAN) else f.line
    return any(lo - tol <= l <= hi + tol for l in lines)


def _detected(findings: list[Finding], path: str, lines: list[int], tol: int) -> bool:
    return any(_localizes(f, path, lines, tol) for f in findings)


def _chance(text: str, s: int, e: int, lines: list[int], tol: int) -> float:
    """P(one uniformly random flagged code line in the view lands within ±tol of the bug)."""
    code = mask_python(text) or split_lines(text)
    view = [l for l in range(s, e + 1) if 0 < l <= len(code) and code[l - 1].strip()]
    if not view:
        return 0.0
    band = {x for l in lines for x in range(l - tol, l + tol + 1)}
    return len(band & set(view)) / len(view)


# tokens that the mutation operators *produce*; a reviewer flagging them knows nothing about bugs
_ARTIFACT = re.compile(r"<=|>=|!=|==|\bis not None\b|\bis None\b|\[1\]|\[-2\]|\bmd5\b|\beval\(|verify\s*=\s*False|"
                       r"yaml\.load\(|shell\s*=\s*True|\bor\b|\band\b|return (?:True|False)\b")


def artifact_review(path: str, text: str, s: int, e: int) -> list[Finding]:
    """Bug-blind reference reviewer: flags every line containing a mutation-output token."""
    code = mask_python(text) or split_lines(text)
    return [Finding(path, l, "medium", "bug", "artifact", "") for l in range(s, e + 1)
            if 0 < l <= len(code) and _ARTIFACT.search(code[l - 1])]


def _bootstrap_diff_ci(a: list[bool], b: list[bool], iters: int = 2000, seed: int = 0) -> list[float] | None:
    """95% bootstrap CI of mean(a) - mean(b) over paired trials."""
    pairs = list(zip(a, b))
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    diffs = sorted(sum(x - y for x, y in (rng.choice(pairs) for _ in pairs)) / len(pairs) for _ in range(iters))
    return [round(diffs[int(0.025 * iters)], 3), round(diffs[int(0.975 * iters) - 1], 3)]


def _review(reviewer: LLMReviewer | None, path: str, text: str, s: int, e: int) -> list[Finding]:
    if reviewer is None:  # static baseline: analyse the whole file, keep findings in the window
        found = [f for f in check_text(path, text) if s <= f.line <= e]
    else:
        found = reviewer.review_items([{"path": path, "kind": "file", "view": numbered(text, s, e)}],
                                      context="Фрагмент кода из репозитория (функция целиком).")
    return [f for f in found if f.category in ("bug", "security")]


def run(idx, llm: LLM | None, source: str = "synthetic", n: int = 20, tol: int = 2, controls: bool = True,
        seed: int = 13, mode: str = "snippet", progress=None) -> dict:
    trials = synthetic_trials(idx, n=n, seed=seed) if source == "synthetic" else history_trials(idx, n=n, tol=tol)
    reviewer = LLMReviewer(llm) if llm is not None else None
    rows = []
    t0 = time.time()
    for i, t in enumerate(trials):
        if progress:
            progress(i, len(trials), t)
        if mode == "diff" and t.clean_text:
            view = numbered_diff(t.clean_text, t.buggy_text, context=20)
            found = (reviewer.review_items([{"path": t.path, "kind": "diff", "view": view}])
                     if reviewer else [f for f in check_text(t.path, t.buggy_text) if f.line in t.lines])
            found = [f for f in found if f.category in ("bug", "security")]
        else:
            found = _review(reviewer, t.path, t.buggy_text, t.view_start, t.view_end)
        p1 = _chance(t.buggy_text, t.view_start, t.view_end, t.lines, tol)
        row = {**{k: v for k, v in asdict(t).items() if k not in ("buggy_text", "clean_text")},
               "detected": _detected(found, t.path, t.lines, tol),
               "n_findings": len(found),
               "n_localized": sum(_localizes(f, t.path, t.lines, tol) for f in found),
               "chance_one_flag": round(p1, 3),
               "chance_same_volume": round(1 - (1 - p1) ** len(found), 3),
               "findings": [f.to_dict() for f in found][:10]}
        if controls and t.clean_text and mode == "snippet":
            cfound = _review(reviewer, t.path, t.clean_text, t.clean_view_start, t.clean_view_end)
            row["control_false_alarm"] = _detected(cfound, t.path, t.clean_lines or t.lines, tol)
            row["control_any_finding"] = bool(cfound)
            row["control_n_findings"] = len(cfound)
            # reference: what a bug-blind "mutation artefact" detector scores on the same trial
            row["artifact_detected"] = _detected(artifact_review(t.path, t.buggy_text, t.view_start, t.view_end),
                                                 t.path, t.lines, tol)
            row["artifact_false_alarm"] = _detected(
                artifact_review(t.path, t.clean_text, t.clean_view_start, t.clean_view_end),
                t.path, t.clean_lines or t.lines, tol)
        rows.append(row)
    return summarize(idx.slug, source, mode, ("llm:" + llm.model) if llm else "static", tol, rows,
                     time.time() - t0, llm.usage if llm else None)


def run_seeds(idx, llm: LLM | None, seeds: list[int], **kw) -> dict:
    """Synthetic trials for several seeds: per-seed summaries + pooled mean (variance across samples)."""
    runs = [run(idx, llm, seed=sd, **kw) for sd in seeds]
    keys = ("detection_rate", "control_false_alarm_rate", "discrimination", "artifact_baseline_discrimination")
    pooled = {}
    for k in keys:
        vals = [r[k] for r in runs if r.get(k) is not None]
        pooled[k] = {"mean": round(sum(vals) / len(vals), 3), "min": min(vals), "max": max(vals)} if vals else None
    return {"seeds": seeds, "pooled": pooled, "runs": runs}


def summarize(slug, source, mode, reviewer, tol, rows, seconds=0.0, usage=None) -> dict:
    n_tr = len(rows) or 1
    by_kind: dict[str, list[bool]] = {}
    by_op: dict[str, list[bool]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r["detected"])
        by_op.setdefault(r["op"], []).append(r["detected"])
    det = rate([r["detected"] for r in rows])
    ctrl_rows = [r for r in rows if "control_false_alarm" in r]
    fa = rate([r["control_false_alarm"] for r in ctrl_rows]) if ctrl_rows else None
    n_find = sum(r["n_findings"] for r in rows)
    return {
        "slug": slug, "source": source, "mode": mode, "reviewer": reviewer,
        "n_trials": len(rows), "tolerance_lines": tol, "max_finding_span": MAX_SPAN,
        # headline: share of injected bugs found (a localized bug/security finding within ±tol lines)
        "detection_rate": det["rate"], "detection_ci95": det["ci95"],
        # same location in the clean code flagged -> false alarm; detection - false alarms = discrimination
        "control_false_alarm_rate": fa["rate"] if fa else None,
        "control_false_alarm_ci95": fa["ci95"] if fa else None,
        "control_any_finding_rate": round(sum(r["control_any_finding"] for r in ctrl_rows) / len(ctrl_rows), 3)
        if ctrl_rows else None,
        "discrimination": round((sum(r["detected"] for r in ctrl_rows) - sum(r["control_false_alarm"] for r in ctrl_rows))
                                / len(ctrl_rows), 3) if ctrl_rows else None,
        "discrimination_ci95": _bootstrap_diff_ci([r["detected"] for r in ctrl_rows],
                                                  [r["control_false_alarm"] for r in ctrl_rows]) if ctrl_rows else None,
        # reference for "can the trial set be gamed?": bug-blind artefact detector
        "artifact_baseline_discrimination": round(
            sum(r["artifact_detected"] for r in ctrl_rows) / len(ctrl_rows)
            - sum(r["artifact_false_alarm"] for r in ctrl_rows) / len(ctrl_rows), 3) if ctrl_rows else None,
        # share of findings that point at the injected bug
        "precision": round(sum(r["n_localized"] for r in rows) / n_find, 3) if n_find else None,
        "precision_ci95": wilson(sum(r["n_localized"] for r in rows), n_find) if n_find else None,
        "n_findings": n_find,
        # what random flagging would score: 1 random line per trial / as many random lines as the reviewer flagged
        "chance_one_flag": round(sum(r["chance_one_flag"] for r in rows) / n_tr, 3),
        "chance_same_volume": round(sum(r["chance_same_volume"] for r in rows) / n_tr, 3),
        "by_kind": {k: rate(v) for k, v in by_kind.items()},
        "by_op": {k: rate(v) for k, v in sorted(by_op.items())},
        "avg_findings_per_trial": round(n_find / n_tr, 2),
        "seconds": round(seconds, 1),
        "usage": usage,
        "trials": rows,
    }


def save(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"bugs_{result['source']}_{result['reviewer'].replace(':', '-')}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    p = out_dir / name
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return p
