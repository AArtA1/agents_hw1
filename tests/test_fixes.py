"""Regression tests for the issues found by the validators."""

import json
import subprocess
from types import SimpleNamespace

import pytest
from conftest import fake_client, text_resp, tool_resp

from repo_agent import config
from repo_agent.agent import RepoAgent, RepoTools
from repo_agent.citations import extract, linkify
from repo_agent.index import RepoIndex
from repo_agent.llm import LLM


def make_repo(tmp_path, files: dict, name="r", remote=None):
    root = tmp_path / name
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True)
    run("init", "-q")
    if remote:
        run("remote", "add", "origin", remote)
    run("add", ".")
    run("-c", "user.email=t@t", "-c", "user.name=Alice", "commit", "-q", "-m", "init")
    return root


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE", tmp_path / "ws")
    return tmp_path


def build(tmp_path, files, **kw):
    return RepoIndex.build(str(make_repo(tmp_path, files, **kw)), with_issues=False)


# ------------------------------------------------------------------ index / paths / lines

def test_dotfiles_resolve_and_non_ascii_names(ws):
    idx = build(ws, {".github/workflows/ci.yml": "on: push\n", "pkg/данные.py": "x = 1\n", "a.py": "y = 2\n"})
    assert idx.resolve_path(".github/workflows/ci.yml") == ".github/workflows/ci.yml"
    assert idx.resolve_path("./a.py") == "a.py"
    assert "pkg/данные.py" in idx.files


def test_form_feed_does_not_shift_lines(ws):
    idx = build(ws, {"m.py": "def a():\n    return 1\n\x0c\ndef b():\n    return 2\n"})
    lines, total = idx.read_lines("m.py", 4, 4)
    assert total == 5 and lines == ["def b():"]
    assert idx.find_symbol("b")[0].line == 4


def test_symbols_inside_if_and_try_blocks(ws):
    idx = build(ws, {"u.py": "import sys\nif sys.platform == 'win32':\n    def proxy_bypass(h):\n        return 1\n"
                             "try:\n    import x\nexcept ImportError:\n    def fallback():\n        pass\n"})
    assert idx.find_symbol("proxy_bypass")[0].line == 3
    assert idx.find_symbol("fallback")


def test_ts_symbols(ws):
    idx = build(ws, {"src/k.ts": "export const norm = (s: string): string => s;\nexport class Ky {\n"
                                 "  async #retryDelay(e: Error): Promise<number> {\n    return 1;\n  }\n}\n"
                                 "res.send = function send(body) {\n  return 1;\n};\n"})
    assert idx.find_symbol("norm") and idx.find_symbol("Ky.#retryDelay") and idx.find_symbol("send")


# ------------------------------------------------------------------ citations

def test_citation_edge_cases(ws):
    idx = build(ws, {"a.py": "\n".join(f"x{i} = {i}" for i in range(10)) + "\n", "Dockerfile": "FROM x\nCMD y\n"},
                remote="https://github.com/o/r.git")
    by = {c.label: c for c in extract("a.py:1-11 a.py:1-10 Dockerfile:2", idx)}
    assert not by["a.py:1-11"].valid and by["a.py:1-10"].valid and by["Dockerfile:2"].valid
    url_cites = extract("[a.py:3](https://github.com/o/r/blob/abc/a.py#L3) and https://github.com/o/r/blob/x/a.py#L9", idx)
    assert [c.label for c in url_cites] == ["a.py:3"] and url_cites[0].valid
    out = linkify("see `a.py#L2-L3` and `x = a.py:3` and a.py:4", idx)
    assert "[a.py#L2-L3](" in out and "`x = a.py:3`" in out and "[a.py:4](" in out


# ------------------------------------------------------------------ agent

def test_truncated_tool_call_keeps_history_valid(demo_index):
    trunc = tool_resp(("search_code", {"query": "x"}))
    trunc.stop_reason = "max_tokens"
    client, msgs = fake_client([trunc, text_resp("demo/core.py:5")])
    agent = RepoAgent(demo_index, LLM(client=client))
    agent.ask("q1")
    agent.ask("q2")
    sent = msgs.requests[1]["messages"]
    # every tool_use in history is followed by a matching tool_result
    tu_ids = [b.id for m in sent if m["role"] == "assistant" for b in m["content"] if getattr(b, "type", "") == "tool_use"]
    tr_ids = [b["tool_use_id"] for m in sent if m["role"] == "user" and isinstance(m["content"], list)
              for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]
    assert tu_ids == tr_ids == ["tu0"]


def test_long_history_is_compacted(demo_index, monkeypatch):
    import repo_agent.agent as agent_mod
    monkeypatch.setattr(agent_mod, "HISTORY_BUDGET_CHARS", 10)
    client, msgs = fake_client([text_resp("first answer demo/core.py:5"), text_resp("second")])
    agent = RepoAgent(demo_index, LLM(client=client))
    agent.ask("first question that is long enough")
    agent.ask("q2")
    second = msgs.requests[1]["messages"]
    assert len(second) == 1 and "first question" in second[0]["content"] and "q2" in second[0]["content"]
    assert agent.turns[-1][0] == "q2"  # the recap is not stored as part of the question


def test_tool_input_edge_cases(demo_index):
    t = RepoTools(demo_index)
    assert "core.py" in t.run("list_dir", {"path": "./demo"})[0]
    assert "demo/" in t.run("list_dir", {"path": "."})[0]
    with pytest.raises(ValueError):
        t.run("read_file", {"path": "demo/core.py", "start_line": 500})
    blame = t.run("blame", {"path": "demo/util.py", "start_line": 1, "end_line": 1})[0]
    assert "T " in blame or "T\t" in blame or "(T" in blame  # author shown
    with pytest.raises(ValueError):
        t.run("blame", {"path": "demo/util.py", "start_line": 5, "end_line": 2})
    assert "demo/util.py:" in t.run("grep", {"pattern": "def clamp", "path_prefix": "./demo"})[0]
    hits = t.run("search_code", {"query": "clamp", "path_prefix": "./demo"})[0]
    assert "demo/util.py" in hits


# ------------------------------------------------------------------ module map

def test_module_map_resolution(ws):
    from repo_agent.modmap import build_graph
    files = {
        "pkg/__init__.py": "VERSION = 1\n",
        "pkg/a.py": "from . import b\nfrom typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .c import C\n",
        "pkg/b.py": "x = 1\n",
        "pkg/c.py": "class C: pass\n",
        "src/core/Ky.ts": "import {x} from '../utils/x.js';\nexport class Ky {}\n",
        "src/utils/x.ts": "export const x = 1;\n",
        "index.js": "module.exports = require('./lib/app');\n",
        "lib/app.js": "module.exports = 1;\n",
        "examples/demo/index.js": "const e = require('../..');\n",
        "go.mod": "module github.com/o/r\n",
        "root.go": "package r\n",
        "doc/gen.go": 'package doc\n\nimport (\n\t"fmt"\n\t"github.com/o/r"\n)\n',
    }
    idx = build(ws, files)
    g = build_graph(idx, include_tests=True)
    edges = {(e["src"], e["dst"]) for e in g["edges"]}
    assert ("pkg/a.py", "pkg/b.py") in edges and ("pkg/a.py", "pkg/__init__.py") not in edges
    assert ("pkg/a.py", "pkg/c.py") not in edges            # TYPE_CHECKING-only import
    assert ("src/core/Ky.ts", "src/utils/x.ts") in edges    # './x.js' -> x.ts
    assert ("examples/demo/index.js", "index.js") in edges  # require('../..')
    assert ("doc/", "(root)/") in edges                      # Go packages, not arbitrary files
    g2 = build_graph(idx)                                    # examples excluded by default
    assert not any(e["src"].startswith("examples/") for e in g2["edges"])


# ------------------------------------------------------------------ summary

def test_entry_points_and_stack(ws):
    from repo_agent.summary import build_summary
    files = {
        "pyproject.toml": '[project]\nname = "flaskish"\ndependencies = ["click"]\n'
                          '[project.optional-dependencies]\ndev = ["pytest"]\n',
        "src/flaskish/__init__.py": "",
        "src/flaskish/__main__.py": "print(1)\n",
        "src/flaskish/app.py": 'class App:\n    """Usage::\n\n        app = Flask(__name__)\n    """\n',
        "docs_src/tutorial/main.py": "from fastapi import FastAPI\napp = FastAPI()\n",
        "web/package.json": json.dumps({"name": "w", "devDependencies": {"express": "1"}}),
        "web/index.js": "module.exports = 1\n",
        "README.md": "> **Flaskish** is a [tiny](http://x) framework for building small web apps quickly.\n",
    }
    s = build_summary(build(ws, files))
    eps = {(e["kind"], e["path"], e["detail"]) for e in s["entry_points"]}
    assert ("python -m", "src/flaskish/__main__.py", "python -m flaskish") in eps
    assert ("library API", "src/flaskish/__init__.py", "import flaskish") in eps
    assert ("npm main", "web/index.js", "index.js (неявный main)") in eps
    assert not any(k == "web app" for k, _, _ in eps)        # docstring example is not an app
    assert s["entry_points_aux"] == 1                        # docs_src app counted, not shown
    assert "Express" not in s["frameworks"] and "Express" in s["dev_tools"] and "Click" in s["frameworks"]
    assert s["readme_intro"].startswith("Flaskish is a tiny framework")


# ------------------------------------------------------------------ static review

def test_static_review_noise():
    from repo_agent.review.static import check_text
    code = ('apiKey = "apiKey"\npassword = "changeme123"\nx = eval(data)  # noqa: S307\n'
            'msg = "never call eval(x) or verify=False"\n'
            'try:\n    f()\nexcept:\n    cleanup()\n    raise\n'
            'db_password = "Zq8!vR2#pL9$wX4@"\n')
    rules = [(f.line, f.rule, f.severity) for f in check_text("app/x.py", code)]
    assert rules == [(10, "secret-hardcoded", "high")]
    assert all(f.severity == "low" for f in check_text("docs/js/t.js", "el.innerHTML = v;\n"))


# ------------------------------------------------------------------ metrics

def test_bug_metric_is_not_gameable(demo_index, monkeypatch):
    from repo_agent.eval import bugs
    from repo_agent.review.static import Finding

    def flag_all(reviewer, path, text, s, e):
        return [Finding(path, l, "high", "bug", "x", "") for l in range(s, e + 1)]

    monkeypatch.setattr(bugs, "_review", flag_all)
    r = bugs.run(demo_index, None, n=5)
    assert r["detection_rate"] == 1.0 and r["control_false_alarm_rate"] == 1.0 and r["discrimination"] == 0.0
    assert r["precision"] < 1.0

    def one_wide(reviewer, path, text, s, e):
        return [Finding(path, s, "high", "bug", "x", "", end_line=e)]

    monkeypatch.setattr(bugs, "_review", one_wide)
    r = bugs.run(demo_index, None, n=5)
    t = bugs.synthetic_trials(demo_index, n=5)
    assert r["detection_rate"] == sum(abs(x.view_start - x.lines[0]) <= 2 for x in t) / len(t)


def test_qa_gold_extraction(ws):
    from repo_agent.eval import qa
    idx = build(ws, {"src/app/app.py": "", "src/app/views.py": "", "tests/apps/app.py": "", "tests/apps/wsgi.py": "",
                     "docs/x.rst": "", "CHANGES.rst": ""})
    item = {"answer": "In `src/app/views.py` the `app.py` module (see docs/x.rst, CHANGES.rst) and a wsgi.py file"}
    assert qa.gold_files(item, idx) == ["src/app/views.py", "src/app/app.py"]


def test_qa_popular_baseline(demo_index):
    from repo_agent.eval import qa
    items = [{"question": "clamp lower bound", "gold_files": ["demo/util.py"]},
             {"question": "add numbers", "gold_files": ["demo/core.py"]},
             {"question": "clamp upper", "gold_files": ["demo/util.py"]}]
    r = qa.evaluate(demo_index, items, mode="retrieval")
    assert r["baseline_popular_files"][0] == "demo/util.py"
    assert r["metrics"]["baseline_popular_hit1"] == round(2 / 3, 3)
    assert r["metrics"]["primary_hit_ci95"] is not None


# ------------------------------------------------------------------ issues & datasets

def test_issues_search_tool_and_refs(demo_index, monkeypatch):
    from repo_agent import issues
    assert issues.referenced_numbers("Fix clamp (#12), see #7 and org/x#99") == [12, 7]
    demo_index.issues = [{"number": 12, "title": "clamp returns wrong upper bound", "body": "x > hi", "state": "closed",
                          "labels": ["bug"], "is_pr": False}]
    from repo_agent.index import BM25, tokenize
    demo_index._issue_bm25 = BM25([tokenize(i["title"] + " " + i["body"]) for i in demo_index.issues])
    out = RepoTools(demo_index).run("search_issues", {"query": "clamp upper bound"})[0]
    assert "#12" in out and "bug" in out


def test_commit_review_includes_referenced_issue(demo_repo, monkeypatch):
    from repo_agent import issues
    from repo_agent.review import llm_review
    run = lambda *a: subprocess.run(["git", "-C", str(demo_repo), *a], check=True, capture_output=True)
    (demo_repo / "demo/util.py").write_text((demo_repo / "demo/util.py").read_text().replace("x < lo", "x <= lo"))
    run("-c", "user.email=t@t", "-c", "user.name=T", "commit", "-qam", "Tweak clamp lower bound (fixes #42)")
    idx = RepoIndex.build(str(demo_repo), with_issues=False)
    monkeypatch.setattr(issues, "get_issue", lambda slug, url, n: {"number": n, "title": "clamp lower bound bug",
                                                                    "body": "details", "state": "closed", "url": "u"})
    client, msgs = fake_client([text_resp(json.dumps({"findings": []}))])
    rep = llm_review.review_commit(idx, LLM(client=client), idx.commits[0].sha)
    assert rep["issues"][0]["number"] == 42
    assert "clamp lower bound bug" in msgs.requests[0]["messages"][0]["content"]


def test_dataset_loaders(tmp_path):
    from repo_agent.eval import datasets
    assert datasets.html_to_text("<p>Use <code>Parameter</code></p><pre><code>a &lt; b</code></pre>").startswith("Use `Parameter`")
    assert datasets._class_names("<code>@Parameter(names=x) JCommander jc; String s</code>")[:2] == ["Parameter", "JCommander"]
    rec = {"title": "Crash in parser", "body": "stack", "repository_url": "https://api.github.com/repos/o/r",
           "html_url": "u", "comments_details": [
               {"body": "me too", "author_association": "NONE"},
               {"body": "Fixed in `src/parser.py`", "author_association": "MEMBER"}]}
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    items = datasets.load_coderepoqa(str(p), repo="o/r")
    assert items[0]["answer"] == "Fixed in `src/parser.py`" and items[0]["repo"] == "o/r"


# ------------------------------------------------------------------ round 2

def test_mask_c_like_scanner():
    from repo_agent.textutil import mask_c_like
    js = mask_c_like('var r = /["\']/; x = 1; y = "a//b"; z = 2;\nconst re = /`/; foo()\nbar() // c\n', "javascript")
    assert "x = 1" in js[0] and "z = 2" in js[0] and "foo()" in js[1] and js[2].strip() == "bar()"
    rs = mask_c_like("fn f<'a>(x: &'a str) { let q = 'x'; }\n", "rust")[0]
    assert "x: &'a str" in rs and "'x'" not in rs


def test_js_methods_need_a_body_and_braces_end_symbols(ws):
    from repo_agent.chunking import extract_symbols
    ts = "class Foo {\n  constructor(a) {\n    doSomething(a)\n    helper(\n      1)\n  }\n}\nconst total = (\n  1\n)\n"
    names = [s.qualname for s in extract_symbols("a.ts", ts)]
    assert names == ["Foo", "Foo.constructor"]
    java = "public class R {\n  public boolean ok() {\n    return x;\n  }\n  int peek() {\n    if (a) { b(); }\n    return 1;\n  }\n}\n"
    spans = {s.name: (s.line, s.end_line) for s in extract_symbols("R.java", java)}
    assert spans == {"R": (1, 9), "ok": (2, 4), "peek": (5, 8)}


def test_issue_fetch_errors_never_break_indexing(ws, monkeypatch):
    from repo_agent import issues
    for exc in (ConnectionResetError("reset"), ValueError("bad json")):
        def boom(url, timeout=20, exc=exc):
            try:
                raise exc
            except (OSError, ValueError) as e:
                raise issues.IssuesError(str(e)) from e
        monkeypatch.setattr(issues, "_get", boom)
        idx = RepoIndex.build(str(make_repo(ws, {"a.py": "x = 1\n"}, name=f"r{type(exc).__name__}",
                                            remote="https://github.com/o/r.git")))
        assert idx.meta["issues_error"] and idx.issues == []
    monkeypatch.setattr(issues, "_get", lambda url, timeout=20: {"not": "a list"})
    with pytest.raises(issues.IssuesError):
        issues.fetch_issues("x", "https://github.com/o/r")


def test_citation_paths_round2(ws):
    idx = build(ws, {"app/[id]/page.tsx": "a\nb\n", "src/@types/x.d.ts": "a\n", "lib/c++/m.cpp": "a\n", "src/t.py": "a\nb\n"},
                remote="https://github.com/o/r.git")
    t = "app/[id]/page.tsx:1-2 src/@types/x.d.ts:1 lib/c++/m.cpp:1 ../src/t.py:2 api.github.com:443 v1.2.3:4"
    assert [(c.label, c.valid) for c in extract(t, idx)] == [
        ("app/[id]/page.tsx:1-2", True), ("src/@types/x.d.ts:1", True), ("lib/c++/m.cpp:1", True), ("src/t.py:2", True)]
    out = linkify("``src/t.py:1`` ~~~\nsrc/t.py:2\n~~~ src/t.py:1", idx)
    assert out.startswith("``src/t.py:1`` ~~~\nsrc/t.py:2\n~~~ [src/t.py:1](")


def test_static_rules_round2():
    from repo_agent.review.static import check_text
    py = ('cur.execute("SELECT * FROM t WHERE id=%s" % x)\ncur.execute(f"SELECT * FROM t WHERE id={x}")\n'
          '# cur.execute("SELECT %s" % x)\ns = "cur.execute(\'SELECT %s\' % x)"\n')
    assert [(f.line, f.rule) for f in check_text("app/db.py", py)] == [(1, "sql-concat"), (2, "sql-concat")]
    java = 'class A {\n  void f(String c) {\n    Runtime.getRuntime().exec(c);\n    new ObjectInputStream(in).readObject();\n  }\n}\n'
    assert {f.rule for f in check_text("src/A.java", java)} == {"java-runtime-exec", "java-deserialization"}
    rs = 'fn main() {\n    b.danger_accept_invalid_certs(true);\n}\n'
    assert [f.rule for f in check_text("src/main.rs", rs)] == ["rust-tls-off"]


def test_summary_round2(ws):
    from repo_agent.summary import build_summary
    files = {"pyproject.toml": '[project]\nname = "tool"\nurls = {"tool docs" = "https://x/tool"}\n\n'
                               '[project.scripts]\ntool = "tool.cli:main"\n',
             "tool/__init__.py": "", "tool/cli.py": "def main(): pass\n",
             "metrics/src/main/java/ParseBenchmark.java": "class P {\n  public static void main(String[] a) {}\n}\n",
             "app/dashboard/page.tsx": "export default function P() {}\n",
             "README.md": "tool\n====\n\n> [!NOTE]\n> Tool is a small [CLI][1] utility that formats code in place and quickly.\n"}
    s = build_summary(build(ws, files))
    kinds = [(e["kind"], e["path"], e["line"]) for e in s["entry_points"]]
    assert kinds[0] == ("console script", "pyproject.toml", 6)
    assert ("next route", "app/dashboard/page.tsx", 1) in kinds
    assert kinds[-1][0] == "benchmark main"
    assert s["readme_intro"].startswith("Tool is a small CLI utility")


def test_module_map_round2(ws):
    from repo_agent.modmap import build_graph
    files = {
        "tsconfig.json": '{\n  // comment\n  "compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["./*"]}},\n'
                         '  "include": ["**/*.ts"],\n}\n',
        "app/page.tsx": "import { auth } from '@/lib/auth'\n", "lib/auth.ts": "export const auth = 1\n",
        "lib/src/main/java/com/x/A.java": "package com.x;\nimport com.x.util.B;\nclass A { C c; B b; }\n",
        "lib/src/main/java/com/x/C.java": "package com.x;\nclass C {}\n",
        "lib/src/main/java/com/x/util/B.java": "package com.x.util;\npublic class B {}\n",
        "Cargo.toml": '[package]\nname = "my-crate"\n', "src/lib.rs": "mod search;\npub use crate::search::run;\n",
        "src/search.rs": "use super::util::x;\npub fn run() {}\n", "src/util/mod.rs": "pub fn x() {}\n",
    }
    edges = {(e["src"], e["dst"]) for e in build_graph(build(ws, files))["edges"]}
    assert ("app/page.tsx", "lib/auth.ts") in edges
    assert ("lib/src/main/java/com/x/A.java", "lib/src/main/java/com/x/util/B.java") in edges
    assert ("lib/src/main/java/com/x/A.java", "lib/src/main/java/com/x/C.java") in edges   # same package
    assert ("src/lib.rs", "src/search.rs") in edges and ("src/search.rs", "src/util/mod.rs") in edges


def test_report_uses_most_recent_eval(ws):
    import os, time as _t
    from repo_agent import report
    d = config.WORKSPACE / "s" / "eval"
    d.mkdir(parents=True)
    (d / "bugs_synthetic_x.json").write_text('{"source": "synthetic"}')
    (d / "bugs_history_x.json").write_text('{"source": "history"}')
    os.utime(d / "bugs_synthetic_x.json", (_t.time() - 100, _t.time() - 100))
    assert report.latest_eval("s", "bugs_")["source"] == "history"


def test_history_trials_have_aligned_controls_and_artifact_baseline(ws):
    from repo_agent.eval import bugs
    body = "\n".join(f"    v{i} = {i}" for i in range(14))
    old = f"import os\n\n\ndef helper():\n    return 1\n\n\ndef parse(items):\n{body}\n    if len(items) > 3:\n        return items[0]\n    return None\n"
    root = make_repo(ws, {"pkg/p.py": old})
    new = old.replace("len(items) > 3", "len(items) >= 3").replace("import os\n", "import os\nimport sys\n")
    (root / "pkg/p.py").write_text(new)
    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-qam",
                    "Fix parse returning None for three items"], check=True)
    idx = RepoIndex.build(str(root), with_issues=False)
    [t] = bugs.history_trials(idx, n=5)
    assert t.lines == [23] and t.clean_lines == [24]                 # fixed version shifted by one line
    assert (t.clean_view_start, t.clean_view_end) == (9, 26)        # same function `parse`, matched by name
    r = bugs.run(idx, None, source="history", n=5)
    assert r["n_trials"] == 1 and "artifact_baseline_discrimination" in r and "discrimination_ci95" in r


def test_qa_ambiguous_bare_name_and_leak(ws):
    from repo_agent.eval import qa
    idx = build(ws, {"src/flask/app.py": "", "src/flask/sansio/app.py": "", "src/flask/cli.py": ""})
    assert qa.gold_files({"answer": "see app.py"}, idx) == ["src/flask/app.py", "src/flask/sansio/app.py"]
    r = qa.evaluate(idx, [{"question": "what does cli do", "gold_files": ["src/flask/cli.py"]},
                          {"question": "how is the app built", "gold_files": ["src/flask/app.py"]}], mode="retrieval")
    assert [row["gold_in_question"] for row in r["rows"]] == [False, False]
    r2 = qa.evaluate(idx, [{"question": "what is in cli module", "gold_files": ["src/flask/cli.py"]}], mode="retrieval")
    assert r2["rows"][0]["gold_in_question"] is False  # 'cli' has <= 3 chars -> ignored
