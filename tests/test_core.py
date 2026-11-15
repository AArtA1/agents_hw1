from repo_agent import citations
from repo_agent.index import RepoIndex, tokenize
from repo_agent.modmap import module_map
from repo_agent.summary import build_summary, summary_markdown


def test_tokenize_splits_identifiers():
    toks = tokenize("getEnvironProxies should_bypass_proxies HTTPAdapter")
    assert "getenvironproxies" in toks and "environ" in toks and "proxies" in toks
    assert "should_bypass_proxies" in toks and "bypass" in toks
    assert "http" in toks and "adapter" in toks


def test_index_search_and_symbols(demo_index):
    idx = demo_index
    assert "demo/core.py" in idx.files and "tests/test_core.py" in idx.files
    top = idx.search("clamp lower upper bound", k=3)
    assert top[0].chunk.path == "demo/util.py"
    syms = idx.find_symbol("Store.load")
    assert syms and syms[0].path == "demo/core.py" and syms[0].line == 14
    assert idx.find_symbol("helper")[0].path == "web/helper.js"
    assert idx.commits[0].subject == "Fix clamp upper bound"
    assert idx.search_commits("clamp fix")[0].files == ["demo/util.py"]


def test_index_roundtrip(demo_index):
    again = RepoIndex.load(demo_index.slug)
    assert len(again.chunks) == len(demo_index.chunks)
    assert again.search("add_numbers")[0].chunk.path == demo_index.search("add_numbers")[0].chunk.path


def test_resolve_path_suffix(demo_index):
    assert demo_index.resolve_path("core.py") == "demo/core.py"
    assert demo_index.resolve_path("./demo/util.py") == "demo/util.py"
    assert demo_index.resolve_path("nope.py") is None


def test_citations_validation(demo_index):
    text = "Сложение в `demo/core.py:5-7`, ограничение в util.py:4-9, а demo/core.py:500 не существует."
    cs = citations.extract(text, demo_index)
    by = {c.label: c for c in cs}
    assert by["demo/core.py:5-7"].valid
    assert by["demo/util.py:4-9"].valid and by["demo/util.py:4-9"].resolved == "demo/util.py"
    assert not by["demo/core.py:500"].valid
    assert citations.cited_files(cs) == ["demo/core.py", "demo/util.py"]


def test_summary_detects_stack_and_entry_points(demo_index):
    s = build_summary(demo_index)
    assert "Flask" in s["frameworks"]
    kinds = {(e["kind"], e["path"]) for e in s["entry_points"]}
    assert ("console script", "pyproject.toml") in kinds
    assert ("python __main__", "demo/cli.py") in kinds
    assert s["readme_intro"].startswith("Demo is a tiny package")
    assert "Точки входа" in summary_markdown(s)


def test_module_map(demo_index):
    m = module_map(demo_index, level="file")
    edges = {(e["src"], e["dst"]) for e in m["edges"]}
    assert ("demo/core.py", "demo/util.py") in edges
    assert ("demo/cli.py", "demo/core.py") in edges
    assert ("web/app.js", "web/helper.js") in edges
    assert m["mermaid"].startswith("flowchart LR")
    d = module_map(demo_index, level="dir", depth=1)
    assert {n["id"] for n in d["nodes"]} >= {"demo", "web"}


def test_list_indexed(demo_index):
    assert RepoIndex.list_indexed() == [demo_index.slug]
