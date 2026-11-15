from conftest import fake_client, json_resp, text_resp, tool_resp

from repo_agent.agent import RepoAgent, RepoTools
from repo_agent.eval import bugs
from repo_agent.llm import LLM
from repo_agent.review.llm_review import LLMReviewer, numbered_git_diff, review_commit, review_repo
from repo_agent.review.static import check_text


def test_tools(demo_index):
    t = RepoTools(demo_index)
    out, files = t.run("read_file", {"path": "core.py", "start_line": 5, "end_line": 7})
    assert "    5 | def add_numbers(a, b):" in out and files == ["demo/core.py"]
    out, _ = t.run("grep", {"pattern": r"shell=True"})
    assert "demo/util.py:13:" in out
    out, _ = t.run("find_symbol", {"name": "clamp"})
    assert "demo/util.py:4-" in out
    out, _ = t.run("list_dir", {"path": "demo"})
    assert "core.py" in out
    out, _ = t.run("git_history", {"path": "demo/util.py"})
    assert "Fix clamp upper bound" in out


def test_agent_loop_with_tools_and_citations(demo_index):
    client, msgs = fake_client([
        tool_resp(("search_code", {"query": "clamp"}), ("read_file", {"path": "demo/util.py", "start_line": 4, "end_line": 9})),
        tool_resp(("read_file", {"path": "missing.py"})),
        text_resp("Ограничение делает `clamp` в demo/util.py:4-9, его вызывает demo/core.py:5-7.\n"
                  "Источники: demo/util.py:4-9, demo/core.py:5-7"),
    ])
    agent = RepoAgent(demo_index, LLM(client=client))
    ans = agent.ask("Где ограничивается результат сложения?")
    assert [s.tool for s in ans.steps] == ["search_code", "read_file", "read_file"]
    assert ans.steps[2].is_error  # missing file -> error goes back to the model
    assert ans.files == ["demo/util.py", "demo/core.py"]
    assert all(c.valid for c in ans.citations)
    req = msgs.requests[0]
    assert req["model"] == "claude-opus-5-5" and req["fallbacks"] == "default"
    assert req["output_config"]["effort"] == "medium" and req["cache_control"] == {"type": "ephemeral"}
    # tool results are returned in one user message, matched by id
    tr = msgs.requests[1]["messages"][-1]["content"]
    assert [b["tool_use_id"] for b in tr] == ["tu0", "tu1"]
    assert msgs.requests[2]["messages"][-1]["content"][0].get("is_error") is True
    # multi-turn: history kept
    assert len(agent.history) == 6


def test_agent_step_cap_forces_final_answer(demo_index):
    client, msgs = fake_client([tool_resp(("search_code", {"query": "x"})),
                                tool_resp(("search_code", {"query": "y"})),
                                text_resp("demo/core.py:5")])
    ans = RepoAgent(demo_index, LLM(client=client), max_steps=2).ask("q")
    assert msgs.requests[-1]["tool_choice"] == {"type": "none"}
    assert ans.files == ["demo/core.py"]


def test_retrieval_mode(demo_index):
    ans = RepoAgent(demo_index, None).ask("subprocess shell run command")
    assert ans.mode == "retrieval" and ans.files[0] == "demo/util.py"


def test_static_rules():
    from conftest import FILES
    f = check_text("demo/util.py", FILES["demo/util.py"])
    rules = {x.rule for x in f}
    assert {"py-subprocess-shell", "secret-hardcoded"} <= rules
    f2 = {x.rule for x in check_text("demo/core.py", FILES["demo/core.py"])}
    assert {"py-mutable-default", "py-bare-except"} <= f2
    assert any(x.rule == "js-innerhtml" for x in check_text("web/helper.js", FILES["web/helper.js"]))


def test_llm_reviewer_parses_findings(demo_index):
    client, msgs = fake_client([json_resp({"findings": [
        {"path": "demo/util.py", "line": 7, "end_line": 8, "severity": "high", "category": "bug",
         "title": "граница", "explanation": "x", "suggestion": "y", "confidence": 0.9},
        {"path": "other.py", "line": 1, "end_line": 1, "severity": "low", "category": "bug",
         "title": "мусор", "explanation": "", "suggestion": "", "confidence": 0.1},
    ]})])
    rep = review_repo(demo_index, LLM(client=client), files=["demo/util.py"])
    llm_f = [f for f in rep["findings"] if f["source"] == "llm"]
    assert len(llm_f) == 1 and llm_f[0]["line"] == 7       # unknown path dropped
    req = msgs.requests[0]
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["output_config"]["effort"] == "high"
    assert "    7 |" in req["messages"][0]["content"]


def test_commit_review_diff_numbers(demo_index):
    sha = demo_index.commits[0].sha
    client, msgs = fake_client([json_resp({"findings": []})])
    rep = review_commit(demo_index, LLM(client=client), sha)
    assert rep["reviewed_files"] == ["demo/util.py"]
    prompt = msgs.requests[0]["messages"][0]["content"]
    assert "+     7 |     if x >= hi:" in prompt and "-       |     if x > hi:" in prompt


def test_numbered_git_diff():
    d = ("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -3,3 +3,3 @@ def f():\n"
         " a = 1\n-b = 2\n+b = 3\n c = 4\n")
    v = numbered_git_diff(d)["x.py"]
    assert "      3 | a = 1" in v and "+     4 | b = 3" in v and "      5 | c = 4" in v


def test_mutation_sites_and_static_eval(demo_index):
    from conftest import FILES
    sites = bugs.find_sites("demo/util.py", FILES["demo/util.py"])
    ops = {op for op, _, _ in sites}
    assert {"lt_to_le", "gt_to_ge"} <= ops
    # mutations never touch strings/comments
    assert not bugs.find_sites("x.py", "s = 'if a == b and c'\n# if x < y\n")
    res = bugs.run(demo_index, None, n=5)
    assert res["n_trials"] >= 2 and 0 <= res["detection_rate"] <= 1


def test_bug_eval_with_fake_llm(demo_index):
    trials = bugs.synthetic_trials(demo_index, n=1)
    t = trials[0]
    hit = {"findings": [{"path": t.path, "line": t.lines[0], "end_line": t.lines[0], "severity": "high",
                         "category": "bug", "title": "b", "explanation": "", "suggestion": "", "confidence": 0.8}]}
    client, _ = fake_client([json_resp(hit), json_resp({"findings": []})])
    res = bugs.run(demo_index, LLM(client=client), n=1)
    assert res["detection_rate"] == 1.0 and res["control_false_alarm_rate"] == 0.0
