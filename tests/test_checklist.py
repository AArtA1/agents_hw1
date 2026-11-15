"""One test (or more) per item of the homework checklist."""

import re
import subprocess
from pathlib import Path

import pytest
from conftest import fake_client, text_resp, tool_resp

from repo_agent import config, repo as repo_mod
from repo_agent.agent import RepoAgent
from repo_agent.llm import LLM, LLMError
from repo_agent.tools import TOOLS, RepoTools, api_tools, validate_args

ROOT = Path(__file__).resolve().parent.parent


# 1. tool descriptions pass the "intern test": what / when / input / output / errors are all written down
def test_tool_descriptions_are_complete():
    for t in TOOLS:
        d = t["description"]
        assert len(d) >= 80, t["name"]
        assert all(p.get("type") for p in t["input_schema"]["properties"].values())
        assert t["input_schema"]["additionalProperties"] is False
        assert t["spec"]["kind"] in ("read", "action")
    assert all("spec" not in t for t in api_tools())  # private metadata is not sent to the model


# 2. arguments are validated against the schema; the error goes back to the model
def test_schema_validation_messages():
    schema = next(t for t in TOOLS if t["name"] == "read_file")["input_schema"]
    errs = validate_args(schema, {"start_line": "10", "extra": 1, "end_line": 0})
    text = " ".join(errs)
    assert "не хватает обязательного поля 'path'" in text and "должно быть integer" in text
    assert "неизвестное поле 'extra'" in text and ">= 1" in text
    assert validate_args(schema, {"path": "a.py", "start_line": True})  # bool is not an integer
    assert validate_args(schema, {"path": "a.py", "start_line": 3}) == []


def test_invalid_args_go_back_to_model_and_agent_recovers(demo_index):
    client, msgs = fake_client([
        tool_resp(("read_file", {"path": "demo/util.py", "start_line": "4"})),   # wrong type
        tool_resp(("read_file", {"path": "demo/util.py", "start_line": 4, "end_line": 9})),
        text_resp("demo/util.py:4-9"),
    ])
    ans = RepoAgent(demo_index, LLM(client=client)).ask("q")
    assert [s.status for s in ans.steps] == ["invalid_args", "ok"]
    err_result = msgs.requests[1]["messages"][-1]["content"][0]
    assert err_result["is_error"] is True and "должно быть integer" in err_result["content"]
    assert ans.files == ["demo/util.py"]


def test_unknown_tool_is_an_error_not_a_crash(demo_index):
    res = RepoTools(demo_index).call("rm_rf", {})
    assert res.status == "unknown_tool" and res.is_error


# 3. the client/"account" (here: the repository, its clone, GitHub owner/token) comes from the session
def test_repository_is_bound_to_session_not_model(demo_index):
    for t in TOOLS:
        props = set(t["input_schema"]["properties"])
        assert not props & {"repo", "repository", "owner", "root", "token", "workspace"}, t["name"]
    res = RepoTools(demo_index).call("read_file", {"path": "../../../../etc/passwd"})
    assert res.status == "error" and "не найден в индексе" in res.content


# 4. a service timeout does not kill the agent
def test_git_timeout_becomes_tool_timeout(demo_index, monkeypatch):
    def slow(*a, **kw):
        raise repo_mod.GitTimeout("git log не уложился в 20 с")
    monkeypatch.setattr(repo_mod, "commits", slow)
    res = RepoTools(demo_index).call("git_history", {"path": "demo/util.py"})
    assert res.status == "timeout" and "Таймаут" in res.content


def test_real_subprocess_timeout(tmp_path):
    with pytest.raises(repo_mod.GitTimeout):
        repo_mod.git_run(tmp_path, "-c", "alias.slow=!sleep 3", "slow", timeout=0.5)


def test_model_timeout_returns_answer_not_exception(demo_index):
    class Down:
        def __init__(self):
            self.beta = self
            self.messages = self

        def create(self, **kw):
            raise LLMError("Claude API не ответил за 120 с (таймаут).")
    agent = RepoAgent(demo_index, LLM(client=Down()))
    ans = agent.ask("q")
    assert ans.error and "Модель недоступна" in ans.text and agent.history == []


def test_fault_injection_then_recovery(demo_index):
    client, _ = fake_client([
        tool_resp(("read_file", {"path": "demo/util.py"})),
        tool_resp(("read_file", {"path": "demo/util.py"})),
        text_resp("demo/util.py:4"),
    ])
    ans = RepoAgent(demo_index, LLM(client=client), fault="read_file:timeout").ask("q")
    assert [s.status for s in ans.steps] == ["timeout", "ok"]


# 5. tool results are limited in size
def test_tool_output_is_truncated(demo_index, monkeypatch):
    monkeypatch.setattr(config, "MAX_TOOL_OUTPUT", 120)
    res = RepoTools(demo_index).call("read_file", {"path": "demo/core.py"})
    assert res.truncated and len(res.content) < 260 and "результат обрезан" in res.content


# 6. step limit
def test_step_limit_from_config(demo_index, monkeypatch):
    monkeypatch.setattr(config, "MAX_STEPS", 2)
    client, msgs = fake_client([tool_resp(("list_dir", {"path": ""}))] * 2 + [text_resp("done")])
    agent = RepoAgent(demo_index, LLM(client=client))
    agent.ask("q")
    assert agent.max_steps == 2 and len(msgs.requests) == 3
    assert msgs.requests[-1]["tool_choice"] == {"type": "none"}


# 7. an action is executed only after a human confirms it
ISSUE = {"title": "clamp ignores upper bound", "body": "demo/util.py:7 uses >= instead of > ..."}


def _issue_flow(demo_index, **agent_kw):
    client, msgs = fake_client([tool_resp(("create_issue", ISSUE)), text_resp("ok")])
    agent = RepoAgent(demo_index, LLM(client=client), **agent_kw)
    ans = agent.ask("оформи issue")
    drafts = config.WORKSPACE / demo_index.slug / "issue_drafts"
    return agent, ans, msgs, drafts


def test_action_rejected_by_human(demo_index):
    agent, ans, msgs, drafts = _issue_flow(demo_index, confirm=lambda *a: False)
    assert ans.steps[0].status == "rejected" and not drafts.exists()
    assert "ОТКЛОНИЛ" in msgs.requests[1]["messages"][-1]["content"][0]["content"]


def test_action_approved_by_human(demo_index):
    seen = []
    agent, ans, msgs, drafts = _issue_flow(demo_index, confirm=lambda tool, args, preview: seen.append(preview) or True)
    assert ans.steps[0].status == "approved" and "clamp ignores upper bound" in seen[0]
    assert len(list(drafts.glob("*.md"))) == 1


def test_action_pending_until_ui_confirms(demo_index):
    agent, ans, msgs, drafts = _issue_flow(demo_index)          # no callback (web UI)
    assert ans.steps[0].status == "pending" and not drafts.exists()
    assert "НЕ выполнено" in msgs.requests[1]["messages"][-1]["content"][0]["content"]
    assert agent.resolve_pending(1, approve=True).startswith("Черновик issue сохранён")
    assert len(list(drafts.glob("*.md"))) == 1
    assert "уже обработано" in agent.resolve_pending(1, approve=True)


def test_action_never_runs_through_plain_run(demo_index):
    with pytest.raises(ValueError):
        RepoTools(demo_index).run("create_issue", ISSUE)


# 8. provider in config, no keys in the repository
def test_provider_and_model_come_from_config(tmp_path, monkeypatch):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[llm]\nprovider = "anthropic"\nmodel = "claude-sonnet-5-5"\n[agent]\nmax_steps = 5\n')
    s = config.load_settings(cfg)
    assert s["llm"]["model"] == "claude-sonnet-5-5" and s["agent"]["max_steps"] == 5
    monkeypatch.setenv("REPO_AGENT_AGENT_MAX_STEPS", "7")
    assert config.load_settings(cfg)["agent"]["max_steps"] == 7
    cfg.write_text('[llm]\nprovider = "openai"\n')
    with pytest.raises(config.ConfigError):
        config.load_settings(cfg)
    real = config.load_settings(ROOT / "repo_agent.toml")
    assert real["llm"]["provider"] in config.SUPPORTED_PROVIDERS and real["llm"]["model"]
    cfg.write_text('[llm]\nprovider = "openai_compatible"\nmodel = "m"\n')
    with pytest.raises(config.ConfigError):  # base_url / api_key_env required
        config.load_settings(cfg)
    cfg.write_text('[actions]\npost_to_github = "false"\n')
    with pytest.raises(config.ConfigError):  # wrong type is an error, not silently truthy
        config.load_settings(cfg)


def test_no_secrets_in_repository():
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, text=True).stdout.split()
    pat = re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{36,}|AKIA[0-9A-Z]{16}")
    leaks = [f for f in tracked if (ROOT / f).is_file() and pat.search((ROOT / f).read_text(errors="ignore"))]
    assert leaks == []
    assert ".env" in (ROOT / ".gitignore").read_text().split()


# ---------------------------------------------------------------- red-team regressions

def test_openai_compatible_adapter_roundtrip():
    from types import SimpleNamespace
    from repo_agent.openai_compat import from_openai_response, to_openai_messages, to_openai_tools
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": [SimpleNamespace(type="text", text="look"),
                                              SimpleNamespace(type="tool_use", id="c1", name="read_file",
                                                              input={"path": "a.py"})]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "boom",
                                          "is_error": True}]}]
    out = to_openai_messages(msgs, system="sys")
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert out[2]["tool_calls"][0]["function"]["name"] == "read_file" and out[3]["content"].startswith("[TOOL ERROR]")
    assert to_openai_tools([{"name": "t", "description": "d", "input_schema": {"type": "object"}}])[0]["type"] == "function"
    r = from_openai_response({"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
        {"id": "x", "function": {"name": "grep", "arguments": "{bad json"}}]}}]})
    assert r.stop_reason == "tool_use" and r.content[0].input == "{bad json"


def test_symlink_outside_repo_is_not_indexed_or_read(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(config, "WORKSPACE", tmp_path / "ws")
    r = tmp_path / "r"
    r.mkdir()
    (r / "a.py").write_text("x = 1\n")
    secret = tmp_path / "secret.env"
    secret.write_text("TOKEN=ghp_" + "a" * 36 + "\n")
    os.symlink(secret, r / "config.env")
    for a in (["init", "-q"], ["add", "."], ["-c", "user.email=a@a", "-c", "user.name=a", "commit", "-qm", "x"]):
        subprocess.run(["git", "-C", str(r), *a], check=True)
    from repo_agent.index import RepoIndex
    idx = RepoIndex.build(str(r), with_issues=False)
    assert idx.files == ["a.py"]
    assert RepoTools(idx).call("read_file", {"path": "config.env"}).status == "error"
    assert repo_mod.read_text(r, "config.env") is None


def test_grep_catastrophic_regex_is_bounded(demo_index, monkeypatch):
    import time
    monkeypatch.setattr(config, "TOOL_TIMEOUT", 1.0)
    (demo_index.root / "demo" / "core.py").write_text("a" * 60 + "b\n")
    t0 = time.time()
    res = RepoTools(demo_index).call("grep", {"pattern": "^(a|aa)+$"})
    assert time.time() - t0 < 5 and res.status in ("timeout", "ok")


def test_confirmation_requires_explicit_true_and_runs_once(demo_index):
    import threading
    client, _ = fake_client([tool_resp(("create_issue", ISSUE)), text_resp("ok")])
    ans = RepoAgent(demo_index, LLM(client=client), confirm=lambda *a: "n").ask("x")
    assert ans.steps[0].status == "rejected"                        # truthy "n" is NOT approval
    client, _ = fake_client([tool_resp(("create_issue", ISSUE)), text_resp("ok")])
    agent = RepoAgent(demo_index, LLM(client=client))
    agent.ask("x")
    threads = [threading.Thread(target=agent.resolve_pending, args=(1, True)) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    drafts = config.WORKSPACE / demo_index.slug / "issue_drafts"
    assert len(list(drafts.glob("*.md"))) == 1                       # double click -> one execution


def test_preview_shows_hidden_control_chars_and_target(demo_index):
    from repo_agent.tools import preview_action, validate_args, TOOL_BY_NAME
    body = "visible \x1b[8mhidden secret\x1b[0m text that is long enough"
    p = preview_action("create_issue", {"title": "t" * 6, "body": body}, demo_index)
    assert "\x1b" not in p and "\\x1b" in p and "Куда:" in p
    assert validate_args(TOOL_BY_NAME["create_issue"]["input_schema"], {"title": "t" * 6, "body": body})
    assert validate_args(TOOL_BY_NAME["create_issue"]["input_schema"],
                         {"title": "t" * 6, "body": "b" * 30, "labels": ["x" * 60]})


def test_malformed_model_output_never_crashes(demo_index):
    from types import SimpleNamespace
    weird = SimpleNamespace(content=[SimpleNamespace(type="tool_use", id=None, name=["x"], input="[" * 5000),
                                     *[SimpleNamespace(type="tool_use", id=f"i{i}", name="list_dir", input={"path": ""})
                                       for i in range(20)]],
                            stop_reason="tool_use", usage=None)
    client, _ = fake_client([weird, text_resp("done")])
    ans = RepoAgent(demo_index, LLM(client=client)).ask("q")
    assert ans.steps[0].status == "unknown_tool"
    assert sum(1 for s in ans.steps if s.status == "ok") == 7            # 8 per step: 1 bad + 7 executed
    assert all(len(s.output_preview) <= 400 for s in ans.steps)

    class Boom:
        def __init__(self):
            self.beta = self
            self.messages = self

        def create(self, **kw):
            raise RuntimeError("unexpected client failure")
    assert RepoAgent(demo_index, LLM(client=Boom())).ask("q").error


def test_hybrid_search_fuses_vector_ranking(demo_index, monkeypatch):
    import numpy as np
    from repo_agent import vectors as vec_mod
    # fake embeddings: the vector ranking puts demo/cli.py first for any query
    fake = np.zeros((len(demo_index.chunks), 4), dtype=np.float32)
    target = [c.id for c in demo_index.chunks if c.path == "demo/cli.py"][0]
    fake[target, 0] = 1.0
    monkeypatch.setattr(vec_mod, "query", lambda vecs, text, k=100: [(target, 1.0)])
    demo_index.vectors = fake
    hits = demo_index.search("zzz unknown words", k=3)
    assert hits and hits[0].chunk.path == "demo/cli.py"   # BM25 finds nothing; vectors still help


def test_pr_review_comments_only_on_diff_lines_and_post_needs_token(demo_index, monkeypatch):
    from repo_agent import github_pr
    diff = ("diff --git a/app/db.py b/app/db.py\n--- a/app/db.py\n+++ b/app/db.py\n@@ -1,2 +1,3 @@\n"
            " import x\n+cur.execute(\"SELECT * FROM t WHERE id=%s\" % uid)\n y = 1\n")
    calls = []

    def fake_req(url, accept="application/vnd.github+json", data=None, token=None):
        calls.append((url, data is not None))
        if accept.endswith("diff"):
            return diff
        if data is not None:
            return {"html_url": "https://github.com/o/r/pull/5#review"}
        return {"title": "t", "body": "b", "head": {"sha": "abc"}, "html_url": "https://github.com/o/r/pull/5"}
    monkeypatch.setattr(github_pr, "_req", fake_req)
    demo_index.meta["web_url"] = "https://github.com/o/r"
    rep = github_pr.review_pr(demo_index, None, 5)
    assert [(c["path"], c["line"]) for c in rep["comments"]] == [("app/db.py", 2)]
    assert not any(post for _, post in calls)                      # reviewing never posts
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with pytest.raises(Exception):
        github_pr.post_review(rep)                                  # no token -> refuse
    assert github_pr.post_review(rep, token="t").endswith("#review")


def test_autofix_proposes_diff_and_applies_on_new_branch_only(demo_index):
    import json as _json
    from repo_agent.autofix import FixError, apply_fix, propose_fix
    finding = {"path": "demo/util.py", "line": 7, "title": "upper bound off by one", "explanation": "x"}
    client, _ = fake_client([text_resp(_json.dumps({"start_line": 7, "end_line": 7,
                                                    "replacement": "    if x > hi:", "explanation": "> вместо >="}))])
    fix = propose_fix(demo_index, LLM(client=client), finding)
    assert "-    if x >= hi:" in fix.diff and "+    if x > hi:" in fix.diff
    before = (demo_index.root / "demo/util.py").read_text()
    head_branch = subprocess.run(["git", "-C", str(demo_index.root), "branch", "--show-current"],
                                 capture_output=True, text=True).stdout.strip()
    res = apply_fix(demo_index, fix, finding["title"])
    assert (demo_index.root / "demo/util.py").read_text() == before            # indexed clone untouched
    assert subprocess.run(["git", "-C", str(demo_index.root), "branch", "--show-current"],
                          capture_output=True, text=True).stdout.strip() == head_branch
    assert "if x > hi:" in (Path(res["worktree"]) / "demo/util.py").read_text()
    client, _ = fake_client([text_resp(_json.dumps({"start_line": 7, "end_line": 7,
                                                    "replacement": "    if x >>> hi:", "explanation": "bad"}))])
    with pytest.raises(FixError):                                               # broken syntax is refused
        propose_fix(demo_index, LLM(client=client), finding)


def test_reviewer_accepts_array_without_path_for_single_file(demo_index):
    import json as _json
    from repo_agent.review.llm_review import LLMReviewer
    client, _ = fake_client([text_resp("```json\n" + _json.dumps([{"line": 7, "title": "bound"}]) + "\n```")])
    out = LLMReviewer(LLM(client=client)).review_items([{"path": "demo/util.py", "kind": "file", "view": "    7 | x"}])
    assert [(f.path, f.line, f.severity, f.category) for f in out] == [("demo/util.py", 7, "medium", "bug")]
