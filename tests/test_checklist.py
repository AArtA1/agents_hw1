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
    assert real["llm"]["provider"] == "anthropic" and real["llm"]["model"]


def test_no_secrets_in_repository():
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, text=True).stdout.split()
    pat = re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{36,}|AKIA[0-9A-Z]{16}")
    leaks = [f for f in tracked if (ROOT / f).is_file() and pat.search((ROOT / f).read_text(errors="ignore"))]
    assert leaks == []
    assert ".env" in (ROOT / ".gitignore").read_text().split()
