"""Live demo: (1) a normal question, (2) the same question when a tool fails — the agent recovers.

`repo-agent demo psf/requests` uses the provider from repo_agent.toml (needs its API key).
`repo-agent demo psf/requests --offline` replays a scripted model policy instead of Claude: the agent loop,
schema validation, error handling, timeouts and output limits are the real ones; only the model's decisions
are scripted (useful when the network/API is unavailable during the demo).
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from . import config
from .agent import RepoAgent
from .index import RepoIndex
from .llm import LLM, has_credentials

QUESTION = "Где задаётся число повторов по умолчанию у HTTPAdapter и как оно используется?"


class ScriptedModel:
    """Offline stand-in for Claude: search -> read (with a deliberate arg mistake in scenario 2) -> answer."""

    def __init__(self, query: str, bad_args: bool):
        self.query, self.bad_args = query, bad_args
        self.beta = SimpleNamespace(messages=self)
        self.n = 0

    @staticmethod
    def _resp(blocks, stop):
        return SimpleNamespace(content=blocks, stop_reason=stop,
                               usage=SimpleNamespace(input_tokens=0, output_tokens=0, cache_read_input_tokens=0))

    def _tool(self, name, args):
        self.n += 1
        return self._resp([SimpleNamespace(type="text", text=""),
                           SimpleNamespace(type="tool_use", id=f"t{self.n}", name=name, input=args)], "tool_use")

    def create(self, **kw):
        last = kw["messages"][-1]["content"]
        results = [b for b in last if isinstance(b, dict) and b.get("type") == "tool_result"] \
            if isinstance(last, list) else []
        if not results:                                   # phase 1: find where the code is
            self.phase = "search"
            return self._tool("search_code", {"query": self.query, "k": 3})
        res = results[0]
        if self.phase == "search":                        # phase 2: read the top hit
            first = str(res["content"]).split("\n", 1)[0].split(" ", 1)[0]
            path, rng = first.rsplit(":", 1)
            a, b = (int(x) for x in rng.split("-"))
            self._loc, self.phase = (path, a, b), "read"
            if self.bad_args:  # model mistake: line numbers as strings -> schema validation error
                return self._tool("read_file", {"path": path, "start_line": str(a), "end_line": str(b)})
            return self._tool("read_file", {"path": path, "start_line": a, "end_line": b})
        path, a, b = self._loc
        if res.get("is_error"):                           # phase 3: recover (fix args / retry after timeout)
            return self._tool("read_file", {"path": path, "start_line": a, "end_line": b})
        answer = (f"Значение по умолчанию задаётся в `{path}:{a}-{b}` (см. прочитанный фрагмент).\n"
                  f"Источники: {path}:{a}-{b}")
        return self._resp([SimpleNamespace(type="text", text=answer)], "end_turn")


def _print_step(s):
    icon = {"ok": "✓", "timeout": "⏱", "invalid_args": "✗", "error": "✗"}.get(s.status, "·")
    print(f"   {icon} {s.tool}({json.dumps(s.input, ensure_ascii=False)[:90]}) [{s.status}]")
    if s.is_error:
        print(f"       ↳ {s.output_preview[:170]}")


def _scenario(title, idx, llm, fault, question):
    print(f"\n=== {title} ===")
    print(f"Вопрос: {question}" + (f"\n(симулирован сбой: {fault})" if fault else ""))
    agent = RepoAgent(idx, llm, fault=fault or "", max_steps=_MAX_STEPS or config.MAX_STEPS)
    ans = agent.ask(question, on_step=_print_step)
    print(f"\nОтвет:\n{ans.text}")
    bad = [c.label for c in ans.citations if not c.valid]
    print(f"\nСсылки: {len(ans.citations)}, невалидных: {len(bad)}; вызовов инструментов: {len(ans.steps)}; "
          f"лимит шагов цикла: {agent.max_steps}")


_MAX_STEPS: int | None = None


def run_demo(args) -> None:
    global _MAX_STEPS
    _MAX_STEPS = getattr(args, "max_steps", None)
    idx = RepoIndex.load_or_build(args.repo, depth=getattr(args, "depth", None))
    if args.offline:
        q = "DEFAULT_RETRIES max_retries HTTPAdapter"
        print("Режим --offline: решения модели воспроизводятся скриптом; цикл агента, валидация аргументов, "
              "таймауты и лимиты — настоящие.")
        _scenario("Сценарий 1: обычный", idx, LLM(client=ScriptedModel(q, False)), "", QUESTION)
        _scenario("Сценарий 2а: ошибка аргументов → модель исправляет", idx,
                  LLM(client=ScriptedModel(q, True)), "", QUESTION)
        _scenario("Сценарий 2б: таймаут инструмента → повтор", idx,
                  LLM(client=ScriptedModel(q, False)), args.fault, QUESTION)
        return
    if not has_credentials():
        sys.exit(f"Нужен ключ провайдера {config.PROVIDER}: переменная {config.API_KEY_ENV} "
                 f"(или запустите с --offline).")
    _scenario("Сценарий 1: обычный", idx, LLM(), "", QUESTION)
    _scenario("Сценарий 2: инструмент вернул ошибку", idx, LLM(), args.fault, QUESTION)
