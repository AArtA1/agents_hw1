"""Agent tools: schemas, argument validation, error handling, timeouts, output limits, confirmation.

Every tool call goes through `RepoTools.call()`:
  1. the tool must exist;
  2. arguments are validated against the tool's JSON schema (types, required, ranges, unknown fields) —
     on failure the model gets an error explaining what is wrong and what is expected;
  3. tools of kind "action" are never executed here: the agent asks a human first (see agent.py);
  4. the tool runs; service problems (git/GitHub/regex too slow) become a timeout error for the model,
     never an exception that kills the agent;
  5. the result is truncated to `agent.max_tool_output_chars`.
Session-bound values (which repository, its clone on disk, the GitHub owner/name and token) are taken
from the session (RepoIndex / environment), never from model arguments.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from . import config, repo as repo_mod
from .index import RepoIndex, _norm_rel
from .issues import IssuesError, format_issue, get_issue, owner_name
from .textutil import split_lines

MAX_READ_LINES = 250


class ToolError(ValueError):
    """Expected, explainable failure (bad input, file not found). Goes back to the model."""


class ServiceTimeout(ToolError):
    """An external service (git, GitHub API) or a slow operation exceeded its time budget."""


# ---------------------------------------------------------------------------------------- schemas

def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


PATH = {"type": "string", "minLength": 1, "maxLength": 400,
        "description": "Путь от корня репозитория, например 'src/requests/sessions.py'. Можно указать только "
                       "имя файла, если оно уникально."}

TOOLS: list[dict] = [
    {
        "name": "search_code",
        "description": (
            "Найти, где в репозитории находится код или документация по теме. Используй ПЕРВЫМ, когда не знаешь, "
            "в каком файле искать. Запрос — 2–6 ключевых слов или имён идентификаторов на английском "
            "(например 'retry adapter max_retries'), а не вопрос целиком. Возвращает до k фрагментов: "
            "'путь:начало-конец', имена функций/классов во фрагменте и первые 8 строк. Это не полный код: "
            "чтобы что-то утверждать, затем прочитай строки через read_file. Если ничего не найдено — "
            "попробуй синонимы, find_symbol или grep."),
        "input_schema": _obj({
            "query": {"type": "string", "minLength": 2, "maxLength": 300, "description": "Ключевые слова / идентификаторы"},
            "path_prefix": {"type": "string", "maxLength": 300, "description": "Искать только в каталоге, например 'src/'"},
            "k": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Сколько фрагментов вернуть (по умолчанию 8)"},
        }, ["query"]),
        "spec": {"kind": "read", "service": "локальный индекс"},
    },
    {
        "name": "read_file",
        "description": (
            f"Прочитать строки файла репозитория с номерами строк (не больше {MAX_READ_LINES} строк за вызов). "
            "Используй, чтобы проверить код перед ответом и получить точные номера строк для ссылок "
            "'путь:начало-конец'. Без start_line читает с начала файла. Если файл длиннее, в конце будет "
            "подсказка, с какой строки читать дальше. Ошибка, если файла нет в индексе (найди путь через "
            "search_code или list_dir) или если start_line больше длины файла."),
        "input_schema": _obj({
            "path": PATH,
            "start_line": {"type": "integer", "minimum": 1, "description": "Первая строка (с 1)"},
            "end_line": {"type": "integer", "minimum": 1, "description": "Последняя строка включительно"},
        }, ["path"]),
        "spec": {"kind": "read", "service": "файлы клона"},
    },
    {
        "name": "find_symbol",
        "description": (
            "Найти, где определена функция, класс или метод, по имени: 'merge_setting', 'Session.send', 'HTTPAdapter'. "
            "Возвращает вид символа, полное имя, 'путь:начало-конец' и первую строку докстринга. Точнее, чем "
            "search_code, когда имя известно. Пустой результат — символа с таким именем нет."),
        "input_schema": _obj({"name": {"type": "string", "minLength": 1, "maxLength": 200}}, ["name"]),
        "spec": {"kind": "read", "service": "локальный индекс"},
    },
    {
        "name": "grep",
        "description": (
            "Поиск строк по регулярному выражению Python во всех проиндексированных файлах — для точных строк: "
            "вызовов, констант, сообщений об ошибках ('raise InvalidURL', 'DEFAULT_RETRIES'). Возвращает "
            "'путь:строка: текст', не больше max_results совпадений. Ограничен по времени: на слишком общем "
            "шаблоне вернёт частичный результат."),
        "input_schema": _obj({
            "pattern": {"type": "string", "minLength": 1, "maxLength": 200},
            "path_prefix": {"type": "string", "maxLength": 300},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
        }, ["pattern"]),
        "spec": {"kind": "read", "service": "файлы клона"},
    },
    {
        "name": "list_dir",
        "description": "Показать файлы и подкаталоги каталога репозитория ('' или '.' — корень). Нужен, чтобы "
                       "понять структуру проекта или найти точный путь.",
        "input_schema": _obj({"path": {"type": "string", "maxLength": 400}}, ["path"]),
        "spec": {"kind": "read", "service": "локальный индекс"},
    },
    {
        "name": "git_history",
        "description": (
            "Коммиты репозитория: поиск по тексту сообщений (query), журнал одного файла (path) или оба сразу. "
            "Для вопросов «когда/почему это изменили». Возвращает 'sha дата автор: сообщение [файлы]'. "
            "Подробности коммита — show_commit."),
        "input_schema": _obj({
            "query": {"type": "string", "maxLength": 300}, "path": PATH,
            "limit": {"type": "integer", "minimum": 1, "maximum": 30},
        }, []),
        "spec": {"kind": "read", "service": "git"},
    },
    {
        "name": "show_commit",
        "description": "Сообщение и diff одного коммита по SHA (минимум 4 hex-символа). Diff обрезается.",
        "input_schema": _obj({"sha": {"type": "string", "pattern": "^[0-9a-fA-F]{4,40}$"}}, ["sha"]),
        "spec": {"kind": "read", "service": "git"},
    },
    {
        "name": "blame",
        "description": "git blame для диапазона строк (не больше 80): коммит, автор и дата каждой строки.",
        "input_schema": _obj({"path": PATH, "start_line": {"type": "integer", "minimum": 1},
                              "end_line": {"type": "integer", "minimum": 1}}, ["path", "start_line", "end_line"]),
        "spec": {"kind": "read", "service": "git"},
    },
    {
        "name": "search_issues",
        "description": "Поиск по issues и pull requests репозитория на GitHub (заголовок + описание): известные "
                       "баги, обсуждения, мотивация изменений. Возвращает '#номер [тип, статус] заголовок'.",
        "input_schema": _obj({"query": {"type": "string", "minLength": 2, "maxLength": 300},
                              "k": {"type": "integer", "minimum": 1, "maximum": 20}}, ["query"]),
        "spec": {"kind": "read", "service": "локальный кэш issues"},
    },
    {
        "name": "get_issue",
        "description": "Issue или PR по номеру с описанием и обсуждением (запрос к GitHub API, может не ответить "
                       "вовремя — тогда повтори позже или обойдись без него).",
        "input_schema": _obj({"number": {"type": "integer", "minimum": 1}}, ["number"]),
        "spec": {"kind": "read", "service": "GitHub API"},
    },
    {
        "name": "create_issue",
        "description": (
            "ДЕЙСТВИЕ: подготовить issue о найденной проблеме (баг, уязвимость) в этом репозитории. "
            "Вызывай, только если пользователь прямо попросил оформить issue. Действие НЕ выполняется сразу: "
            "его подтверждает человек. Результат — 'ожидает подтверждения', 'отклонено человеком' или "
            "путь/ссылка созданного issue. Никогда не говори, что issue создан, пока не получил такой результат. "
            "В body укажи файл и строки ('путь:начало-конец'), шаги воспроизведения и ожидаемое поведение."),
        "input_schema": _obj({
            "title": {"type": "string", "minLength": 5, "maxLength": 120},
            "body": {"type": "string", "minLength": 20, "maxLength": 8000},
            "labels": {"type": "array", "items": {"type": "string", "maxLength": 50}, "maxItems": 5},
        }, ["title", "body"]),
        "spec": {"kind": "action", "service": "файл черновика / GitHub API"},
    },
]

TOOL_BY_NAME = {t["name"]: t for t in TOOLS}


def api_tools() -> list[dict]:
    """Tool definitions as sent to the model (without our private 'spec' metadata)."""
    return [{k: v for k, v in t.items() if k != "spec"} for t in TOOLS]


# ------------------------------------------------------------------------------------- validation

_TYPES = {"string": str, "integer": int, "boolean": bool, "array": list, "object": dict}


def validate_args(schema: dict, args) -> list[str]:
    """Minimal JSON-schema check for our tool schemas. Returns human-readable problems (empty = valid)."""
    if not isinstance(args, dict):
        return [f"аргументы должны быть объектом JSON, получено {type(args).__name__}"]
    errs = []
    props = schema.get("properties", {})
    for r in schema.get("required", []):
        if r not in args:
            errs.append(f"не хватает обязательного поля '{r}'")
    if schema.get("additionalProperties") is False:
        for k in args:
            if k not in props:
                errs.append(f"неизвестное поле '{k}' (допустимы: {', '.join(props)})")
    for k, v in args.items():
        p = props.get(k)
        if not p:
            continue
        t = p.get("type")
        ok = isinstance(v, _TYPES[t]) and not (t == "integer" and isinstance(v, bool))
        if not ok:
            errs.append(f"поле '{k}' должно быть {t}, получено {type(v).__name__} {json.dumps(v, ensure_ascii=False)[:40]}")
            continue
        if t == "integer":
            if "minimum" in p and v < p["minimum"]:
                errs.append(f"поле '{k}' должно быть >= {p['minimum']}, получено {v}")
            if "maximum" in p and v > p["maximum"]:
                errs.append(f"поле '{k}' должно быть <= {p['maximum']}, получено {v}")
        if t == "string":
            if len(v) < p.get("minLength", 0):
                errs.append(f"поле '{k}' короче {p['minLength']} символов")
            if "maxLength" in p and len(v) > p["maxLength"]:
                errs.append(f"поле '{k}' длиннее {p['maxLength']} символов")
            if "pattern" in p and not re.fullmatch(p["pattern"], v):
                errs.append(f"поле '{k}' не соответствует формату {p['pattern']}")
        if t == "array":
            if "maxItems" in p and len(v) > p["maxItems"]:
                errs.append(f"в поле '{k}' больше {p['maxItems']} элементов")
            it = p.get("items", {}).get("type")
            if it and not all(isinstance(x, _TYPES[it]) for x in v):
                errs.append(f"элементы поля '{k}' должны быть {it}")
    return errs


# ------------------------------------------------------------------------------------- results

@dataclass
class ToolResult:
    content: str
    status: str = "ok"            # ok | invalid_args | error | timeout | pending | rejected | unknown_tool
    files: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def is_error(self) -> bool:
        return self.status in ("invalid_args", "error", "timeout", "unknown_tool")


def truncate(text: str, limit: int | None = None) -> tuple[str, bool]:
    limit = limit or config.MAX_TOOL_OUTPUT
    if len(text) <= limit:
        return text, False
    return text[:limit] + f"\n... [результат обрезан: показано {limit} из {len(text)} символов; сузь запрос]", True


class RepoTools:
    """Tool implementations bound to ONE session: the repository is fixed here, not chosen by the model."""

    def __init__(self, idx: RepoIndex, fault: str | None = None):
        self.idx = idx
        self._fault = (fault if fault is not None else config.DEMO_FAULT) or ""
        self._fault_used = False

    # ------------------------------------------------------------------ entry point used by the agent
    def call(self, name: str, args) -> ToolResult:
        spec = TOOL_BY_NAME.get(name)
        if spec is None:
            return ToolResult(f"Ошибка: инструмента '{name}' нет. Доступны: {', '.join(TOOL_BY_NAME)}.", "unknown_tool")
        problems = validate_args(spec["input_schema"], args)
        if problems:
            schema = json.dumps(spec["input_schema"]["properties"], ensure_ascii=False)[:600]
            return ToolResult(f"Ошибка аргументов {name}: " + "; ".join(problems) +
                              f". Исправь аргументы и вызови снова. Ожидаемые поля: {schema}", "invalid_args")
        try:
            self._maybe_fault(name)
            out, files = self.run(name, args)
        except ServiceTimeout as e:
            return ToolResult(f"Таймаут: {e}. Сервис не ответил вовремя — повтори вызов один раз, сузь запрос "
                              f"или продолжи без этого инструмента и скажи пользователю, чего не удалось проверить.",
                              "timeout")
        except ToolError as e:
            return ToolResult(f"Ошибка: {e}", "error")
        except Exception as e:  # unexpected bug inside a tool must not kill the agent
            return ToolResult(f"Внутренняя ошибка инструмента {name}: {type(e).__name__}: {e}", "error")
        text, cut = truncate(out)
        return ToolResult(text, "ok", files, cut)

    def _maybe_fault(self, name: str) -> None:
        """Demo/test hook: make the first call of one tool fail (config demo.fault = 'tool:timeout|error')."""
        if not self._fault or self._fault_used:
            return
        tool, _, kind = self._fault.partition(":")
        if tool == name:
            self._fault_used = True
            if kind == "error":
                raise ToolError("симулированный сбой инструмента (demo.fault)")
            raise ServiceTimeout(f"{name}: симулированный таймаут сервиса ({config.TOOL_TIMEOUT:.0f} с, demo.fault)")

    def run(self, name: str, args: dict) -> tuple[str, list[str]]:
        """Execute without validation (tests / internal use). Raises ToolError on bad input."""
        if TOOL_BY_NAME.get(name, {}).get("spec", {}).get("kind") == "action":
            raise ToolError("действия выполняются только через подтверждение человеком")
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            raise ToolError(f"unknown tool {name}")
        return fn(**args)

    def _path(self, p: str) -> str:
        r = self.idx.resolve_path(p)
        if not r:
            raise ToolError(f"файл '{p}' не найден в индексе. Найди точный путь через search_code или list_dir.")
        return r

    # ------------------------------------------------------------------ read tools
    def t_search_code(self, query: str, path_prefix: str | None = None, k: int = 8):
        hits = self.idx.search(query, k=k, path_prefix=path_prefix or None)
        if not hits:
            return "Ничего не найдено. Попробуй другие ключевые слова, find_symbol или grep.", []
        lines = []
        for h in hits:
            c = h.chunk
            preview = "\n".join(f"    {l[:config.MAX_LINE_CHARS]}" for l in split_lines(c.text)[:8])
            syms = f" symbols: {', '.join(c.symbols[:6])}" if c.symbols else ""
            lines.append(f"{c.path}:{c.start}-{c.end} (score {h.score:.1f}){syms}\n{preview}")
        return "\n\n".join(lines), []

    def t_read_file(self, path: str, start_line: int | None = None, end_line: int | None = None):
        rel = self._path(path)
        start = start_line or 1
        end = end_line or start + MAX_READ_LINES - 1
        if end < start:
            raise ToolError(f"end_line ({end}) меньше start_line ({start}).")
        end = min(end, start + MAX_READ_LINES - 1)
        lines, total = self.idx.read_lines(rel, start, end)
        if start > total:
            raise ToolError(f"в файле {rel} всего {total} строк, start_line={start} за пределами файла.")
        cap = config.MAX_LINE_CHARS

        def fmt(i, l):
            return f"{i:>5} | {l[:cap]}" + (f" …[+{len(l) - cap} симв.]" if len(l) > cap else "")
        body = "\n".join(fmt(i, l) for i, l in enumerate(lines, start))
        last = start + len(lines) - 1
        tail = f"\n... (в файле {total} строк; продолжение: start_line={last + 1})" if last < total else ""
        return f"{rel} (строки {start}-{last} из {total})\n{body}{tail}", [rel]

    def t_find_symbol(self, name: str):
        syms = self.idx.find_symbol(name)[:15]
        if not syms:
            return f"Символ {name!r} не найден. Попробуй search_code или grep.", []
        return "\n".join(f"{s.kind} {s.qualname}: {s.path}:{s.line}-{s.end_line}" + (f" — {s.doc}" if s.doc else "")
                         for s in syms), []

    def t_grep(self, pattern: str, path_prefix: str | None = None, max_results: int = 40):
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"некорректное регулярное выражение: {e}")
        prefix = _norm_rel(path_prefix) if path_prefix else None
        deadline = time.monotonic() + config.TOOL_TIMEOUT
        out = []
        for f in self.idx.files:
            if prefix and not f.startswith(prefix):
                continue
            if time.monotonic() > deadline:
                if not out:
                    raise ServiceTimeout(f"grep: поиск не уложился в {config.TOOL_TIMEOUT:.0f} с")
                return "\n".join(out) + "\n... (поиск прерван по времени — результат неполный)", []
            for i, line in enumerate(split_lines(repo_mod.read_text(self.idx.root, f) or ""), 1):
                if rx.search(line):
                    out.append(f"{f}:{i}: {line.strip()[:200]}")
                    if len(out) >= max_results:
                        return "\n".join(out) + "\n... (есть ещё совпадения)", []
        return ("\n".join(out) or "Совпадений нет."), []

    def t_list_dir(self, path: str = ""):
        prefix = _norm_rel(path or "").strip("/")
        prefix = prefix + "/" if prefix else ""
        dirs, files = set(), []
        for f in self.idx.files:
            if f.startswith(prefix):
                rest = f[len(prefix):]
                (dirs.add(rest.split("/", 1)[0] + "/") if "/" in rest else files.append(rest))
        if not dirs and not files:
            return f"Каталог {path!r} пуст или не проиндексирован.", []
        return "\n".join(sorted(dirs) + sorted(files)), []

    def t_git_history(self, query: str | None = None, path: str | None = None, limit: int = 10):
        try:
            if path:
                commits = repo_mod.commits(self.idx.root, limit=limit, path=self._path(path),
                                           timeout=config.TOOL_TIMEOUT)
                if query:
                    q = query.lower()
                    commits = [c for c in commits if q in f"{c.subject} {c.body}".lower()] or commits
            elif query:
                commits = self.idx.search_commits(query, k=limit)
            else:
                commits = self.idx.commits[:limit]
        except repo_mod.GitTimeout as e:
            raise ServiceTimeout(str(e))
        if not commits:
            return "Коммиты не найдены (возможно, неглубокий клон).", []
        return "\n".join(f"{c.sha[:10]} {c.date} {c.author}: {c.subject}  [{', '.join(c.files[:5])}]"
                         for c in commits), []

    def t_show_commit(self, sha: str):
        try:
            out = repo_mod.show_commit(self.idx.root, sha.strip(), max_chars=config.MAX_TOOL_OUTPUT,
                                       timeout=config.TOOL_TIMEOUT)
        except repo_mod.GitTimeout as e:
            raise ServiceTimeout(str(e))
        if not out:
            raise ToolError(f"коммит {sha} не найден")
        return out, []

    def t_blame(self, path: str, start_line: int, end_line: int):
        rel = self._path(path)
        if end_line < start_line:
            raise ToolError(f"некорректный диапазон строк {start_line}-{end_line}.")
        try:
            out = repo_mod.blame(self.idx.root, rel, start_line, min(end_line, start_line + 80),
                                 timeout=config.TOOL_TIMEOUT)
        except repo_mod.GitTimeout as e:
            raise ServiceTimeout(str(e))
        if out.startswith("Ошибка"):
            raise ToolError(out)
        return out, [rel]

    def t_search_issues(self, query: str, k: int = 8):
        if not self.idx.issues:
            return ("Issues не загружены (репозиторий не на GitHub, нет сети или исчерпан лимит API). "
                    "Можно запросить конкретный номер через get_issue."), []
        hits = self.idx.search_issues(query, k=k)
        if not hits:
            return "Ничего не найдено.", []
        return "\n".join(f"#{i['number']} [{'PR' if i['is_pr'] else 'issue'}, {i['state']}] {i['title']}"
                         f"{'  labels: ' + ', '.join(i['labels']) if i['labels'] else ''}"
                         f"\n    {re.sub(r'\s+', ' ', i['body'])[:200]}" for i in hits), []

    def t_get_issue(self, number: int):
        try:
            return format_issue(get_issue(self.idx.slug, self.idx.meta.get("web_url"), number,
                                          timeout=config.TOOL_TIMEOUT)), []
        except IssuesError as e:
            if "недоступен" in str(e) or "timed out" in str(e).lower():
                raise ServiceTimeout(f"GitHub API: {e}")
            raise ToolError(str(e))


# ------------------------------------------------------------------------------------- actions

def preview_action(name: str, args: dict) -> str:
    if name == "create_issue":
        labels = f"\nМетки: {', '.join(args.get('labels') or [])}" if args.get("labels") else ""
        return f"Создать issue: «{args['title']}»{labels}\n\n{args['body']}"
    return json.dumps(args, ensure_ascii=False, indent=2)


def execute_action(idx: RepoIndex, name: str, args: dict) -> ToolResult:
    """Run a side-effecting tool. Called ONLY after a human approved it."""
    if name != "create_issue":
        return ToolResult(f"Ошибка: неизвестное действие {name}", "error")
    drafts = config.WORKSPACE / idx.slug / "issue_drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w-]+", "-", args["title"].lower()).strip("-")[:40] or "issue"
    path = drafts / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}.md"
    labels = args.get("labels") or []
    path.write_text(f"# {args['title']}\n\n{args['body']}\n" + (f"\nLabels: {', '.join(labels)}\n" if labels else ""))
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    on = owner_name(idx.meta.get("web_url"))
    if config.POST_ISSUES_TO_GITHUB and token and on:
        from .issues import create_issue
        try:
            url = create_issue(on[0], on[1], args["title"], args["body"], labels, token, timeout=config.TOOL_TIMEOUT)
            return ToolResult(f"Issue создан на GitHub: {url} (черновик: {path})")
        except IssuesError as e:
            return ToolResult(f"Не удалось создать issue на GitHub ({e}); черновик сохранён: {path}", "error")
    return ToolResult(f"Черновик issue сохранён: {path} (публикация на GitHub выключена в конфиге "
                      f"actions.post_to_github)")
