"""Command line interface: `repo-agent <command> <repo> ...` (repo = owner/name, GitHub URL or local path)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .index import RepoIndex
from .llm import LLM, LLMError, has_credentials


def _out_dir(idx: RepoIndex) -> Path:
    d = config.WORKSPACE / idx.slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load(args) -> RepoIndex:
    def progress(i, n, rel):
        print(f"\r  индексирую {i}/{n} {rel[:60]:<60}", end="", file=sys.stderr)

    idx = RepoIndex.load_or_build(args.repo, rebuild=getattr(args, "rebuild", False),
                                  depth=getattr(args, "depth", None), progress=progress)
    print("\r" + " " * 90 + "\r", end="", file=sys.stderr)
    return idx


def _llm(args) -> LLM | None:
    if getattr(args, "no_llm", False):
        return None
    if not has_credentials():
        print(f"⚠ {config.API_KEY_ENV} не задан — работаю без LLM.", file=sys.stderr)
        return None
    return LLM(model=getattr(args, "model", None))


def cmd_index(args):
    idx = _load(args)
    print(json.dumps(idx.meta, ensure_ascii=False, indent=2))


def cmd_summary(args):
    from .summary import build_summary, narrative, summary_markdown

    idx = _load(args)
    s = build_summary(idx)
    md = summary_markdown(s)
    llm = _llm(args) if args.llm else None
    if llm:
        md = "## Кратко\n\n" + narrative(idx, s) + "\n\n" + md
    (_out_dir(idx) / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2, default=str))
    (_out_dir(idx) / "summary.md").write_text(md)
    print(md)


def cmd_map(args):
    from .modmap import module_map

    idx = _load(args)
    m = module_map(idx, level=args.level, depth=args.dir_depth, include_tests=args.tests)
    (_out_dir(idx) / "modules.mmd").write_text(m["mermaid"])
    (_out_dir(idx) / "modules.json").write_text(json.dumps({k: v for k, v in m.items() if k != "mermaid"},
                                                           ensure_ascii=False, indent=2))
    print(m["mermaid"])
    print(f"\n# уровень={m['level']} узлов={len(m['nodes'])} связей={len(m['edges'])} "
          f"взаимных импортов={len(m['cycles'])} -> {_out_dir(idx) / 'modules.mmd'}", file=sys.stderr)


def _print_answer(ans, idx):
    print(ans.text)
    if ans.citations:
        print("\n— проверка ссылок:")
        for c in ans.citations:
            mark = "✓" if c.valid else "✗"
            url = idx.blob_url(c.resolved, c.start, c.end) if c.resolved else None
            print(f"  {mark} {c.label}" + (f"  {url}" if url else ""))


_ICON = {"ok": "·", "invalid_args": "✗ аргументы", "error": "✗ ошибка", "timeout": "⏱ таймаут",
         "pending": "⏸ ждёт подтверждения", "rejected": "⛔ отклонено", "approved": "✔ подтверждено",
         "unknown_tool": "✗ нет инструмента"}


def _on_step(s):
    print(f"  {_ICON.get(s.status, '·')} {s.tool}({json.dumps(s.input, ensure_ascii=False)[:100]})"
          + (f"\n      → {s.output_preview[:160]}" if s.is_error else ""), file=sys.stderr)


def _confirm_tty(tool: str, args: dict, preview: str) -> bool:
    """Human-in-the-loop for actions: the agent waits for an explicit 'y' in the terminal."""
    print(f"\n⚠️  Агент хочет выполнить действие «{tool}»:\n{'-' * 60}\n{preview}\n{'-' * 60}", file=sys.stderr)
    if not sys.stdin.isatty():  # piped input can't be a conscious human decision
        print("Нет интерактивного терминала — действие отклонено.", file=sys.stderr)
        return False
    return input("Выполнить? [y/N] ").strip().lower() in ("y", "yes", "д", "да")


def _agent(args, idx):
    from .agent import RepoAgent
    llm = _llm(args)
    if args.fault and llm is None:
        print("⚠ --fault работает только с Claude (в режиме без LLM инструменты не вызываются). "
              "Без ключа сбой и восстановление показывает: repo-agent demo <repo> --offline", file=sys.stderr)
    return RepoAgent(idx, llm, max_steps=args.max_steps, confirm=_confirm_tty, fault=args.fault)


def cmd_ask(args):
    idx = _load(args)
    _print_answer(_agent(args, idx).ask(args.question, on_step=_on_step), idx)


def cmd_chat(args):
    idx = _load(args)
    agent = _agent(args, idx)
    print(f"Чат по {idx.slug}. Пустая строка или Ctrl-D — выход, /reset — новый диалог.")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        if q == "/reset":
            agent.reset()
            continue
        _print_answer(agent.ask(q, on_step=_on_step), idx)


def cmd_demo(args):
    """Two demo scenarios: a normal question, and the same question with a tool failure the agent recovers from."""
    from .demo import run_demo
    run_demo(args)


def cmd_review(args):
    from .review.llm_review import report_markdown, review_commit, review_repo

    idx = _load(args)
    llm = _llm(args)
    try:
        rep = review_commit(idx, llm, args.commit) if args.commit else review_repo(idx, llm, files=args.files, top=args.top)
    except LLMError as e:
        sys.exit(f"Ошибка: {e}")
    name = f"review_{args.commit[:10]}" if args.commit else "review"
    (_out_dir(idx) / f"{name}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    md = report_markdown(rep, idx, min_severity=args.min_severity)
    (_out_dir(idx) / f"{name}.md").write_text(md)
    print(md)
    if llm:
        print(f"\nusage: {llm.usage}", file=sys.stderr)


def cmd_eval_qa(args):
    from .eval import qa

    idx = _load(args)
    from .eval import datasets

    full_name = idx.slug.replace("__", "/")
    if args.dataset == "swe-qa":
        name = args.split or qa.sweqa_name_for(idx.slug)
        if not name:
            sys.exit(f"Для {idx.slug} нет сплита SWE-QA; укажите --split (одно из: {', '.join(qa.SWEQA_REPOS)}) "
                     f"или другой --dataset")
        items = qa.load_sweqa(name)
    elif args.dataset == "stackrepoqa":
        items = datasets.load_stackrepoqa(full_name)
        if not items:
            sys.exit(f"В StackRepoQA нет вопросов для {full_name}. Репозитории (Java): "
                     f"{', '.join(list(datasets.stackrepoqa_repos())[:20])} …")
    elif args.dataset.startswith("coderepoqa:"):
        items = datasets.load_coderepoqa(args.dataset.split(":", 1)[1], repo=full_name)
    else:
        items = qa.load_jsonl(args.dataset)
    ds_label = (f"swe-qa/{args.split or qa.sweqa_name_for(idx.slug)}" if args.dataset == "swe-qa" else
                "coderepoqa" if args.dataset.startswith("coderepoqa:") else
                args.dataset if args.dataset == "stackrepoqa" else Path(args.dataset).name)
    llm = _llm(args) if args.mode == "agent" else None
    if args.mode == "agent" and llm is None:
        sys.exit(f"Режим agent требует ключ {config.API_KEY_ENV} (или используйте --mode retrieval).")
    res = qa.evaluate(idx, items, mode=args.mode, llm=llm, limit=args.limit, judge=args.judge, dataset=ds_label,
                      progress=lambda i, n, q: print(f"[{i + 1}/{n}] {q[:80]}", file=sys.stderr))
    p = qa.save(res, _out_dir(idx) / "eval")
    print(json.dumps({k: res[k] for k in ("slug", "dataset", "mode", "model", "n_questions", "n_without_gold", "metrics",
                                          "seconds", "usage")}, ensure_ascii=False, indent=2))
    print(f"-> {p}", file=sys.stderr)


def cmd_eval_bugs(args):
    from .eval import bugs

    idx = _load(args)
    llm = _llm(args)
    prog = lambda i, n, t: print(f"[{i + 1}/{n}] {t.op} {t.path}:{t.lines[0]}", file=sys.stderr)
    if args.seeds and args.source == "synthetic":
        seeds = [int(x) for x in args.seeds.split(",")]
        multi = bugs.run_seeds(idx, llm, seeds, source=args.source, n=args.n, tol=args.tol,
                               controls=not args.no_controls, mode=args.mode, progress=prog)
        for r in multi["runs"]:
            bugs.save(r, _out_dir(idx) / "eval")
        print(json.dumps({"seeds": seeds, "pooled": multi["pooled"],
                          "per_seed": [{k: r[k] for k in ("detection_rate", "control_false_alarm_rate", "discrimination")}
                                       for r in multi["runs"]]}, ensure_ascii=False, indent=2))
        return
    res = bugs.run(idx, llm, source=args.source, n=args.n, tol=args.tol, controls=not args.no_controls,
                   seed=args.seed, mode=args.mode, progress=prog)
    p = bugs.save(res, _out_dir(idx) / "eval")
    print(json.dumps({k: v for k, v in res.items() if k != "trials"}, ensure_ascii=False, indent=2))
    print(f"-> {p}", file=sys.stderr)


def cmd_report(args):
    from . import report
    from .modmap import module_map
    from .summary import build_summary, narrative

    idx = _load(args)
    s = build_summary(idx)
    m = module_map(idx)
    rp = _out_dir(idx) / "review.json"
    rev = json.loads(rp.read_text()) if rp.exists() else None
    llm = _llm(args) if args.llm else None
    nar = narrative(idx, s) if llm else None
    p = report.write(idx, report.render(idx, s, m, rev, nar))
    print(p)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="repo-agent", description="Ассистент по репозиторию: индекс, Q&A, карта, ревью.")
    ap.add_argument("--model", default=None, help=f"модель Claude (по умолчанию {config.MODEL})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("repo", help="owner/name, URL GitHub или локальный путь")
        p.add_argument("--rebuild", action="store_true", help="переиндексировать")
        p.add_argument("--depth", type=int, default=None, help="git clone --depth (по умолчанию полная история)")
        p.set_defaults(fn=fn)
        return p

    add("index", cmd_index, "клонировать и проиндексировать")
    p = add("summary", cmd_summary, "автосводка проекта (стек, точки входа)")
    p.add_argument("--llm", action="store_true", help="добавить текстовое описание от Claude")
    p = add("map", cmd_map, "карта модулей (Mermaid)")
    p.add_argument("--level", choices=["auto", "file", "dir"], default="auto")
    p.add_argument("--dir-depth", type=int, default=2)
    p.add_argument("--tests", action="store_true", help="включить тесты")
    for name, fn, h in (("ask", cmd_ask, "один вопрос"), ("chat", cmd_chat, "интерактивный чат")):
        p = add(name, fn, h)
        if name == "ask":
            p.add_argument("question")
        p.add_argument("--no-llm", action="store_true", help="только поиск (без Claude)")
        p.add_argument("--max-steps", type=int, default=None, help=f"лимит шагов (по умолчанию {config.MAX_STEPS})")
        p.add_argument("--fault", default=None, help="симулировать сбой: <инструмент>:<timeout|error>, "
                                                       "например read_file:timeout")
    p = add("demo", cmd_demo, "демо: обычный сценарий и сценарий с ошибкой инструмента")
    p.add_argument("--offline", action="store_true", help="без API: воспроизвести записанный сценарий модели")
    p.add_argument("--fault", default="read_file:timeout")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-llm", action="store_true", help=argparse.SUPPRESS)
    p = add("review", cmd_review, "ревью: баги и безопасность")
    p.add_argument("--commit", help="ревью конкретного коммита (SHA)")
    p.add_argument("--files", nargs="*", help="файлы для LLM-ревью (по умолчанию — самые рискованные)")
    p.add_argument("--top", type=int, default=8, help="сколько файлов отдавать LLM")
    p.add_argument("--min-severity", default="low", choices=["critical", "high", "medium", "low"])
    p.add_argument("--no-llm", action="store_true", help="только статический анализ")
    p = add("eval-qa", cmd_eval_qa, "метрика: доля ответов с верным файлом")
    p.add_argument("--dataset", default="swe-qa",
                   help="swe-qa | stackrepoqa | coderepoqa:<файл.json[l]> | путь к JSONL {question, answer?, gold_files?}")
    p.add_argument("--split", help="сплит SWE-QA (requests, flask, ...)")
    p.add_argument("--mode", choices=["agent", "retrieval"], default="agent")
    p.add_argument("--limit", type=int)
    p.add_argument("--judge", action="store_true", help="LLM-судья сравнивает ответ с эталоном")
    p = add("eval-bugs", cmd_eval_bugs, "метрика: доля найденных внесённых багов")
    p.add_argument("--source", choices=["synthetic", "history"], default="synthetic")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--tol", type=int, default=2, help="допуск по строкам")
    p.add_argument("--mode", choices=["snippet", "diff"], default="snippet")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--seeds", help="несколько выборок мутаций, например 13,1,2,3,4 (среднее и разброс)")
    p.add_argument("--no-controls", action="store_true")
    p.add_argument("--no-llm", action="store_true", help="baseline: только статический анализ")
    p = add("report", cmd_report, "HTML-отчёт (сводка, карта, ревью, метрики)")
    p.add_argument("--llm", action="store_true")

    args = ap.parse_args(argv)
    if config.CONFIG_ERROR:
        sys.exit(f"Ошибка конфигурации: {config.CONFIG_ERROR}")
    if args.model:
        config.MODEL = args.model
    args.fn(args)


if __name__ == "__main__":
    main()
