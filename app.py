"""Streamlit UI: `uv run streamlit run app.py`."""

from __future__ import annotations

import html
import json

import streamlit as st

from repo_agent import config
from repo_agent.agent import RepoAgent
from repo_agent.citations import linkify
from repo_agent.index import RepoIndex
from repo_agent.llm import LLM, LLMError, has_credentials
from repo_agent.modmap import module_map
from repo_agent.report import MERMAID_CDN, latest_eval
from repo_agent.summary import build_summary, narrative, summary_markdown

st.set_page_config(page_title="Repo Agent", page_icon="🧭", layout="wide")


# ------------------------------------------------------------------ caching
@st.cache_resource(show_spinner=False)
def get_index(slug: str) -> RepoIndex:
    return RepoIndex.load(slug)


@st.cache_data(show_spinner=False)
def get_summary(slug: str, head: str) -> dict:
    return build_summary(get_index(slug))


@st.cache_data(show_spinner=False)
def get_map(slug: str, head: str, level: str, depth: int, tests: bool) -> dict:
    return module_map(get_index(slug), level=level, depth=depth, include_tests=tests)


STEP_ICON = {"ok": "🔧", "invalid_args": "❌", "error": "⚠️", "timeout": "⏱️", "pending": "⏸️",
             "rejected": "⛔", "approved": "✅", "unknown_tool": "❌"}


def pct(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def ci(c) -> str:
    return f"{c[0] * 100:.0f}–{c[1] * 100:.0f}%" if c else "—"


def lang_of(path: str) -> str:
    return {"python": "python", "javascript": "javascript", "typescript": "typescript", "go": "go",
            "rust": "rust", "java": "java", "markdown": "markdown", "yaml": "yaml", "toml": "toml",
            "json": "json", "shell": "bash"}.get(config.language_of(path) or "", "text")


def show_code(idx: RepoIndex, path: str, start: int | None, end: int | None, pad: int = 3, max_lines: int = 80):
    """Code viewer with real line numbers; cited lines are marked with ▶."""
    start = start or 1
    end = end or start
    a = max(1, start - pad)
    lines, total = idx.read_lines(path, a, min(end + pad, a + max_lines - 1))
    body = "\n".join(f"{'▶' if start <= i <= end else ' '}{i:>5}  {l}" for i, l in enumerate(lines, a))
    st.code(body, language=lang_of(path))
    url = idx.blob_url(path, start, end)
    if url:
        st.caption(f"[Открыть на GitHub]({url}) · файл: {total} строк")


def mermaid(code: str, height: int = 650):
    dark = st.get_option("theme.base") == "dark"
    # mermaid source is html-escaped (file names come from the repository)
    st.iframe(f"""
<div id="wrap" style="overflow:auto;height:{height - 20}px;background:{'#0e1117' if dark else '#fff'}">
<pre class="mermaid">{html.escape(code)}</pre></div>
<script src="{MERMAID_CDN}"></script>
<script>
mermaid.initialize({{startOnLoad:false,theme:'{'dark' if dark else 'default'}',maxTextSize:200000,
flowchart:{{useMaxWidth:true}}}});
// tabs start hidden (zero-size iframe): render only once the frame is actually visible
(function wait(){{ if (window.innerWidth > 0 && window.innerHeight > 0) mermaid.run();
  else setTimeout(wait, 200); }})();
</script>""", height=height)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🧭 Repo Agent")
    spec = st.text_input("Репозиторий", placeholder="owner/name, URL GitHub или путь", value="psf/requests")
    c1, c2 = st.columns(2)
    do_index = c1.button("Индексировать", type="primary", width="stretch")
    rebuild = c2.checkbox("заново", help="Переклонировать индекс (репозиторий не перекачивается)")
    if do_index and spec.strip():
        bar = st.progress(0.0, text="Клонирую…")
        try:
            idx_new = RepoIndex.load_or_build(spec.strip(), rebuild=rebuild,
                                              progress=lambda i, n, rel: bar.progress(min(i / max(n, 1), 1.0), text=rel))
            get_index.clear()
            st.session_state["slug"] = idx_new.slug
            bar.progress(1.0, text=f"Готово: {idx_new.meta['n_files']} файлов, {idx_new.meta['n_chunks']} фрагментов")
        except Exception as e:  # show any clone/index error to the user
            bar.empty()
            st.error(str(e))

    indexed = RepoIndex.list_indexed()
    if not indexed:
        st.info("Укажите публичный репозиторий и нажмите «Индексировать».")
        st.stop()
    cur = st.session_state.get("slug")
    slug = st.selectbox("Проиндексированные", indexed, index=indexed.index(cur) if cur in indexed else 0)
    st.session_state["slug"] = slug

    st.divider()
    creds = has_credentials()
    model = st.text_input("Модель Claude", value=config.MODEL)
    use_llm = st.toggle("Использовать Claude", value=creds, disabled=not creds,
                        help=None if creds else "Задайте ANTHROPIC_API_KEY, чтобы включить LLM")
    if not creds:
        st.caption("⚠ ANTHROPIC_API_KEY не найден: чат работает в режиме поиска, ревью — только статический анализ.")
    st.caption(f"Провайдер: {config.PROVIDER} · лимит шагов: {config.MAX_STEPS} · "
               f"таймаут инструментов: {config.TOOL_TIMEOUT:.0f} с (repo_agent.toml)")
    fault = st.selectbox("Демо: симулировать сбой инструмента", ["нет", "read_file:timeout", "search_code:timeout",
                                                                 "get_issue:timeout", "read_file:error"],
                         help="Первый вызов выбранного инструмента вернёт ошибку — агент должен с ней справиться")
    fault = "" if fault == "нет" else fault

idx = get_index(slug)
head = idx.meta.get("head", "")
llm = LLM(model=model) if use_llm else None

st.title(slug.replace("__", "/"))
m = idx.meta
st.caption(f"commit `{head[:10]}` · {m['n_files']} файлов · {m['n_chunks']} фрагментов · "
           f"{m['n_symbols']} символов · {m['n_commits']} коммитов · индекс от {m['built_at']}")

tab_chat, tab_sum, tab_map, tab_rev, tab_eval = st.tabs(["💬 Чат", "📋 Сводка", "🗺️ Карта модулей", "🔍 Ревью", "📊 Метрики"])

# ------------------------------------------------------------------ chat
with tab_chat:
    agents = st.session_state.setdefault("agents", {})
    chats = st.session_state.setdefault("chats", {})
    key = (slug, bool(llm), model, fault)
    if key not in agents:
        agents[key] = RepoAgent(idx, llm, summary_md=summary_markdown(get_summary(slug, head)), fault=fault)
    agent = agents[key]
    log = chats.setdefault(key, [])

    top = st.columns([6, 1])
    top[0].caption("Ответы со ссылками `path:строки` — каждая ссылка проверяется по индексу (✓ / ✗).")
    if top[1].button("Новый диалог"):
        agent.reset()
        log.clear()
        st.rerun()

    def render_answer(item):
        st.markdown(linkify(item["text"], idx))
        if item["citations"]:
            with st.expander(f"Ссылки на код ({len(item['citations'])})"):
                for c in item["citations"]:
                    st.markdown(f"{'✓' if c['valid'] else '✗'} **{c['label']}**")
                    if c["valid"] and c["resolved"]:
                        show_code(idx, c["resolved"], c["start"], c["end"])
        if item["steps"]:
            n_err = sum(1 for s in item["steps"] if s["is_error"])
            with st.expander(f"Шаги агента ({len(item['steps'])}" + (f", ошибок инструментов: {n_err}" if n_err else "")
                             + ")", expanded=bool(n_err)):
                for s in item["steps"]:
                    st.markdown(f"{STEP_ICON.get(s.get('status', 'ok'), '🔧')} `{s['tool']}` "
                                f"`{json.dumps(s['input'], ensure_ascii=False)[:200]}` — {s.get('status', 'ok')}")
                    if s["is_error"]:
                        st.caption(s["output_preview"][:300])

    for item in log:
        with st.chat_message(item["role"]):
            if item["role"] == "assistant":
                render_answer(item)
            else:
                st.markdown(item["text"])

    if not log:
        st.markdown("**Примеры вопросов:**")
        for ex in ("Какие точки входа у проекта и что происходит при запуске?",
                   "Где обрабатываются ошибки сети и как работают повторы?",
                   "Какие последние изменения были в самом часто меняемом файле и зачем?"):
            st.markdown(f"- {ex}")

    # human-in-the-loop: actions proposed by the agent run only after an explicit click
    for act in agent.pending:
        if act.status != "pending":
            continue
        with st.container(border=True):
            st.warning(f"Агент предлагает действие #{act.id} «{act.tool}» — выполнится только после подтверждения.")
            st.code(act.preview, language="markdown")
            c1, c2 = st.columns(2)
            if c1.button("Подтвердить", key=f"ok_{act.id}", type="primary"):
                log.append({"role": "assistant", "text": f"✔ Действие #{act.id} подтверждено: "
                            + agent.resolve_pending(act.id, True), "citations": [], "steps": []})
                st.rerun()
            if c2.button("Отклонить", key=f"no_{act.id}"):
                log.append({"role": "assistant", "text": f"⛔ Действие #{act.id}: "
                            + agent.resolve_pending(act.id, False), "citations": [], "steps": []})
                st.rerun()

    q = st.chat_input("Вопрос о коде…")
    if q:
        log.append({"role": "user", "text": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            status = st.status("Ищу в коде…", expanded=False) if llm else None
            try:
                ans = agent.ask(q, on_step=(lambda s: status.write(
                    f"{STEP_ICON.get(s.status, '🔧')} `{s.tool}` {json.dumps(s.input, ensure_ascii=False)[:120]}"
                    + (f" — {s.status}: {s.output_preview[:150]}" if s.is_error else ""))) if status else None)
                if status:
                    status.update(label=f"Готово: {len(ans.steps)} шагов" if not ans.error else "Модель недоступна",
                                  state="complete" if not ans.error else "error")
                item = {"role": "assistant", "text": ans.text,
                        "citations": [{"label": c.label, "valid": c.valid, "resolved": c.resolved,
                                       "start": c.start, "end": c.end} for c in ans.citations],
                        "steps": [s.__dict__ for s in ans.steps]}
                log.append(item)
                render_answer(item)
                if any(a.status == "pending" for a in agent.pending):
                    st.rerun()
            except LLMError as e:
                if status:
                    status.update(label="Ошибка", state="error")
                st.error(str(e))

# ------------------------------------------------------------------ summary
with tab_sum:
    s = get_summary(slug, head)
    if llm and st.button("Сгенерировать описание (Claude)"):
        with st.spinner("Claude пишет сводку…"):
            try:
                st.session_state[f"narr_{slug}"] = narrative(idx, s)
            except LLMError as e:
                st.error(str(e))
    if st.session_state.get(f"narr_{slug}"):
        st.info(st.session_state[f"narr_{slug}"])
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(linkify(summary_markdown(s), idx))
    with col2:
        st.subheader("Языки")
        st.dataframe([{"язык": l["language"], "файлов": l["files"], "строк": l["lines"],
                       "доля": f"{l['share'] * 100:.0f}%"} for l in s["languages"]], hide_index=True)
        st.subheader("Точки входа — код")
        for e in s["entry_points"][:8]:
            with st.expander(f"{e['path']}:{e['line']} — {e['kind']}"):
                show_code(idx, e["path"], e["line"], e["line"], pad=4)
        with st.expander("Зависимости"):
            st.write(", ".join(s["dependencies"]) or "—")

# ------------------------------------------------------------------ module map
with tab_map:
    c1, c2, c3 = st.columns(3)
    level = c1.selectbox("Уровень", ["auto", "file", "dir"], help="auto: файлы, если их немного, иначе каталоги")
    depth = c2.slider("Глубина каталогов", 1, 4, 2, disabled=level == "file")
    tests = c3.checkbox("Включая тесты, docs и примеры")
    mm = get_map(slug, head, level, depth, tests)
    st.caption(f"Уровень: **{mm['level']}** · узлов {len(mm['nodes'])} · связей {len(mm['edges'])} · "
               f"импортов между файлами {mm['file_edges']}")
    if mm["edges"]:
        mermaid(mm["mermaid"])
    else:
        st.info("Внутренних импортов не найдено (поддерживаются Python, JS/TS, Go).")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Хабы")
        st.dataframe(sorted(({"модуль": n["id"], "входящих": n["fan_in"], "исходящих": n["fan_out"]}
                             for n in mm["nodes"]), key=lambda r: -(r["входящих"] + r["исходящих"]))[:20],
                     hide_index=True)
    with c2:
        st.subheader("Взаимные импорты")
        st.write("\n".join(f"- `{a}` ⇄ `{b}`" for a, b in mm["cycles"][:20]) or "нет")
        st.download_button("Скачать Mermaid (.mmd)", mm["mermaid"], file_name=f"{slug}_modules.mmd")

# ------------------------------------------------------------------ review
with tab_rev:
    from repo_agent.review.llm_review import report_markdown, review_commit, review_repo, select_files
    from repo_agent.review.static import check_repo

    scope = st.radio("Что проверять", ["Репозиторий (самые рискованные файлы)", "Выбранные файлы", "Коммит"],
                     horizontal=True)
    files_sel, commit_sel, top_n = None, None, 6
    if scope.startswith("Репозиторий"):
        top_n = st.slider("Файлов для LLM-ревью", 1, 20, 6)
        st.caption("Кандидаты: " + ", ".join(f"`{f}`" for f in select_files(idx, check_repo(idx), top=top_n)))
    elif scope == "Выбранные файлы":
        files_sel = st.multiselect("Файлы", [f for f in idx.files if config.is_code(f)])
    else:
        opts = {f"{c.sha[:10]} {c.date} — {c.subject[:80]}": c.sha for c in idx.commits[:200]}
        commit_sel = opts.get(st.selectbox("Коммит", list(opts))) if opts else None
    st.caption("С Claude: статический анализ + LLM-ревью. Без Claude: только статические правила."
               if llm else "Claude выключен: только статические правила.")
    if st.button("Запустить ревью", type="primary"):
        with st.spinner("Ревью…"):
            try:
                if commit_sel:
                    rep = review_commit(idx, llm, commit_sel)
                else:
                    rep = review_repo(idx, llm, files=files_sel or None, top=top_n)
                st.session_state[f"review_{slug}"] = rep
                out = config.WORKSPACE / slug
                out.mkdir(parents=True, exist_ok=True)
                (out / "review.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
            except LLMError as e:
                st.error(str(e))
    rep = st.session_state.get(f"review_{slug}")
    if rep:
        c = rep["counts"]
        cols = st.columns(5)
        cols[0].metric("Всего", c["total"])
        for col, sev in zip(cols[1:], ["critical", "high", "medium", "low"]):
            col.metric(sev, c["by_severity"].get(sev, 0))
        sev_f = st.multiselect("Важность", ["critical", "high", "medium", "low"], default=["critical", "high", "medium"])
        hide_tests = st.checkbox("Скрыть замечания в тестах", value=True)
        icon = {"critical": "🟥", "high": "🟧", "medium": "🟨", "low": "⬜"}
        shown = [f for f in rep["findings"] if f["severity"] in sev_f and
                 not (hide_tests and ("/test" in f["path"] or f["path"].startswith("test")))]
        if not shown:
            st.success("Нет замечаний выбранной важности.")
        for f in shown[:150]:
            with st.expander(f"{icon[f['severity']]} [{f['category']}] {f['title']} — {f['path']}:{f['line']}"):
                st.markdown(f"{f['explanation']}")
                if f.get("suggestion"):
                    st.markdown(f"**Исправление:** {f['suggestion']}")
                st.caption(f"источник: {f['source']} {f['rule'] if f['source'] == 'static' else ''} · "
                           f"уверенность {f['confidence']:.2f}")
                if idx.has_file(f["path"]):
                    show_code(idx, f["path"], f["line"], f.get("end_line") or f["line"])
        st.download_button("Скачать отчёт (.md)", report_markdown(rep, idx), file_name=f"{slug}_review.md")

# ------------------------------------------------------------------ metrics
with tab_eval:
    from repo_agent.eval import bugs as bugs_eval
    from repo_agent.eval import qa as qa_eval

    st.markdown("Метрики считаются CLI-командами (`repo-agent eval-qa`, `repo-agent eval-bugs`) или здесь. "
                "Прогоны с Claude тратят токены: начните с небольшого лимита.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Доля ответов с верным файлом")
        split = qa_eval.sweqa_name_for(slug)
        st.caption(f"Датасет: SWE-QA / {split}" if split else
                   "Для этого репозитория нет сплита SWE-QA — используйте CLI: --dataset stackrepoqa | "
                   "coderepoqa:<файл> | свой JSONL")
        qa_mode = st.radio("Режим", ["retrieval", "agent"], horizontal=True, key="qa_mode",
                           help="retrieval — BM25 без LLM (baseline); agent — Claude с инструментами")
        qa_limit = st.number_input("Вопросов", 1, 200, 10, key="qa_limit")
        if split and st.button("Запустить QA-оценку"):
            if qa_mode == "agent" and not llm:
                st.error("Нужен Claude (ANTHROPIC_API_KEY).")
            else:
                bar = st.progress(0.0)
                res = qa_eval.evaluate(idx, qa_eval.load_sweqa(split), mode=qa_mode, llm=llm, limit=int(qa_limit),
                                       dataset=f"swe-qa/{split}",
                                       progress=lambda i, n, q: bar.progress(i / n, text=q[:80]))
                qa_eval.save(res, config.WORKSPACE / slug / "eval")
                bar.empty()
        res = latest_eval(slug, "qa_")
        if res:
            mt = res["metrics"]
            st.caption(f"Последний прогон: {res.get('dataset', '?')} · {res['mode']} · {res['n_questions']} вопросов · "
                       f"{res.get('model') or 'без LLM'}")
            a, b, c = st.columns(3)
            a.metric("Верный файл (первый)", pct(mt.get("primary_hit")),
                     help="Первый процитированный файл — эталонный. 95% ДИ: " + ci(mt.get("primary_hit_ci95")))
            b.metric("Верный среди 3", pct(mt.get("file_hit3")))
            c.metric("Валидные ссылки", pct(mt.get("cite_valid")))
            st.caption(f"Для сравнения: всегда называть самый частый эталонный файл — "
                       f"{pct(mt.get('baseline_popular_hit1'))}")
            st.dataframe([{"вопрос": r["question"][:90], "верный файл": r.get("primary_hit"),
                           "процитировано": ", ".join(r.get("cited_files", [])[:3]), "эталон": ", ".join(r["gold"][:3])}
                          for r in res["rows"]], hide_index=True)
    with c2:
        st.subheader("Доля найденных внесённых багов")
        src = st.radio("Источник багов", ["synthetic", "history"], horizontal=True,
                       help="synthetic — мутации кода; history — реальные баги из fix-коммитов")
        n_tr = st.number_input("Проб", 1, 100, 10)
        if st.button("Запустить оценку ревью"):
            bar = st.progress(0.0)
            res = bugs_eval.run(idx, llm, source=src, n=int(n_tr),
                                progress=lambda i, n, t: bar.progress(i / n, text=f"{t.op} {t.path}:{t.lines[0]}"))
            bugs_eval.save(res, config.WORKSPACE / slug / "eval")
            bar.empty()
        res = latest_eval(slug, "bugs_")
        if res:
            st.caption(f"Последний прогон: {res['source']} · {res['reviewer']} · {res['n_trials']} проб")
            if not res["n_trials"]:
                st.info("Нет проб: внесение багов поддерживается для Python-кода; для проб из истории нужен "
                        "полный клон и fix-коммиты.")
            else:
                a, b, c = st.columns(3)
                a.metric("Найдено багов", pct(res["detection_rate"]), help="95% ДИ: " + ci(res.get("detection_ci95")))
                b.metric("Ложные тревоги", pct(res.get("control_false_alarm_rate")),
                         help="То же место в исправном коде помечено как баг")
                c.metric("Различающая способность", pct(res.get("discrimination")),
                         help="Найдено − ложные тревоги; 0 = не лучше, чем помечать всё подряд")
                st.caption(f"Точность замечаний: {pct(res.get('precision'))} · случайное угадывание при том же числе "
                           f"замечаний: {pct(res.get('chance_same_volume'))} · «детектор артефактов мутаций» "
                           f"(не понимает код): {pct(res.get('artifact_baseline_discrimination'))}")
                st.dataframe([{"тип": k, "проб": v["n"], "найдено": pct(v["rate"])} for k, v in res["by_op"].items()],
                             hide_index=True)
