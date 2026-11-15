"""Self-contained HTML report: summary, module map (Mermaid), review, latest eval metrics."""

from __future__ import annotations

import html
import json
from pathlib import Path

from . import config
from .index import RepoIndex

MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"


def _e(s) -> str:
    return html.escape(str(s))


def _loc(idx: RepoIndex, path: str, line: int | None = None, end: int | None = None) -> str:
    label = f"{path}:{line}" if line else path
    url = idx.blob_url(path, line, end)
    return f'<a href="{_e(url)}"><code>{_e(label)}</code></a>' if url else f"<code>{_e(label)}</code>"


def latest_eval(slug: str, prefix: str) -> dict | None:
    d = config.WORKSPACE / slug / "eval"
    files = sorted(d.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime) if d.exists() else []
    return json.loads(files[-1].read_text()) if files else None


def render(idx: RepoIndex, summary: dict, mmap: dict, review: dict | None, narrative: str | None = None) -> str:
    s = summary
    langs = "".join(f"<tr><td>{_e(l['language'])}</td><td>{l['files']}</td><td>{l['lines']}</td>"
                    f"<td>{l['share']*100:.0f}%</td></tr>" for l in s["languages"][:8])
    eps = "".join(f"<li>{_loc(idx, e['path'], e['line'])} — {_e(e['kind'])}: {_e(e['detail'])}</li>"
                  for e in s["entry_points"][:30]) or "<li>не найдены</li>"
    nodes = sorted(mmap["nodes"], key=lambda n: -(n["fan_in"] + n["fan_out"]))[:15]
    hubs = "".join(f"<tr><td><code>{_e(n['id'])}</code></td><td>{n['fan_in']}</td><td>{n['fan_out']}</td></tr>"
                   for n in nodes)
    cycles = ", ".join(f"<code>{_e(a)}</code> ⇄ <code>{_e(b)}</code>" for a, b in mmap["cycles"][:10]) or "нет"

    rev_html = "<p>Ревью ещё не запускалось: <code>repo-agent review …</code></p>"
    if review:
        c = review["counts"]
        items = []
        for f in review["findings"][:80]:
            items.append(
                f'<div class="f sev-{_e(f["severity"])}"><div class="fh"><span class="badge">{_e(f["severity"])}</span>'
                f'<span class="badge cat">{_e(f["category"])}</span> <b>{_e(f["title"])}</b></div>'
                f'<div class="fl">{_loc(idx, f["path"], f["line"], f.get("end_line"))} · {_e(f["source"])}'
                f' · уверенность {f["confidence"]:.2f}</div><div>{_e(f["explanation"])}</div>'
                + (f'<div class="sg"><b>Исправление:</b> {_e(f["suggestion"])}</div>' if f.get("suggestion") else "")
                + "</div>")
        rev_html = (f"<p>Область: {_e(review['scope'])}. Всего: <b>{c['total']}</b> · "
                    + " · ".join(f"{_e(k)}: {v}" for k, v in c["by_severity"].items()) + "</p>" + "".join(items))

    metrics_html = ""
    qa = latest_eval(idx.slug, "qa_agent") or latest_eval(idx.slug, "qa_retrieval")
    bugs = latest_eval(idx.slug, "bugs_")
    if qa:
        m = qa["metrics"]
        mode_label = "агент с Claude" if qa["mode"] == "agent" else "поиск BM25 без LLM (baseline)"
        metrics_html += (f"<h3>Вопросы-ответы: {_e(qa.get('dataset', '?'))}, {_e(mode_label)}, "
                         f"{qa['n_questions']} вопросов</h3><ul>"
                         f"<li>Доля ответов с верным файлом (первый процитированный файл — эталонный): "
                         f"<b>{_pct(m.get('primary_hit'))}</b> {_ci(m.get('primary_hit_ci95'))}</li>"
                         f"<li>Верный файл среди первых трёх: {_pct(m.get('file_hit3'))} {_ci(m.get('file_hit3_ci95'))}</li>"
                         f"<li>Валидность ссылок (файл и строки существуют): {_pct(m.get('cite_valid'))}</li>"
                         f"<li>Для сравнения — всегда называть самый частый эталонный файл: "
                         f"{_pct(m.get('baseline_popular_hit1'))}</li></ul>")
    if bugs and bugs.get("n_trials"):
        metrics_html += (f"<h3>Внесённые баги ({_e(bugs['source'])}, {_e(bugs['reviewer'])}, {bugs['n_trials']} проб)</h3><ul>"
                         f"<li>Доля найденных багов: <b>{_pct(bugs['detection_rate'])}</b> {_ci(bugs.get('detection_ci95'))}</li>"
                         f"<li>Ложные тревоги на том же месте в исправном коде: {_pct(bugs.get('control_false_alarm_rate'))}</li>"
                         f"<li>Различающая способность (найдено − ложные тревоги): {_pct(bugs.get('discrimination'))} "
                         f"{_ci(bugs.get('discrimination_ci95'))}</li>"
                         f"<li>Для сравнения — «детектор артефактов мутаций», не понимающий код: "
                         f"{_pct(bugs.get('artifact_baseline_discrimination'))}</li>"
                         + "".join(f"<li>{_e(k)}: найдено {v['k']} из {v['n']}</li>" for k, v in bugs.get("by_kind", {}).items()) +
                         f"<li>Точность замечаний (указывают на баг): {_pct(bugs.get('precision'))}</li>"
                         f"<li>Случайное угадывание при том же числе замечаний: {_pct(bugs.get('chance_same_volume'))}</li></ul>")
    if not metrics_html:
        metrics_html = "<p>Оценка ещё не запускалась: <code>repo-agent eval-qa …</code>, <code>repo-agent eval-bugs …</code></p>"

    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Repo report: {_e(idx.slug)}</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#6b6b70;--card:#f6f6f8;--bd:#e3e3e8;--acc:#3b5bdb}}
@media (prefers-color-scheme: dark){{:root{{--bg:#121214;--fg:#ececf0;--mut:#9a9aa3;--card:#1c1c20;--bd:#2c2c33;--acc:#8ea2ff}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,sans-serif;max-width:1100px;margin:0 auto;padding:16px}}
a{{color:var(--acc)}} code{{font-size:13px}} h2{{border-bottom:1px solid var(--bd);padding-bottom:4px;margin-top:32px}}
table{{border-collapse:collapse}} td,th{{border:1px solid var(--bd);padding:4px 8px;text-align:left}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:12px;overflow-x:auto}}
.f{{border-left:4px solid var(--bd);background:var(--card);padding:8px 12px;margin:8px 0;border-radius:4px}}
.sev-critical{{border-color:#e03131}} .sev-high{{border-color:#f76707}} .sev-medium{{border-color:#f59f00}}
.badge{{font-size:12px;padding:1px 6px;border-radius:4px;background:var(--bd);margin-right:4px}}
.fl{{color:var(--mut);font-size:13px}} .sg{{margin-top:4px}}
</style></head><body>
<h1>{_e(idx.slug)}</h1>
<p>{_loc(idx, s['readme_path']) if s.get('readme_path') else ''} · commit <code>{_e((s.get('head') or '')[:10])}</code>
· {s['stats']['n_files']} файлов · {s['stats']['n_symbols']} символов · {s['stats']['n_commits']} коммитов</p>
{f'<div class="card">{_e(narrative)}</div>' if narrative else (f'<blockquote>{_e(s["readme_intro"])}</blockquote>' if s.get("readme_intro") else '')}
<h2>Стек</h2>
<table><tr><th>Язык</th><th>Файлов</th><th>Строк</th><th>Доля</th></tr>{langs}</table>
<p><b>Фреймворки/библиотеки:</b> {_e(', '.join(s['frameworks']) or '—')}<br>
<b>Сборка/инфраструктура:</b> {_e(', '.join(sorted({m['label'] for m in s['manifests']})) or '—')}<br>
<b>Тесты:</b> {s['tests']['files']} файлов; CI: {_e(', '.join(s['tests']['ci']) or 'нет')}</p>
<h2>Точки входа</h2><ul>{eps}</ul>
<h2>Карта модулей</h2>
<p>Уровень: {_e(mmap['level'])}; узлов {len(mmap['nodes'])}, связей {len(mmap['edges'])}. Взаимные импорты: {cycles}</p>
<div class="card"><pre class="mermaid">{_e(mmap['mermaid'])}</pre></div>
<h3>Самые связанные модули</h3><table><tr><th>Модуль</th><th>Входящие</th><th>Исходящие</th></tr>{hubs}</table>
<h2>Ревью</h2>{rev_html}
<h2>Метрики</h2>{metrics_html}
<script src="{MERMAID_CDN}"></script>
<script>mermaid.initialize({{startOnLoad:true,theme:matchMedia('(prefers-color-scheme: dark)').matches?'dark':'default',maxTextSize:200000,flowchart:{{useMaxWidth:true}}}});</script>
</body></html>"""


def _pct(x) -> str:
    return "—" if x is None else f"{x*100:.1f}%"


def _ci(ci) -> str:
    return f"(95% ДИ {ci[0]*100:.0f}–{ci[1]*100:.0f}%)" if ci else ""


def write(idx: RepoIndex, html_text: str) -> Path:
    p = config.WORKSPACE / idx.slug / "report.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html_text)
    return p
