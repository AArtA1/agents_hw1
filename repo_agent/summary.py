"""Automatic project summary: languages, stack, entry points, layout.

Fully deterministic (works without an API key); `narrative()` optionally asks Claude
to turn the facts into a short human description.
"""

from __future__ import annotations

import json
import posixpath
import re
import tomllib
from collections import Counter
from pathlib import PurePosixPath

from . import config
from .index import RepoIndex
from .repo import read_text
from .textutil import is_aux_path, is_test_path, line_count, mask_python, split_lines

# dependency name (lowercase) -> human label
FRAMEWORKS = {
    # python
    "django": "Django", "flask": "Flask", "fastapi": "FastAPI", "starlette": "Starlette",
    "aiohttp": "aiohttp", "tornado": "Tornado", "sqlalchemy": "SQLAlchemy", "pydantic": "Pydantic",
    "celery": "Celery", "pytest": "pytest", "numpy": "NumPy", "pandas": "pandas", "torch": "PyTorch",
    "tensorflow": "TensorFlow", "scikit-learn": "scikit-learn", "click": "Click", "typer": "Typer",
    "streamlit": "Streamlit", "requests": "requests", "httpx": "httpx", "urllib3": "urllib3",
    "anthropic": "Anthropic SDK", "openai": "OpenAI SDK", "langchain": "LangChain", "uvicorn": "Uvicorn",
    "gunicorn": "Gunicorn", "alembic": "Alembic", "jinja2": "Jinja2", "sphinx": "Sphinx",
    "werkzeug": "Werkzeug", "marshmallow": "marshmallow", "redis": "Redis", "boto3": "boto3",
    "grpcio": "gRPC", "transformers": "Transformers", "langgraph": "LangGraph",
    # js
    "react": "React", "next": "Next.js", "vue": "Vue", "nuxt": "Nuxt", "svelte": "Svelte",
    "@angular/core": "Angular", "express": "Express", "koa": "Koa", "fastify": "Fastify",
    "@nestjs/core": "NestJS", "typescript": "TypeScript", "jest": "Jest", "vitest": "Vitest",
    "webpack": "webpack", "vite": "Vite", "prisma": "Prisma", "@prisma/client": "Prisma", "mongoose": "Mongoose",
    "next-auth": "NextAuth.js", "@trpc/server": "tRPC", "zod": "Zod", "drizzle-orm": "Drizzle",
    "electron": "Electron", "tailwindcss": "Tailwind CSS",
    # go / rust / jvm
    "github.com/gin-gonic/gin": "Gin", "github.com/labstack/echo": "Echo", "github.com/spf13/cobra": "Cobra",
    "google.golang.org/grpc": "gRPC", "tokio": "Tokio", "actix-web": "Actix", "axum": "Axum",
    "serde": "Serde", "clap": "clap", "spring-boot": "Spring Boot",
}

MANIFESTS = {
    "pyproject.toml": "Python (pyproject)", "setup.py": "Python (setuptools)", "setup.cfg": "Python (setuptools)",
    "requirements.txt": "Python (pip)", "Pipfile": "Python (pipenv)", "package.json": "Node.js (npm)",
    "go.mod": "Go modules", "Cargo.toml": "Rust (cargo)", "pom.xml": "Java (Maven)",
    "build.gradle": "JVM (Gradle)", "build.gradle.kts": "JVM (Gradle)", "Gemfile": "Ruby (bundler)",
    "composer.json": "PHP (composer)", "Dockerfile": "Docker", "docker-compose.yml": "Docker Compose",
    "docker-compose.yaml": "Docker Compose", "Makefile": "Make", "tox.ini": "tox", "noxfile.py": "nox",
    ".github/workflows": "GitHub Actions",
}


_REQ_SPLIT = re.compile(r"[<>=!~\[; @]")


def _req_name(spec: str) -> str:
    return _REQ_SPLIT.split(spec.strip(), maxsplit=1)[0].lower().replace("_", "-")


def _deps_python(idx: RepoIndex) -> tuple[set[str], set[str]]:
    """(runtime, dev) dependency names from pyproject / requirements / setup.py."""
    run: set[str] = set()
    dev: set[str] = set()
    for f in idx.files:
        name = PurePosixPath(f).name
        if is_aux_path(f) or is_test_path(f):
            continue
        if name == "pyproject.toml":
            try:
                data = tomllib.loads(read_text(idx.root, f) or "")
            except tomllib.TOMLDecodeError:
                continue
            proj = data.get("project", {})
            run |= {_req_name(d) for d in proj.get("dependencies", [])}
            dev |= {_req_name(d) for d in sum(proj.get("optional-dependencies", {}).values(), [])}
            for grp in data.get("dependency-groups", {}).values():
                dev |= {_req_name(d) for d in grp if isinstance(d, str)}
            poetry = data.get("tool", {}).get("poetry", {})
            run |= {k.lower() for k in poetry.get("dependencies", {}) if k.lower() != "python"}
            dev |= {k.lower() for k in poetry.get("dev-dependencies", {})}
            for g in poetry.get("group", {}).values():
                dev |= {k.lower() for k in g.get("dependencies", {})}
        elif re.match(r"requirements.*\.(txt|in)$", name):
            target = run if re.fullmatch(r"requirements\.(txt|in)", name) else dev
            for line in split_lines(read_text(idx.root, f) or ""):
                line = line.strip()
                if line and not line.startswith(("#", "-")):
                    target.add(_req_name(line))
        elif name == "setup.py":
            text = read_text(idx.root, f) or ""
            m = re.search(r"install_requires\s*=\s*\[(.*?)\]", text, re.S)
            if m:
                run |= {_req_name(x) for x in re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))}
            m = re.search(r"(?:tests_require|extras_require)\s*=\s*[\[{](.*?)[\]}]\s*,", text, re.S)
            if m:
                dev |= {_req_name(x) for x in re.findall(r"['\"]([A-Za-z0-9_.\-]+)\s*(?:[<>=!~][^'\"]*)?['\"]", m.group(1))}
    return run, dev - run


def _deps_node(idx: RepoIndex) -> tuple[set[str], set[str], list[dict]]:
    run, dev, pkgs = set(), set(), []
    for f in idx.files:
        if PurePosixPath(f).name == "package.json" and "node_modules" not in f and not is_aux_path(f) \
                and not is_test_path(f):
            try:
                data = json.loads(read_text(idx.root, f) or "{}")
            except json.JSONDecodeError:
                continue
            run |= set((data.get("dependencies") or {}).keys()) | set((data.get("peerDependencies") or {}).keys())
            dev |= set((data.get("devDependencies") or {}).keys())
            pkgs.append({"path": f, "data": data})
    return run, dev - run, pkgs


def _deps_other(idx: RepoIndex) -> set[str]:
    deps = set()
    for f in idx.files:
        name = PurePosixPath(f).name
        if name == "go.mod":
            for m in re.finditer(r"^\s*([\w./-]+\.[\w./-]+)\s+v", read_text(idx.root, f) or "", re.M):
                deps.add(m.group(1))
        elif name == "Cargo.toml":
            try:
                data = tomllib.loads(read_text(idx.root, f) or "")
            except tomllib.TOMLDecodeError:
                continue
            deps |= set(data.get("dependencies", {}).keys())
        elif name == "pom.xml" or name.startswith("build.gradle"):
            for m in re.finditer(r"(spring-boot)", read_text(idx.root, f) or ""):
                deps.add(m.group(1))
    return deps


# priority: how directly the entry point answers "how do I run/use this project"
EP_PRIORITY = {"console script": 0, "npm bin": 0, "python -m": 1, "go main": 1, "rust main": 1, "jvm main": 1,
               "library API": 2, "npm main": 2, "npm exports": 2, "web app": 3, "app module": 4, "docker": 5,
               "procfile": 5, "next route": 5, "npm script": 6, "python __main__": 7, "benchmark main": 9}
_BENCH = re.compile(r"(^|/)(bench|benches|benchmarks?|perf|jmh|metrics)(/|$)|Benchmark", re.I)
_NEXT_ROUTE = re.compile(r"(^|/)(app/(.*/)?(page|route)|pages/api/.+)\.(t|j)sx?$")


def _py_module_name(idx: RepoIndex, path: str) -> str:
    """Dotted import name of a module/package dir, honouring src-layout ('src/flask/__main__.py' -> 'flask')."""
    parts = list(PurePosixPath(path).parent.parts)
    files = idx._file_set
    i = len(parts)
    while i > 0 and "/".join(parts[:i]) + "/__init__.py" in files:
        i -= 1
    return ".".join(parts[i:]) or PurePosixPath(path).parent.name


def entry_points(idx: RepoIndex) -> tuple[list[dict], int]:
    """Where execution starts. Returns (entry points sorted by priority, number skipped in docs/examples/tests)."""
    eps: list[dict] = []
    skipped = 0

    def add(kind, path, line, detail):
        eps.append({"kind": kind, "path": path, "line": line, "detail": detail})

    pkg_names: set[str] = set()
    for f in idx.files:
        name = PurePosixPath(f).name
        lang = config.language_of(f)
        aux = is_aux_path(f) or is_test_path(f)
        if lang == "python":
            text = read_text(idx.root, f) or ""
            code = mask_python(text) or split_lines(text)
            found = []
            for i, line in enumerate(code, 1):
                if re.match(r"""^if\s+__name__\s*==""", line) and "__main__" in split_lines(text)[i - 1]:
                    found.append(("python __main__", i, "запуск как скрипт"))
                    break
            if name == "__main__.py":
                found.append(("python -m", 1, f"python -m {_py_module_name(idx, f)}"))
            if (name in ("manage.py", "wsgi.py", "asgi.py") and f.count("/") <= 1) or \
                    (name in ("app.py", "main.py") and "/" not in f):
                found.append(("app module", 1, name))
            for i, line in enumerate(code, 1):  # top-level `app = Flask(...)` in real code, not in docstrings
                m = re.match(r"^(\w+)\s*(?::\s*\w+\s*)?=\s*(FastAPI|Flask|Starlette|Quart|Sanic)\(", line)
                if m:
                    found.append(("web app", i, f"{m.group(2)} app `{m.group(1)}`"))
                    break
            if found and aux:
                skipped += len(found)
            elif found:
                for k, ln, d in found:
                    add(k, f, ln, d)
        elif name == "pyproject.toml" and not aux:
            try:
                data = tomllib.loads(read_text(idx.root, f) or "")
            except tomllib.TOMLDecodeError:
                continue
            proj = data.get("project", {})
            if proj.get("name"):
                pkg_names.add(proj["name"].lower().replace("-", "_"))
            scripts = dict(proj.get("scripts", {}) or {})
            scripts.update(data.get("tool", {}).get("poetry", {}).get("scripts", {}) or {})
            for k, v in scripts.items():
                add("console script", f, _toml_key_line(idx, f, ("project.scripts", "tool.poetry.scripts"), k),
                    f"{k} -> {v}")
        elif name == "setup.py" and not aux:
            text = read_text(idx.root, f) or ""
            for m in re.finditer(r"['\"]([\w-]+)\s*=\s*([\w.]+:[\w.]+)['\"]", text):
                add("console script", f, text[:m.start()].count("\n") + 1, f"{m.group(1)} -> {m.group(2)}")
            m = re.search(r"\bname\s*=\s*['\"]([\w.\-]+)['\"]", text)
            if m:
                pkg_names.add(m.group(1).lower().replace("-", "_"))
        elif name == "package.json" and not aux and "node_modules" not in f:
            try:
                data = json.loads(read_text(idx.root, f) or "{}")
            except json.JSONDecodeError:
                continue
            base = posixpath.dirname(f)
            if data.get("main"):
                add("npm main", f, _line_of(idx, f, '"main"'), data["main"])
            exp = data.get("exports")
            if isinstance(exp, str):
                add("npm exports", f, _line_of(idx, f, '"exports"'), exp)
            elif isinstance(exp, dict):
                dot = exp.get(".", exp)
                target = dot if isinstance(dot, str) else json.dumps(dot, ensure_ascii=False)[:120]
                add("npm exports", f, _line_of(idx, f, '"exports"'), target)
            if not data.get("main") and not exp:
                implicit = posixpath.join(base, "index.js") if base else "index.js"
                if idx.has_file(implicit):
                    add("npm main", implicit, 1, "index.js (неявный main)")
            bins = data.get("bin")
            if isinstance(bins, str):
                add("npm bin", f, _line_of(idx, f, '"bin"'), bins)
            elif isinstance(bins, dict):
                for k, v in bins.items():
                    add("npm bin", f, _line_of(idx, f, k), f"{k} -> {v}")
            for k in ("start", "dev", "serve"):
                if k in (data.get("scripts") or {}):
                    add("npm script", f, _line_of(idx, f, f'"{k}"'), f"npm run {k}: {data['scripts'][k]}")
        elif lang == "go":
            text = read_text(idx.root, f) or ""
            if re.search(r"^package main\b", text, re.M):
                m = re.search(r"^func main\(\)", text, re.M)
                if m:
                    if aux:
                        skipped += 1
                    else:
                        add("go main", f, text[:m.start()].count("\n") + 1, "func main()")
        elif lang in ("javascript", "typescript") and _NEXT_ROUTE.search(f) and not aux:
            add("next route", f, 1, "/" + re.sub(r"^.*?(app|pages)/|/?(page|route)?\.(t|j)sx?$", "", f))
        elif lang == "rust" and name == "main.rs" and not aux:
            add("rust main", f, 1, "fn main")
        elif name in ("Dockerfile", "Containerfile") and not aux:
            for i, line in enumerate(split_lines(read_text(idx.root, f) or ""), 1):
                if re.match(r"^\s*(CMD|ENTRYPOINT)\b", line):
                    add("docker", f, i, line.strip())
        elif name == "Procfile" and not aux:
            for i, line in enumerate(split_lines(read_text(idx.root, f) or ""), 1):
                if ":" in line:
                    add("procfile", f, i, line.strip())
        elif lang in ("java", "kotlin") and not aux:
            text = read_text(idx.root, f) or ""
            m = re.search(r"public\s+static\s+void\s+main\s*\(|^fun\s+main\s*\(|@SpringBootApplication", text, re.M)
            if m:
                add("jvm main", f, text[:m.start()].count("\n") + 1, m.group(0).strip())
    # Python library: its public API is the top-level package __init__.py
    for pkg in sorted(pkg_names):
        for cand in (f"{pkg}/__init__.py", f"src/{pkg}/__init__.py", f"lib/{pkg}/__init__.py"):
            if idx.has_file(cand):
                add("library API", cand, 1, f"import {pkg}")
                break
    for e in eps:
        if e["kind"] in ("jvm main", "go main", "rust main", "python __main__") and _BENCH.search(e["path"]):
            e["kind"] = "benchmark main"
    eps.sort(key=lambda e: (EP_PRIORITY.get(e["kind"], 9), e["path"].count("/"), e["path"]))
    return eps, skipped


def _toml_key_line(idx: RepoIndex, rel: str, sections: tuple[str, ...], key: str) -> int:
    """Line of `key = ...` inside one of the given [sections] of a TOML file."""
    current = None
    for i, line in enumerate(split_lines(read_text(idx.root, rel) or ""), 1):
        m = re.match(r"^\s*\[([^\]]+)\]\s*$", line)
        if m:
            current = m.group(1).strip()
            continue
        if current in sections and re.match(rf"""^\s*["']?{re.escape(key)}["']?\s*=""", line):
            return i
    return _line_of(idx, rel, key)


def _line_of(idx: RepoIndex, rel: str, needle: str) -> int:
    for i, line in enumerate(split_lines(read_text(idx.root, rel) or ""), 1):
        if needle in line:
            return i
    return 1


def _is_test(path: str) -> bool:
    return is_test_path(path)


def _readme(idx: RepoIndex) -> tuple[str | None, str]:
    cands = sorted((f for f in idx.files if PurePosixPath(f).name.lower().startswith("readme")),
                   key=lambda f: (f.count("/"), len(f)))
    if not cands:
        return None, ""
    return cands[0], (read_text(idx.root, cands[0]) or "")


def _readme_intro(text: str, limit: int = 700) -> str:
    """First meaningful paragraph of a README (skip badges, headings, html)."""
    paras = re.split(r"\n\s*\n", text)
    for p in paras:
        lines = [re.sub(r"^\s*>\s?", "", l) for l in p.strip().split("\n")]          # blockquotes
        lines = [l for l in lines if not re.match(r"^\s*\[!\w+\]\s*$", l)]          # GitHub [!NOTE]
        keep = []
        for l in lines:                                                              # setext headings
            if re.match(r"^\s*[=\-]{3,}\s*$", l):
                if keep:
                    keep.pop()
                continue
            keep.append(l)
        s = "\n".join(keep).strip()
        s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)                                   # images
        s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)                                # [text](url) -> text
        s = re.sub(r"\[([^\]]+)\](?:\[[^\]]*\])?", r"\1", s).strip()                  # [ref] / [text][ref]
        if not s or s.startswith(("#", "<", "[![", "!", "=", "-", "|", "..", ":", "```")) or len(s) < 40:
            continue
        urls = sum(len(u) for u in re.findall(r"https?://\S+", s))
        if urls > 0.4 * len(s) or re.match(r"^\**(documentation|source code|docs|website|homepage)\**\s*:", s, re.I):
            continue
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\*\*|__|`", "", s)                       # markdown emphasis / code
        s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)        # [text](url) -> text
        return s[:limit]
    return ""


def build_summary(idx: RepoIndex) -> dict:
    lang_lines: Counter = Counter()
    lang_files: Counter = Counter()
    for f in idx.files:
        lang = config.language_of(f)
        if lang and config.is_code(f):
            n = line_count(read_text(idx.root, f) or "")
            lang_lines[lang] += n
            lang_files[lang] += 1
    total = sum(lang_lines.values()) or 1

    py_run, py_dev = _deps_python(idx)
    node_run, node_dev, _pkgs = _deps_node(idx)
    deps = py_run | node_run | _deps_other(idx)
    own = {idx.slug.split("__")[-1].lower(), idx.slug.split("__")[-1].lower().replace("-", "_")}
    deps -= own
    dev_deps = (py_dev | node_dev) - deps - own
    frameworks = sorted({label for dep, label in FRAMEWORKS.items() if dep in deps})
    dev_tools = sorted({label for dep, label in FRAMEWORKS.items() if dep in dev_deps} - set(frameworks))
    eps, eps_aux = entry_points(idx)

    manifests = []
    for m, label in MANIFESTS.items():
        hit = [f for f in idx.files if f == m or f.startswith(m + "/") or f.endswith("/" + m)]
        if hit:
            manifests.append({"file": sorted(hit, key=len)[0], "label": label})

    # top-level layout
    top: Counter = Counter()
    for f in idx.files:
        parts = PurePosixPath(f).parts
        top[parts[0] + ("/" if len(parts) > 1 else "")] += 1

    readme_path, readme = _readme(idx)
    tests = [f for f in idx.files if is_test_path(f) and config.is_code(f)]
    ci = [f for f in idx.files if f.startswith(".github/workflows/") or f in (".gitlab-ci.yml", ".travis.yml")]

    return {
        "slug": idx.slug,
        "web_url": idx.meta.get("web_url"),
        "head": idx.meta.get("head"),
        "readme_path": readme_path,
        "readme_intro": _readme_intro(readme),
        "languages": [{"language": l, "lines": n, "files": lang_files[l], "share": round(n / total, 3)}
                      for l, n in lang_lines.most_common()],
        "frameworks": frameworks,
        "dev_tools": dev_tools,
        "dependencies": sorted(deps)[:80],
        "dev_dependencies": sorted(dev_deps)[:80],
        "manifests": manifests,
        "entry_points": eps,
        "entry_points_aux": eps_aux,
        "layout": [{"path": p, "files": n} for p, n in top.most_common(15)],
        "tests": {"files": len(tests), "ci": ci},
        "history": {
            "commits_indexed": len(idx.commits),
            "last_commit": idx.commits[0].__dict__ if idx.commits else None,
            "top_authors": Counter(c.author for c in idx.commits).most_common(5),
            "hot_files": Counter(f for c in idx.commits for f in c.files).most_common(10),
        },
        "stats": {k: idx.meta[k] for k in ("n_files", "n_chunks", "n_symbols", "n_commits")},
    }


def summary_markdown(s: dict) -> str:
    L = [f"# {s['slug']}", ""]
    if s.get("web_url"):
        L.append(f"Репозиторий: {s['web_url']} (commit `{(s.get('head') or '')[:10]}`)")
    if s.get("readme_intro"):
        L += ["", f"> {s['readme_intro']}"]
    L += ["", "## Стек"]
    langs = ", ".join(f"{x['language']} {x['share']*100:.0f}% ({x['lines']} строк)" for x in s["languages"][:6])
    L.append(f"- **Языки:** {langs or '—'}")
    L.append(f"- **Фреймворки/библиотеки:** {', '.join(s['frameworks']) or '—'}")
    if s.get("dev_tools"):
        L.append(f"- **Инструменты разработки:** {', '.join(s['dev_tools'])}")
    L.append(f"- **Сборка/инфраструктура:** {', '.join(sorted({m['label'] for m in s['manifests']})) or '—'}")
    L.append(f"- **Тесты:** {s['tests']['files']} файлов; CI: {', '.join(s['tests']['ci']) or 'нет'}")
    L += ["", "## Точки входа"]
    if s["entry_points"]:
        for e in s["entry_points"][:25]:
            L.append(f"- `{e['path']}:{e['line']}` — {e['kind']}: {e['detail']}")
        if len(s["entry_points"]) > 25:
            L.append(f"- … и ещё {len(s['entry_points']) - 25}")
    else:
        L.append("- не найдены (вероятно, библиотека без CLI)")
    if s.get("entry_points_aux"):
        L.append(f"- (ещё {s['entry_points_aux']} в документации/примерах/тестах — не показаны)")
    L += ["", "## Структура (верхний уровень)"]
    L += [f"- `{x['path']}` — {x['files']} файлов" for x in s["layout"]]
    h = s["history"]
    L += ["", "## История"]
    if h["last_commit"]:
        lc = h["last_commit"]
        L.append(f"- Последний коммит: `{lc['sha'][:8]}` {lc['date']} — {lc['subject']} ({lc['author']})")
    L.append(f"- Активные авторы: {', '.join(f'{a} ({n})' for a, n in h['top_authors'])}")
    L.append(f"- Часто меняемые файлы: {', '.join(f'`{f}` ({n})' for f, n in h['hot_files'][:6])}")
    return "\n".join(L)


NARRATIVE_PROMPT = """Ниже — факты о репозитории, собранные статическим анализом, и начало README.
Напиши по-русски краткую сводку проекта (5–8 предложений): что это за проект, для кого, основной стек,
как его запускать/использовать (точки входа), как устроен код (ключевые каталоги). Опирайся только на факты.
Ссылайся на файлы в формате `path:line`, когда упоминаешь точку входа.

ФАКТЫ:
{facts}

README (начало):
{readme}
"""


def narrative(idx: RepoIndex, s: dict) -> str:
    from .llm import LLM  # lazy: summary must work without the SDK configured

    _, readme = _readme(idx)
    facts = json.dumps({k: s[k] for k in ("languages", "frameworks", "manifests", "entry_points", "layout")},
                       ensure_ascii=False)[:12000]
    return LLM().complete(NARRATIVE_PROMPT.format(facts=facts, readme=readme[:6000]), effort="low")
