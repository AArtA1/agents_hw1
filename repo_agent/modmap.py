"""Module map: internal import graph (Python / JS / TS / Go), aggregated and rendered as Mermaid."""

from __future__ import annotations

import ast
import json
import posixpath
import re
from collections import Counter, defaultdict
from pathlib import PurePosixPath

from . import config
from .index import RepoIndex
from .repo import read_text
from .textutil import is_aux_path, is_test_path, mask_c_like

_JS_IMPORT = re.compile(r"""(?:\bfrom\s*|\bimport\s*\(\s*|\brequire\s*\(\s*|^\s*import\s+)['"]([^'"]+)['"]""", re.M)
_GO_IMPORT_BLOCK = re.compile(r"^import\s*\((.*?)^\)", re.S | re.M)
_GO_IMPORT_ONE = re.compile(r'^import\s+(?:[\w.]+\s+)?"([^"]+)"', re.M)
_JS_SRC_EXT = (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")
_JS_TRY = ("", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts",
           "/index.ts", "/index.tsx", "/index.js", "/index.jsx", "/index.mjs")


# ------------------------------------------------------------------ python

def _py_file(base: str, fileset: set[str]) -> str | None:
    """'pkg/mod' -> 'pkg/mod.py' or 'pkg/mod/__init__.py' if indexed."""
    base = base.strip("/")
    for cand in (base + ".py", (base + "/__init__.py") if base else "__init__.py"):
        if cand in fileset:
            return cand
    return None


def _python_module_table(files: list[str]) -> dict[str, str]:
    """dotted-name suffix -> file. 'src/requests/utils.py' registers 'requests.utils' and 'utils'."""
    by_key: dict[str, list[str]] = defaultdict(list)
    for f in files:
        if not f.endswith(".py"):
            continue
        parts = list(PurePosixPath(f).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        for i in range(len(parts)):
            key = ".".join(parts[i:])
            if key:
                by_key[key].append(f)
    # ambiguous key -> prefer the shortest path (closest to a package root)
    return {k: min(v, key=len) for k, v in by_key.items()}


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or \
           (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _runtime_imports(tree: ast.AST):
    """Import nodes, skipping `if TYPE_CHECKING:` blocks (they don't create runtime coupling)."""
    stack = [tree]
    while stack:
        node = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and _is_type_checking(child.test):
                stack.extend(child.orelse)
                continue
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                yield child
            stack.append(child)


def _py_imports(path: str, text: str, table: dict[str, str], fileset: set[str]) -> set[str]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return set()
    pkg_parts = list(PurePosixPath(path).parent.parts)
    targets: set[str] = set()

    def resolve(name: str) -> str | None:
        while name:
            if name in table:
                return table[name]
            name = name.rpartition(".")[0]
        return None

    for node in _runtime_imports(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                t = resolve(a.name)
                if t:
                    targets.add(t)
        elif node.level:  # relative: from . import x / from .mod import y / from ..pkg.mod import z
            up = node.level - 1
            if up > len(pkg_parts):
                continue
            base = "/".join(pkg_parts[: len(pkg_parts) - up])
            mod_path = posixpath.join(base, *node.module.split(".")) if node.module else base
            mod_file = _py_file(mod_path, fileset)
            for a in node.names:
                sub = _py_file(posixpath.join(mod_path, a.name), fileset) if a.name != "*" else None
                if sub:                       # the name is a submodule
                    targets.add(sub)
                elif mod_file:                # the name is an attribute of the module/package
                    targets.add(mod_file)
        elif node.module:
            hit = False
            for a in node.names:  # from pkg import submodule
                t = table.get(f"{node.module}.{a.name}")
                if t:
                    targets.add(t)
                    hit = True
            if not hit:
                t = resolve(node.module)
                if t:
                    targets.add(t)
    targets.discard(path)
    return targets


# ------------------------------------------------------------------ js / ts

def _load_tsconfigs(idx: RepoIndex) -> dict[str, list[tuple[str, list[str]]]]:
    """dir of tsconfig/jsconfig -> [(alias prefix, [target dirs])] incl. baseUrl ('' prefix)."""
    out: dict[str, list[tuple[str, list[str]]]] = {}
    for f in idx.files:
        if PurePosixPath(f).name not in ("tsconfig.json", "jsconfig.json") or "node_modules" in f:
            continue
        raw = _jsonc(read_text(idx.root, f) or "")
        try:
            opts = (json.loads(raw).get("compilerOptions") or {})
        except (ValueError, AttributeError):
            continue
        d = posixpath.dirname(f)
        base = posixpath.normpath(posixpath.join(d, opts.get("baseUrl") or "."))
        rules = []
        for alias, targets in (opts.get("paths") or {}).items():
            prefix = alias.split("*", 1)[0]
            dirs = [posixpath.normpath(posixpath.join(base, t.split("*", 1)[0])) for t in targets if isinstance(t, str)]
            rules.append((prefix, dirs))
        if opts.get("baseUrl"):
            rules.append(("", [base]))
        out[d] = rules
    return out


_JSONC_TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/|,(?=\s*[}\]])', re.S)


def _jsonc(raw: str) -> str:
    """JSON with comments / trailing commas (tsconfig) -> JSON; strings are left intact."""
    return _JSONC_TOKEN.sub(lambda m: m.group(0) if m.group(0).startswith('"') else "", raw)


def _alias_targets(path: str, spec: str, tsconfigs: dict) -> list[str]:
    d = posixpath.dirname(path)
    while True:
        if d in tsconfigs:
            outs = []
            for prefix, dirs in sorted(tsconfigs[d], key=lambda r: -len(r[0])):
                if spec.startswith(prefix) and (prefix or not spec.startswith("@")):
                    rest = spec[len(prefix):]
                    outs += [posixpath.normpath(posixpath.join(x, rest)) for x in dirs]
            return outs
        if not d:
            return []
        d = posixpath.dirname(d)


def _js_imports(path: str, text: str, fileset: set[str], tsconfigs: dict | None = None) -> set[str]:
    out = set()
    base = posixpath.dirname(path)
    code = "\n".join(_strip_js_comments(text))
    for m in _JS_IMPORT.finditer(code):
        spec = m.group(1)
        if not spec.startswith("."):
            for target in _alias_targets(path, spec, tsconfigs or {}):  # "@/lib/auth" -> lib/auth.ts
                hit = _js_resolve(target, fileset)
                if hit:
                    out.add(hit)
                    break
            continue
        target = posixpath.normpath(posixpath.join(base, spec))
        if target.startswith(".."):
            continue
        stem = target
        for ext in (".js", ".jsx", ".mjs", ".cjs"):  # TS/ESM: './core/Ky.js' refers to Ky.ts
            if target.endswith(ext):
                stem = target[: -len(ext)]
                break
        cands = [target] + [stem + e for e in _JS_TRY]
        for c in cands:
            c = posixpath.normpath(c)
            c = c[2:] if c.startswith("./") else c
            if c in fileset:
                out.add(c)
                break
        else:
            if target in (".", ""):  # require('../..') -> package root: package.json main / index.js
                root_main = _root_main(fileset)
                if root_main:
                    out.add(root_main)
    out.discard(path)
    return out




def _js_resolve(target: str, fileset: set[str]) -> str | None:
    stem = target
    for ext in (".js", ".jsx", ".mjs", ".cjs"):
        if target.endswith(ext):
            stem = target[: -len(ext)]
            break
    for c in [target] + [stem + e for e in _JS_TRY]:
        c = posixpath.normpath(c)
        c = c[2:] if c.startswith("./") else c
        if c in fileset:
            return c
    return None


def _root_main(fileset: set[str]) -> str | None:
    for c in ("index.js", "index.ts", "index.mjs", "src/index.ts", "src/index.js", "lib/index.js"):
        if c in fileset:
            return c
    return None


def _strip_js_comments(text: str) -> list[str]:
    # keep string contents (import specifiers live there): only drop comments
    no_block = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return [re.sub(r"(^|\s)//.*$", r"\1", l) for l in no_block.split("\n")]


# ------------------------------------------------------------------ java / kotlin

_JAVA_PKG = re.compile(r"^\s*package\s+([\w.]+)", re.M)
_JAVA_IMPORT = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+?)(\.\*)?\s*;?\s*$", re.M)


def _java_tables(idx: RepoIndex, files: list[str]):
    fqn: dict[str, str] = {}
    pkg_of: dict[str, str] = {}
    by_pkg: dict[str, list[str]] = defaultdict(list)
    for f in files:
        if config.language_of(f) not in ("java", "kotlin"):
            continue
        m = _JAVA_PKG.search(read_text(idx.root, f) or "")
        pkg = m.group(1) if m else ""
        pkg_of[f] = pkg
        by_pkg[pkg].append(f)
        fqn[(pkg + "." if pkg else "") + PurePosixPath(f).stem] = f
    return fqn, pkg_of, by_pkg


def _java_imports(path: str, text: str, tables) -> set[str]:
    fqn, pkg_of, by_pkg = tables
    out = set()
    for m in _JAVA_IMPORT.finditer(text):
        name = m.group(1)
        if m.group(2):  # wildcard: pkg.* -> classes of that package used in the code
            code = text
            for f in by_pkg.get(name, []):
                if re.search(rf"\b{re.escape(PurePosixPath(f).stem)}\b", code):
                    out.add(f)
            continue
        while name and name not in fqn:  # static imports / nested classes: a.b.C.member -> a.b.C
            name = name.rpartition(".")[0]
        if name:
            out.add(fqn[name])
    # same-package classes need no import: count the ones referenced in code
    code = "\n".join(mask_c_like(text, "java"))
    for f in by_pkg.get(pkg_of.get(path, ""), []):
        if f != path and re.search(rf"\b{re.escape(PurePosixPath(f).stem)}\b", code):
            out.add(f)
    out.discard(path)
    return out


# ------------------------------------------------------------------ rust

def _rust_tables(idx: RepoIndex, files: list[str]):
    """crate name -> (src dir, {module path -> file})."""
    crates: dict[str, tuple[str, dict[str, str]]] = {}
    for f in idx.files:
        if PurePosixPath(f).name != "Cargo.toml":
            continue
        m = re.search(r"^\[package\][^\[]*?^name\s*=\s*\"([^\"]+)\"", read_text(idx.root, f) or "", re.M | re.S)
        if not m:
            continue
        src = posixpath.join(posixpath.dirname(f), "src") if posixpath.dirname(f) else "src"
        mods: dict[str, str] = {}
        for rf in files:
            if not rf.endswith(".rs") or not rf.startswith(src + "/"):
                continue
            rel = rf[len(src) + 1:-3].split("/")
            if rel[-1] in ("lib", "main") and len(rel) == 1:
                mods[""] = mods.get("") or rf if rel[-1] == "main" else rf
                continue
            if rel[-1] == "mod":
                rel = rel[:-1]
            mods["::".join(rel)] = rf
        crates[m.group(1).replace("-", "_")] = (src, mods)
    return crates


def _rust_imports(path: str, text: str, crates) -> set[str]:
    own = None
    for name, (src, mods) in crates.items():
        if path.startswith(src + "/"):
            own = (name, src, mods)
            break
    if not own:
        return set()
    name, src, mods = own
    rel = path[len(src) + 1:-3].split("/")
    if rel[-1] in ("mod", "lib", "main"):
        rel = rel[:-1]
    here = rel  # module path of this file
    code = "\n".join(mask_c_like(text, "rust"))
    out = set()

    def add(crate_mods: dict[str, str], parts: list[str]) -> None:
        while True:
            key = "::".join(parts)
            if key in crate_mods:
                out.add(crate_mods[key])
                return
            if not parts:
                return
            parts = parts[:-1]

    for m in re.finditer(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;", code, re.M):
        add(mods, here + [m.group(1)])
    for m in re.finditer(r"\b(crate|super|self|\w+)::((?:\w+::)*\w+)", code):
        head, tail = m.group(1), m.group(2).split("::")
        if head == "crate":
            add(mods, tail)
        elif head == "super":
            add(mods, here[:-1] + tail)
        elif head == "self":
            add(mods, here + tail)
        elif head in crates and head != name:
            add(crates[head][1], tail)
    out.discard(path)
    return out


# ------------------------------------------------------------------ go

def _go_imports(text: str, module: str | None) -> set[str]:
    """Imported packages of the same module, as repo-relative directories ('' = root package)."""
    if not module:
        return set()
    specs = _GO_IMPORT_ONE.findall(text)
    for block in _GO_IMPORT_BLOCK.findall(text):
        specs += re.findall(r'"([^"]+)"', block)
    out = set()
    for s in specs:
        if s == module or s.startswith(module + "/"):
            out.add(s[len(module):].strip("/"))
    return out


def _go_node(d: str) -> str:
    return (d or "(root)") + "/"   # package node id: directory with trailing slash


# ------------------------------------------------------------------ graph

def build_graph(idx: RepoIndex, include_tests: bool = False) -> dict:
    """File-level graph. Go files are collapsed into package nodes ('dir/')."""
    files = [f for f in idx.files if config.is_code(f) and
             (include_tests or (not is_test_path(f) and not is_aux_path(f)))]
    fileset = set(files)
    allfiles = set(idx.files)
    py_table = _python_module_table(files)
    go_module = None
    for f in idx.files:
        if f == "go.mod" or (f.endswith("/go.mod") and go_module is None):
            m = re.search(r"^module\s+(\S+)", read_text(idx.root, f) or "", re.M)
            go_module = m.group(1) if m else None
            if f == "go.mod":
                break
    go_dirs = {posixpath.dirname(f) for f in files if f.endswith(".go")}
    tsconfigs = _load_tsconfigs(idx)
    java_tables = _java_tables(idx, files)
    rust_crates = _rust_tables(idx, files)

    nodes: set[str] = set()
    edges: Counter = Counter()
    for f in files:
        text = read_text(idx.root, f) or ""
        lang = config.language_of(f)
        if lang == "go":
            src = _go_node(posixpath.dirname(f))
            nodes.add(src)
            deps = {_go_node(d) for d in _go_imports(text, go_module) if d in go_dirs}
        else:
            src = f
            nodes.add(f)
            if lang == "python":
                deps = _py_imports(f, text, py_table, allfiles)
            elif lang in ("javascript", "typescript"):
                deps = _js_imports(f, text, fileset, tsconfigs)
            elif lang in ("java", "kotlin"):
                deps = _java_imports(f, text, java_tables)
            elif lang == "rust":
                deps = _rust_imports(f, text, rust_crates)
            else:
                deps = set()
        for d in deps:
            if d != src and (d in fileset or d.endswith("/")):
                edges[(src, d)] += 1
    return {"files": sorted(nodes), "edges": [{"src": a, "dst": b} for (a, b) in edges]}


def _group_of(node: str, level: str, depth: int) -> str:
    if level == "file":
        return node
    if node.endswith("/"):  # Go package
        parts = PurePosixPath(node.rstrip("/")).parts
        return "/".join(parts[:depth]) if node != "(root)/" else "(root)"
    parent = PurePosixPath(node).parent.parts
    return "/".join(parent[:depth]) or "(root)"


def aggregate(graph: dict, level: str = "auto", depth: int = 2, max_nodes: int = 45) -> dict:
    """Collapse the file graph to files or directories (level='auto' picks what fits)."""
    connected = {e["src"] for e in graph["edges"]} | {e["dst"] for e in graph["edges"]}
    if level == "auto":
        level = "file" if len(connected) <= max_nodes else "dir"
        if level == "dir":  # deepest directory level that still fits (deep Java/monorepo layouts)
            for d in range(depth, 10):
                n_groups = len({_group_of(f, "dir", d) for f in connected})
                if n_groups > max_nodes:
                    break
                depth = d
    nodes: Counter = Counter()
    for f in (connected if level == "file" else graph["files"]):
        nodes[_group_of(f, level, depth)] += 1
    w: Counter = Counter()
    for e in graph["edges"]:
        a, b = _group_of(e["src"], level, depth), _group_of(e["dst"], level, depth)
        if a != b:
            w[(a, b)] += 1
    truncated = 0
    if len(nodes) > max_nodes:  # keep the most connected nodes (aux code is already excluded)
        deg: Counter = Counter()
        for (a, b), n in w.items():
            deg[a] += n
            deg[b] += n
        keep = {n for n, _ in deg.most_common(max_nodes)}
        truncated = len(nodes) - len(keep)
        nodes = Counter({n: c for n, c in nodes.items() if n in keep})
        w = Counter({k: v for k, v in w.items() if k[0] in keep and k[1] in keep})
    fan_in, fan_out = Counter(), Counter()
    for (a, b), n in w.items():
        fan_out[a] += 1
        fan_in[b] += 1
    return {
        "level": level,
        "nodes": [{"id": n, "files": c, "fan_in": fan_in[n], "fan_out": fan_out[n]} for n, c in nodes.items()],
        "edges": [{"src": a, "dst": b, "weight": n} for (a, b), n in sorted(w.items())],
        "cycles": _cycles(w),
        "truncated_nodes": truncated,
    }


def _cycles(w: Counter) -> list[list[str]]:
    """Mutual (2-node) import cycles — the common, easy-to-explain smell."""
    pairs = set(w)
    return [list(p) for p in sorted({tuple(sorted((a, b))) for (a, b) in pairs if (b, a) in pairs})]


def _mid(name: str) -> str:
    return "n_" + re.sub(r"[^A-Za-z0-9_]", "_", name)


def _label(s: str) -> str:
    return s.replace('"', "'")


def to_mermaid(agg: dict) -> str:
    lines = ["flowchart LR"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for n in agg["nodes"]:
        g = posixpath.dirname(n["id"].rstrip("/")) if agg["level"] == "file" else ""
        groups[g].append(n)
    for g, ns in sorted(groups.items()):
        if g:
            lines.append(f'  subgraph {_mid("dir_" + g)}["{_label(g)}/"]')
        for n in ns:
            if agg["level"] == "file":
                label = PurePosixPath(n["id"].rstrip("/")).name + ("/" if n["id"].endswith("/") else "")
                extra = ""
            else:
                label, extra = n["id"], f"<br/>{n['files']} файлов"
            lines.append(f'    {_mid(n["id"])}["{_label(label)}{extra}"]')
        if g:
            lines.append("  end")
    for e in agg["edges"]:
        arrow = "-->" if e["weight"] == 1 else f'-- {e["weight"]} -->'
        lines.append(f"  {_mid(e['src'])} {arrow} {_mid(e['dst'])}")
    return "\n".join(lines)


def module_map(idx: RepoIndex, level: str = "auto", depth: int = 2, include_tests: bool = False) -> dict:
    g = build_graph(idx, include_tests=include_tests)
    agg = aggregate(g, level=level, depth=depth)
    agg["mermaid"] = to_mermaid(agg)
    agg["file_edges"] = len(g["edges"])
    return agg
