"""Split files into retrievable chunks and extract symbol definitions."""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field

from . import config
from .textutil import mask_c_like, split_lines


@dataclass
class Chunk:
    id: int
    path: str
    start: int          # 1-based, inclusive
    end: int            # 1-based, inclusive
    kind: str           # "code" | "doc" | "config"
    symbols: list[str] = field(default_factory=list)
    text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Symbol:
    name: str
    qualname: str
    kind: str           # function | class | method | type
    path: str
    line: int           # first line incl. decorators
    end_line: int
    doc: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- symbols

def safe_parse(text: str):
    """ast.parse without SyntaxWarning noise from third-party code; None on syntax errors."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return ast.parse(text)
        except (SyntaxError, ValueError):
            return None


_JS_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "function", "else", "do", "try", "with",
                "super", "new", "typeof", "await", "yield", "throw", "delete", "import"}

_TS_TYPE = r"(?:\s*:\s*[^=;{]+?)?"
_JS_RULES: list[tuple[str, re.Pattern]] = [
    ("function", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:async\s+)?function\*?\s+([A-Za-z_$][\w$]*)")),
    ("class", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)")),
    # const f = (a: T): R => ..., const f = async <T>(x) => ..., const f = x => ..., const f = function
    ("function", re.compile(
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)" + _TS_TYPE +
        r"\s*=\s*(?:async\s*)?(?:function\b|(?:<[^>]*>\s*)?\([^)]*\)" + _TS_TYPE + r"\s*=>"
        r"|[A-Za-z_$][\w$]*\s*=>)")),
    # res.send = function send(...) / module.exports.x = function / proto.handle = async (...) =>
    ("function", re.compile(r"^\s*((?:[A-Za-z_$][\w$]*\.)+[A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>)")),
]
_TS_EXTRA = [("type", re.compile(r"^\s*(?:export\s+)?(?:declare\s+)?(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)"))]
# indented member: [modifiers] name<T>(args)[: type] {
_JS_METHOD = re.compile(
    r"^(\s+)(?:(?:public|private|protected|static|async|readonly|override|abstract|get|set)\s+)*"
    r"(#?[A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\([^)]*\)\s*(?::\s*[^{;=]+)?\s*\{\s*$")

_REGEX_SYMBOLS: dict[str, list[tuple[str, re.Pattern]]] = {
    "javascript": _JS_RULES,
    "typescript": _JS_RULES + _TS_EXTRA,
    "go": [
        ("function", re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)")),
        ("type", re.compile(r"^type\s+([A-Za-z_]\w*)\s+(?:struct|interface)")),
    ],
    "rust": [
        ("function", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_]\w*)")),
        ("type", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+([A-Za-z_]\w*)")),
    ],
    "java": [
        ("class", re.compile(r"^\s*(?:(?:public|private|protected|abstract|final|static|sealed)\s+)*(?:class|interface|enum|record)\s+([A-Za-z_]\w*)")),
        ("method", re.compile(r"^\s*(?:(?:public|private|protected|static|final|synchronized|abstract|default)\s+)+[\w<>\[\], ?]+\s+([a-z_]\w*)\s*\(")),
        # package-private methods: `int peek() {` (must open a body to avoid matching calls)
        ("method", re.compile(r"^\s+(?!return\b|new\b|else\b)[\w<>\[\],.?]+\s+([a-z_]\w*)\s*\([^)]*\)\s*(?:throws\s+[\w., ]+)?\{\s*$")),
    ],
    "ruby": [
        ("function", re.compile(r"^\s*def\s+(?:self\.)?([A-Za-z_]\w*[?!]?)")),
        ("class", re.compile(r"^\s*(?:class|module)\s+([A-Z]\w*)")),
    ],
    "php": [
        ("function", re.compile(r"^\s*(?:(?:public|private|protected|static|abstract|final)\s+)*function\s+([A-Za-z_]\w*)")),
        ("class", re.compile(r"^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait)\s+([A-Za-z_]\w*)")),
    ],
}
_REGEX_SYMBOLS["kotlin"] = [("function", re.compile(r"^\s*(?:\w+\s+)*fun\s+(?:<[^>]*>\s*)?(?:\w+\.)?([A-Za-z_]\w*)")),
                            ("class", re.compile(r"^\s*(?:\w+\s+)*(?:class|object|interface)\s+([A-Za-z_]\w*)"))]
_REGEX_SYMBOLS["csharp"] = _REGEX_SYMBOLS["java"]
_REGEX_SYMBOLS["scala"] = _REGEX_SYMBOLS["kotlin"]

# compound statements whose bodies may hold module/class-level definitions
_BLOCKS = (ast.If, ast.Try, ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While) + \
          ((ast.TryStar,) if hasattr(ast, "TryStar") else ())


def _python_symbols(path: str, text: str) -> list[Symbol]:
    tree = safe_parse(text)
    if tree is None:
        return []
    out: list[Symbol] = []

    def visit(nodes: list[ast.stmt], prefix: str, in_class: bool) -> None:
        for child in nodes:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                is_cls = isinstance(child, ast.ClassDef)
                kind = "class" if is_cls else ("method" if in_class else "function")
                qn = f"{prefix}{child.name}"
                start = min([d.lineno for d in child.decorator_list] + [child.lineno])
                doc = (ast.get_docstring(child) or "").strip().split("\n\n")[0][:300]
                out.append(Symbol(child.name, qn, kind, path, start,
                                  getattr(child, "end_lineno", child.lineno) or child.lineno, doc))
                if is_cls:  # descend into classes (methods), not into function bodies
                    visit(child.body, qn + ".", True)
            elif isinstance(child, _BLOCKS):  # `if sys.platform == ...: def f()`, try/except imports, ...
                for attr in ("body", "orelse", "finalbody"):
                    visit(getattr(child, attr, []) or [], prefix, in_class)
                for h in getattr(child, "handlers", []) or []:
                    visit(h.body, prefix, in_class)

    visit(tree.body, "", False)
    return out


def extract_symbols(path: str, text: str) -> list[Symbol]:
    lang = config.language_of(path)
    if lang == "python":
        return _python_symbols(path, text)
    pats = _REGEX_SYMBOLS.get(lang or "")
    if not pats:
        return []
    lines = split_lines(text)
    found: list[Symbol] = []
    cls_stack: list[tuple[int, str]] = []  # (indent, class name) for JS/TS methods
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        while cls_stack and indent <= cls_stack[-1][0] and not line.lstrip().startswith("}"):
            cls_stack.pop()
        matched = False
        for kind, pat in pats:
            m = pat.match(line)
            if m:
                full = m.group(1)
                name = full.rsplit(".", 1)[-1]
                found.append(Symbol(name, full, kind, path, i, i))
                if kind == "class" and lang in ("javascript", "typescript"):
                    cls_stack.append((indent, name))
                matched = True
                break
        if not matched and cls_stack and lang in ("javascript", "typescript"):
            m = _JS_METHOD.match(line)
            if m and m.group(2).lstrip("#") not in _JS_KEYWORDS and len(m.group(1)) > cls_stack[-1][0]:
                found.append(Symbol(m.group(2), f"{cls_stack[-1][1]}.{m.group(2)}", "method", path, i, i))
    # end lines: the matching closing brace for brace languages; otherwise up to the next definition
    for a, b in zip(found, found[1:]):
        a.end_line = max(a.line, b.line - 1)
    if found:
        found[-1].end_line = len(lines)
    if lang in _BRACE_LANGS:
        code = mask_c_like(text, lang)
        for sym in found:
            end = _brace_end(code, sym.line)
            if end:
                sym.end_line = end
    return found


_BRACE_LANGS = {"javascript", "typescript", "go", "rust", "java", "kotlin", "csharp", "scala", "php", "swift",
                "c", "cpp"}


def _brace_end(code: list[str], line: int, lookahead: int = 4) -> int | None:
    """Line of the brace closing the first '{' opened within `lookahead` lines of `line` (None if none)."""
    depth, opened = 0, False
    for i in range(line - 1, len(code)):
        row = code[i]
        if not opened and i - (line - 1) >= lookahead:
            return None
        if not opened and ";" in row and "{" not in row and i > line - 1:
            return None  # a declaration/expression without a body
        for ch in row:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}" and opened:
                depth -= 1
                if depth == 0:
                    return i + 1
    return None


# ----------------------------------------------------------------- chunks

def _windows(start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    spans, s = [], start
    while s <= end:
        e = min(end, s + size - 1)
        spans.append((s, e))
        if e == end:
            break
        s = e - overlap + 1
    return spans


def _units(symbols: list[Symbol], size: int) -> list[tuple[int, int]]:
    """Outermost definitions become units; a large one is split into its header + direct children."""
    syms = sorted(symbols, key=lambda x: (x.line, -x.end_line))

    def children(parent: Symbol) -> list[Symbol]:
        inner = [c for c in syms if c is not parent and parent.line <= c.line and c.end_line <= parent.end_line
                 and (c.line, c.end_line) != (parent.line, parent.end_line)]
        direct = []
        for c in inner:
            if not any(o is not c and o.line <= c.line and c.end_line <= o.end_line and
                       (o.line, o.end_line) != (c.line, c.end_line) for o in inner):
                direct.append(c)
        return direct

    def split(sym: Symbol) -> list[tuple[int, int]]:
        if sym.end_line - sym.line + 1 <= 2 * size:
            return [(sym.line, sym.end_line)]
        kids = children(sym)
        if not kids:
            return [(sym.line, sym.end_line)]
        out, cursor = [], sym.line
        for k in kids:
            if k.line < cursor:
                continue
            if k.line > cursor:
                out.append((cursor, k.line - 1))
            out += split(k)
            cursor = k.end_line + 1
        if cursor <= sym.end_line:
            out.append((cursor, sym.end_line))
        return out

    top, last_end = [], 0
    for x in syms:
        if x.line > last_end:
            top.append(x)
            last_end = x.end_line
    return [u for t in top for u in split(t)]


def chunk_file(path: str, text: str, symbols: list[Symbol], next_id: int) -> list[Chunk]:
    lines = split_lines(text)
    n = len(lines)
    if n == 0:
        return []
    lang = config.language_of(path)
    kind = "code" if config.is_code(path) else ("doc" if lang in ("markdown", "rst", "text") else "config")
    size, overlap = config.CHUNK_LINES, config.CHUNK_OVERLAP

    if symbols and kind == "code":
        # Structure-aware: chunks start at definitions (Python: exact AST spans; other languages:
        # from one definition to the next), gaps (imports, constants) become their own chunks.
        units = _units(symbols, size)
        spans: list[tuple[int, int]] = []
        cursor = 1
        for a, b in units:
            if b < cursor:
                continue
            a = max(a, cursor)
            if a > cursor:
                spans += _windows(cursor, a - 1, size, overlap)
            spans += _windows(a, b, 2 * size, overlap) if b - a + 1 > 2 * size else [(a, b)]
            cursor = b + 1
        if cursor <= n:
            spans += _windows(cursor, n, size, overlap)
        spans = [(a, b) for a, b in spans if any(lines[i - 1].strip() for i in range(a, b + 1))]
    else:
        wsize = size * 2 if kind == "doc" else size
        spans = _windows(1, n, wsize, overlap)

    chunks = []
    for a, b in spans:
        syms = [s.qualname for s in symbols if a <= s.line <= b or (s.line < a and s.end_line >= b)]
        chunks.append(Chunk(next_id + len(chunks), path, a, b, kind, syms[:12],
                            "\n".join(lines[a - 1:b])))
    return chunks
