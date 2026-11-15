"""Text helpers shared by indexing, citations, review and evals.

Line numbering everywhere follows git/GitHub/Python AST: lines are separated by "\\n" only
(str.splitlines() would also split on \\x0c, \\x1c-\\x1e, \\x85, \\u2028, \\u2029 and shift numbers).
"""

from __future__ import annotations

import io
import re
import tokenize

# language-tag directories with translated docs (fastapi docs/de, docs/zh-hant, ...)
LANG_CODES = set("""ar az bn de el em es fa fr he hi hu id it ja ko nl pl pt pt-br ru sv th tr uk ur vi yo
zh zh-cn zh-hans zh-hant zh-tw""".split())

AUX_DIRS = {"docs", "docs_src", "documentation", "examples", "example", "benchmarks", "tutorial", "tutorials"}


def split_lines(text: str) -> list[str]:
    """Split on '\\n' only (dropping a trailing '\\r' per line); a final newline adds no empty line."""
    if not text:
        return []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [l[:-1] if l.endswith("\r") else l for l in lines]


def line_count(text: str) -> int:
    return len(split_lines(text))


def is_test_path(path: str) -> bool:
    p = path.lower()
    name = p.rsplit("/", 1)[-1]
    return ("/test/" in f"/{p}" or "/tests/" in f"/{p}" or "/__tests__/" in f"/{p}" or "/testing/" in f"/{p}"
            or name.startswith("test_") or name.endswith(("_test.py", "_test.go")) or name == "conftest.py"
            or ".test." in name or ".spec." in name)


def is_aux_path(path: str) -> bool:
    """Documentation, examples, benchmarks — not production code."""
    parts = path.lower().split("/")
    return any(p in AUX_DIRS for p in parts[:-1])


def is_translation_path(path: str) -> bool:
    """docs/<lang>/..., i18n/<lang>/..., locale(s)/<lang>/... except English."""
    parts = path.lower().split("/")
    for i, p in enumerate(parts[:-1]):
        if i > 0 and p in LANG_CODES and parts[i - 1] in ("docs", "doc", "i18n", "locale", "locales", "translations"):
            return True
    return False


def mask_python(text: str) -> list[str] | None:
    """Per-line Python code with string literals and comments blanked (same length). None on tokenize error."""
    lines = split_lines(text)
    masked = [list(l) for l in lines]
    try:
        toks = list(tokenize.generate_tokens(io.StringIO("\n".join(lines) + "\n").readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    kinds = {tokenize.STRING, tokenize.COMMENT}
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END", "TSTRING_START", "TSTRING_MIDDLE", "TSTRING_END"):
        if hasattr(tokenize, name):
            kinds.add(getattr(tokenize, name))
    for tok in toks:
        if tok.type not in kinds:
            continue
        (sr, sc), (er, ec) = tok.start, tok.end
        for r in range(sr, er + 1):
            if r - 1 >= len(masked):
                break
            row = masked[r - 1]
            a = sc if r == sr else 0
            b = ec if r == er else len(row)
            for i in range(a, min(b, len(row))):
                row[i] = " "
    return ["".join(r) for r in masked]


_REGEX_PREV = set("=(,:[!&|?{};+-*%<>~^") | {""}
_REGEX_KW = {"return", "typeof", "case", "in", "of", "delete", "void", "throw", "new", "else", "do", "yield", "await"}


def mask_c_like(text: str, lang: str | None = None) -> list[str]:
    """JS/TS/Go/Java/Rust/C...: blank string/char/template/regex literals and comments (line structure kept).

    A small scanner (not a regex) so that regex literals (/["']/), template literals and Rust
    lifetimes ('a) don't swallow real code.
    """
    js = lang in (None, "javascript", "typescript")
    single_is_char = lang in ("rust", "go", "java", "kotlin", "c", "cpp", "csharp", "scala", "swift")
    out = list(text)
    n = len(text)
    i = 0
    prev = ""        # last significant non-space char (for regex-literal detection)
    prev_word = ""   # last identifier/keyword

    def blank(a: int, b: int) -> None:
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
            continue
        if c == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
            continue
        if c in "\"`" or (c == "'" and not single_is_char):
            if c == "`" and not js and lang != "go":
                i += 1
                continue
            j = i + 1
            while j < n:
                if text[j] == "\\" and c != "`" or (text[j] == "\\" and c == "`" and lang != "go"):
                    j += 2
                    continue
                if text[j] == c:
                    break
                if text[j] == "\n" and c != "`":
                    break
                j += 1
            blank(i + 1, j)
            i = j + 1
            prev, prev_word = c, ""
            continue
        if c == "'" and single_is_char:
            m = re.match(r"'(?:\\(?:u\{[0-9a-fA-F]+\}|x[0-9a-fA-F]{2}|.)|[^\\'\n])'", text[i:i + 12])
            if m:  # a real char literal; otherwise a Rust lifetime / label: leave it
                blank(i + 1, i + len(m.group(0)) - 1)
                i += len(m.group(0))
                prev, prev_word = "'", ""
                continue
            i += 1
            continue
        if js and c == "/" and (prev in _REGEX_PREV or prev_word in _REGEX_KW):
            j, in_class = i + 1, False
            while j < n and text[j] != "\n":
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "[":
                    in_class = True
                elif text[j] == "]":
                    in_class = False
                elif text[j] == "/" and not in_class:
                    break
                j += 1
            if j < n and text[j] == "/":
                blank(i + 1, j)
                i = j + 1
                prev, prev_word = "/", ""
                continue
        if c.isalnum() or c in "_$":
            j = i
            while j < n and (text[j].isalnum() or text[j] in "_$"):
                j += 1
            prev_word, prev = text[i:j], "a"
            i = j
            continue
        if not c.isspace():
            prev, prev_word = c, ""
        i += 1
    return split_lines("".join(out))


def mask_code(path: str, text: str) -> list[str]:
    """Best-effort code-only view of a file for regex rules (falls back to the raw lines)."""
    from . import config

    lang = config.language_of(path)
    if lang == "python":
        return mask_python(text) or split_lines(text)
    if lang in ("javascript", "typescript", "go", "java", "kotlin", "csharp", "c", "cpp", "rust", "swift",
                "scala", "php"):
        return mask_c_like(text, lang)
    return split_lines(text)
