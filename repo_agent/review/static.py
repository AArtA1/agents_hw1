"""Deterministic checks for common bugs and security problems (works offline).

Python gets AST-based checks; every language gets regex rules for dangerous APIs and secrets.
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass

from .. import config
from ..textutil import is_aux_path, is_test_path, mask_code, split_lines


@dataclass
class Finding:
    path: str
    line: int
    severity: str        # critical | high | medium | low
    category: str        # bug | security | performance | maintainability
    title: str
    explanation: str
    suggestion: str = ""
    rule: str = ""
    source: str = "static"
    confidence: float = 0.6
    end_line: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# ------------------------------------------------------------------ regex rules
# (rule id, languages or None for all, regex, severity, category, title, suggestion)
REGEX_RULES = [
    ("secret-aws-key", None, r"AKIA[0-9A-Z]{16}", "critical", "security",
     "Похоже на AWS access key в коде", "Удалите ключ, отзовите его и берите секрет из окружения/хранилища."),
    ("secret-private-key", None, r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----", "critical", "security",
     "Приватный ключ в репозитории", "Удалите ключ из истории и выпустите новый."),
    ("secret-github-token", None, r"\bgh[pousr]_[A-Za-z0-9]{36,}\b", "critical", "security",
     "GitHub token в коде", "Отзовите токен и используйте секреты CI."),
    ("secret-hardcoded", None,
     r"""(?i)\b\w*?(?:password|passwd|pwd|secret|api_?key|access_?token|auth_?token)\b['"]?\s*[:=]\s*['"][^'"\s]{6,}['"]""",
     "high", "security", "Захардкоженный секрет/пароль", "Читайте значение из переменных окружения или хранилища секретов."),
    ("js-eval", ("javascript", "typescript"), r"\beval\s*\(|new\s+Function\s*\(", "high", "security",
     "eval / new Function — выполнение произвольного кода", "Избегайте eval; парсите данные явно (JSON.parse и т.п.)."),
    ("js-innerhtml", ("javascript", "typescript"), r"\.innerHTML\s*=|dangerouslySetInnerHTML|document\.write\s*\(",
     "medium", "security", "Вставка HTML без экранирования (XSS)", "Используйте textContent или санитайзер (DOMPurify)."),
    ("js-child-exec", ("javascript", "typescript"), r"child_process.*\bexec(?:Sync)?\s*\(|\bexec(?:Sync)?\s*\(\s*`",
     "high", "security", "Запуск shell-команды со строкой (command injection)", "Используйте execFile/spawn с массивом аргументов."),
    ("tls-verify-off", None, r"verify\s*=\s*False|rejectUnauthorized\s*:\s*false|InsecureSkipVerify\s*:\s*true",
     "high", "security", "Отключена проверка TLS-сертификата", "Не отключайте проверку сертификатов; укажите CA bundle."),
    ("weak-hash", None, r"\b(?:hashlib\.)?(?:md5|sha1)\s*\(|createHash\(\s*['\"](?:md5|sha1)['\"]",
     "low", "security", "Слабая хэш-функция (MD5/SHA1)", "Для безопасности используйте SHA-256+/bcrypt/argon2."),
    ("sql-concat", None,
     r"""(?i)\b(?:execute|executemany|query|raw)\s*\(\s*(?:f['"]|['"][^'"]*\b(?:select|insert|update|delete)\b[^'"]*['"]\s*(?:%|\+|\.format))""",
     "high", "security", "SQL-запрос собирается строкой (SQL injection)", "Используйте параметризованные запросы."),
    ("go-sql-sprintf", ("go",), r"(?:Query|Exec)(?:Context)?\([^)]*fmt\.Sprintf", "high", "security",
     "SQL-запрос через fmt.Sprintf (SQL injection)", "Используйте плейсхолдеры ($1, ?)."),
    # Java / Kotlin
    ("java-runtime-exec", ("java", "kotlin"), r"Runtime\.getRuntime\(\)\.exec\s*\(|new\s+ProcessBuilder\s*\(\s*\"(?:sh|bash|cmd)",
     "high", "security", "Запуск внешней команды (command injection)", "Не передавайте внешние данные в команду; используйте массив аргументов."),
    ("java-deserialization", ("java", "kotlin"), r"new\s+ObjectInputStream\s*\(|\.readObject\s*\(\s*\)",
     "medium", "security", "Java-десериализация (возможен RCE на недоверенных данных)", "Используйте JSON или фильтр ObjectInputFilter."),
    ("java-weak-hash", ("java", "kotlin"), r"MessageDigest\.getInstance\(\s*\"(?:MD5|SHA-?1)\"", "low", "security",
     "Слабая хэш-функция (MD5/SHA1)", "Используйте SHA-256+."),
    ("java-insecure-random", ("java", "kotlin"), r"new\s+Random\s*\(\s*\).*(?:token|password|secret|key)", "medium", "security",
     "java.util.Random для секретов", "Используйте SecureRandom."),
    ("java-trust-all", ("java", "kotlin"), r"X509TrustManager|setHostnameVerifier\s*\(\s*\(?.*->\s*true", "medium", "security",
     "Возможно отключена проверка TLS-сертификатов", "Не используйте trust-all TrustManager/HostnameVerifier."),
    # Rust
    ("rust-tls-off", ("rust",), r"danger_accept_invalid_(?:certs|hostnames)\s*\(\s*true", "high", "security",
     "Отключена проверка TLS-сертификата", "Не отключайте проверку сертификатов."),
    ("rust-shell", ("rust",), r"Command::new\(\s*\"(?:sh|bash|cmd)\"\s*\)", "medium", "security",
     "Запуск команды через shell", "Запускайте программу напрямую с аргументами."),
    ("rust-mem-transmute", ("rust",), r"\bmem::transmute\b|\bget_unchecked(?:_mut)?\s*\(", "low", "bug",
     "Небезопасная операция (transmute/get_unchecked)", "Проверьте инварианты или используйте безопасную альтернативу."),
]
# rules that must see string contents (SQL text, secret values); the match must still START in code
_RAW_RULES = {"secret-aws-key", "secret-private-key", "secret-github-token", "secret-hardcoded", "sql-concat",
              "go-sql-sprintf", "java-runtime-exec", "java-weak-hash", "rust-shell", "java-insecure-random"}

_COMPILED = [(r[0], r[1], re.compile(r[2]), *r[3:]) for r in REGEX_RULES]

DANGEROUS_CALLS = {
    "eval": ("medium", "security", "eval() выполняет произвольный код", "Используйте ast.literal_eval или явный парсинг."),
    "exec": ("medium", "security", "exec() выполняет произвольный код", "Избегайте exec для внешних данных."),
    "os.system": ("high", "security", "os.system запускает shell (command injection)", "subprocess.run([...]) без shell."),
    "os.popen": ("high", "security", "os.popen запускает shell", "subprocess.run([...]) без shell."),
    "pickle.loads": ("high", "security", "Десериализация pickle из недоверенных данных = RCE", "Используйте JSON."),
    "pickle.load": ("medium", "security", "pickle.load из недоверенного источника = RCE", "Используйте JSON или подпись данных."),
    "marshal.loads": ("high", "security", "marshal.loads из недоверенных данных", "Используйте JSON."),
    "tempfile.mktemp": ("medium", "security", "tempfile.mktemp подвержен race condition", "tempfile.mkstemp / NamedTemporaryFile."),
    "random.random": ("low", "security", "random не криптостойкий", "Для токенов используйте secrets."),
}


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


class _PyVisitor(ast.NodeVisitor):
    def __init__(self, path: str):
        self.path = path
        self.out: list[Finding] = []

    def add(self, node, sev, cat, title, expl, sugg, rule, conf=0.6):
        self.out.append(Finding(self.path, getattr(node, "lineno", 1), sev, cat, title, expl, sugg, rule,
                                confidence=conf, end_line=getattr(node, "end_lineno", None)))

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        body_is_pass = all(isinstance(s, (ast.Pass, ast.Continue)) or
                           (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)) for s in node.body)
        reraises = bool(node.body) and isinstance(node.body[-1], ast.Raise) and node.body[-1].exc is None
        if node.type is None and not reraises:  # `except: cleanup(); raise` is a legitimate idiom
            self.add(node, "medium", "bug", "Голый except:",
                     "Перехватывает всё, включая KeyboardInterrupt/SystemExit, и скрывает ошибки.",
                     "Ловите конкретные исключения (except ValueError: ...).", "py-bare-except")
        elif body_is_pass and _dotted(node.type) in ("Exception", "BaseException"):
            self.add(node, "medium", "bug", "Исключение молча проглатывается",
                     "except Exception: pass скрывает ошибки и усложняет отладку.",
                     "Логируйте исключение или ловите конкретный тип.", "py-swallow-exception")
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        for d in node.args.defaults + node.args.kw_defaults:
            if isinstance(d, (ast.List, ast.Dict, ast.Set)) or (
                    isinstance(d, ast.Call) and _dotted(d.func) in ("list", "dict", "set")):
                self.add(d, "medium", "bug", f"Изменяемое значение по умолчанию в {node.name}()",
                         "Объект по умолчанию создаётся один раз и разделяется между вызовами.",
                         "Используйте None и создавайте объект внутри функции.", "py-mutable-default")
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Compare(self, node: ast.Compare):
        for op, right in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Is, ast.IsNot)) and isinstance(right, ast.Constant) and \
                    isinstance(right.value, (str, int, float, bytes)) and not isinstance(right.value, bool):
                self.add(node, "medium", "bug", "Сравнение с литералом через `is`",
                         "`is` сравнивает идентичность объектов, а не значения — результат зависит от интернирования.",
                         "Используйте == / !=.", "py-is-literal", 0.8)
            if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(right, ast.Constant) and right.value is None:
                self.add(node, "low", "bug", "Сравнение с None через ==",
                         "__eq__ может быть переопределён; идиоматично `is None`.", "Используйте `is None`.",
                         "py-eq-none", 0.5)
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.Is)) and \
                ast.dump(node.left) == ast.dump(node.comparators[0]):
            self.add(node, "high", "bug", "Сравнение выражения с самим собой",
                     "Условие всегда истинно/ложно — вероятно, опечатка.", "Проверьте операнды.", "py-self-compare", 0.8)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1 and ast.dump(node.targets[0]) == ast.dump(node.value) and \
                isinstance(node.value, (ast.Name, ast.Attribute)):
            self.add(node, "medium", "bug", "Присваивание переменной самой себе",
                     "Не имеет эффекта — вероятно, опечатка.", "Проверьте, что именно хотели присвоить.",
                     "py-self-assign", 0.8)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict):
        seen = set()
        for k in node.keys:
            if isinstance(k, ast.Constant):
                if k.value in seen:
                    self.add(k, "medium", "bug", f"Дублирующийся ключ словаря {k.value!r}",
                             "Более раннее значение молча перезаписывается.", "Удалите дубликат.", "py-dup-key", 0.8)
                seen.add(k.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        name = _dotted(node.func)
        if name in DANGEROUS_CALLS:
            sev, cat, title, sugg = DANGEROUS_CALLS[name]
            self.add(node, sev, cat, title, f"Вызов {name}().", sugg, "py-" + name.replace(".", "-"), 0.5)
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        if name.startswith("subprocess.") and isinstance(kw.get("shell"), ast.Constant) and kw["shell"].value is True:
            self.add(node, "high", "security", "subprocess с shell=True",
                     "Если в команду попадают внешние данные — command injection.",
                     "Передавайте список аргументов без shell=True.", "py-subprocess-shell", 0.7)
        if name in ("yaml.load", "yaml.load_all") and "Loader" not in kw and len(node.args) < 2:
            self.add(node, "high", "security", "yaml.load без безопасного Loader",
                     "Может создавать произвольные объекты Python.", "yaml.safe_load(...)", "py-yaml-load", 0.8)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert):
        if isinstance(node.test, ast.Tuple) and node.test.elts:
            self.add(node, "high", "bug", "assert с кортежем всегда истинен",
                     "assert (cond, 'msg') проверяет непустой кортеж.", "assert cond, 'msg'", "py-assert-tuple", 0.9)
        self.generic_visit(node)


def _is_test(path: str) -> bool:
    p = path.lower()
    return "/test" in p or p.startswith("test") or "_test." in p or ".spec." in p


def check_text(path: str, text: str) -> list[Finding]:
    lang = config.language_of(path)
    out: list[Finding] = []
    if lang == "python":
        try:
            tree = ast.parse(text)
            v = _PyVisitor(path)
            v.visit(tree)
            out += v.out
        except (SyntaxError, ValueError) as e:
            out.append(Finding(path, getattr(e, "lineno", 1) or 1, "high", "bug", "Синтаксическая ошибка",
                               str(e), "", "py-syntax", confidence=0.9))
    if lang in (None, "markdown", "rst", "text"):
        return out
    raw_lines = split_lines(text)
    code_lines = mask_code(path, text)  # strings and comments blanked: API rules must not fire inside them
    for i, line in enumerate(raw_lines, 1):
        code = code_lines[i - 1] if i - 1 < len(code_lines) else line
        stripped = line.strip()
        for rid, langs, rx, sev, cat, title, sugg in _COMPILED:
            if langs and lang not in langs:
                continue
            if rid == "weak-hash" and "usedforsecurity=False" in line:
                continue
            if rid in _RAW_RULES:
                m = rx.search(line)
                # anchored in code (not inside a comment/string), except key/token material anywhere
                if m and rid not in ("secret-aws-key", "secret-private-key", "secret-github-token") and \
                        (m.start() >= len(code) or code[m.start()] == " "):
                    m = None
            else:
                m = rx.search(code)
            if not m:
                continue
            if rid == "secret-hardcoded":
                if _placeholder_secret(m.group(0)):
                    continue
                if not _high_entropy(m.group(0)):
                    sev = "medium"
            out.append(Finding(path, i, sev, cat, title, f"Строка: `{stripped[:160]}`", sugg, rid,
                               confidence=0.5))
    # suppressions: `# noqa`, `# nosec`, `// NOSONAR`, eslint-disable-line
    out = [f for f in out if not _suppressed(raw_lines, f.line)]
    # findings in tests, docs and examples are much less important
    if is_test_path(path) or is_aux_path(path):
        for f in out:
            f.severity = "low"
            f.confidence = min(f.confidence, 0.3)
    return _dedupe(out)


_SUPPRESS = re.compile(r"#\s*(?:noqa|nosec)\b|NOSONAR|eslint-disable(?:-next)?-line|nolint", re.I)
_PLACEHOLDER = re.compile(r"(?i)example|sample|dummy|changeme|change_me|placeholder|your[_-]|xxx+|\*\*\*|<[^>]*>|"
                          r"\$\{|\{\{|%s|test|fake|secret123|password123|^pass(word)?$")


def _suppressed(lines: list[str], line_no: int) -> bool:
    cur = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
    prev = lines[line_no - 2] if line_no >= 2 else ""
    return bool(_SUPPRESS.search(cur)) or "eslint-disable-next-line" in prev or "nosec" in prev.lower()


def _secret_parts(match: str) -> tuple[str, str]:
    m = re.match(r"""(?i)\W*([\w]+)\W*\s*[:=]\s*['"]([^'"]*)['"]""", match)
    return (m.group(1), m.group(2)) if m else ("", match)


def _placeholder_secret(match: str) -> bool:
    key, val = _secret_parts(match)
    norm = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
    if norm(val) == norm(key) or norm(val) in ("password", "secret", "apikey", "token", "none", "null"):
        return True  # enum/constant names like apiKey = "apiKey"
    return bool(_PLACEHOLDER.search(val))


def _high_entropy(match: str) -> bool:
    _, val = _secret_parts(match)
    classes = sum(bool(re.search(p, val)) for p in (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]"))
    return len(val) >= 16 and classes >= 3


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen, out = set(), []
    for f in findings:
        key = (f.path, f.line, f.rule)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def check_repo(idx, paths: list[str] | None = None) -> list[Finding]:
    from ..repo import read_text

    out: list[Finding] = []
    for f in paths or idx.files:
        if not config.is_code(f):
            continue
        text = read_text(idx.root, f)
        if text:
            out += check_text(f, text)
    return sorted(out, key=lambda x: (SEV_ORDER.get(x.severity, 9), x.path, x.line))
