"""Central settings: repo_agent.toml (provider, model, limits) + environment overrides.

Secrets are never read from the config file: ANTHROPIC_API_KEY / GITHUB_TOKEN come from the environment.
Override any value with REPO_AGENT_<SECTION>_<KEY> (e.g. REPO_AGENT_LLM_MODEL) or point
REPO_AGENT_CONFIG at another TOML file.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

SUPPORTED_PROVIDERS = {"anthropic", "openai_compatible"}
_DEFAULTS = {
    "llm": {"provider": "anthropic", "model": "claude-opus-5-5", "chat_effort": "medium", "review_effort": "high",
            "timeout_seconds": 120, "max_retries": 3, "fallbacks": True, "base_url": "", "api_key_env": ""},
    "agent": {"max_steps": 12, "max_tool_output_chars": 20000, "max_line_chars": 400, "tool_timeout_seconds": 20},
    "actions": {"post_to_github": False},
    "demo": {"fault": ""},
}


class ConfigError(ValueError):
    pass


PROJECT_DIR = Path(__file__).resolve().parent.parent
EFFORTS = {"low", "medium", "high", "xhigh", "max"}


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE lines); never overrides variables already set in the environment."""
    for p in ([path] if path else [Path.cwd() / ".env", PROJECT_DIR / ".env"]):
        if not p or not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip().removeprefix("export ").strip(), val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val
        return


def _coerce(name: str, value: str, like):
    if isinstance(like, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(like, int):
        try:
            return int(value)
        except ValueError:
            raise ConfigError(f"{name}={value!r}: ожидается целое число") from None
    return value


def load_settings(path: str | Path | None = None) -> dict:
    explicit = path or os.environ.get("REPO_AGENT_CONFIG")
    path = Path(explicit or PROJECT_DIR / "repo_agent.toml")
    if explicit and not path.exists():
        raise ConfigError(f"Файл конфигурации не найден: {path}")
    cfg = {k: dict(v) for k, v in _DEFAULTS.items()}
    if path.exists():
        try:
            data = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"Ошибка в {path}: {e}") from None
        for section, values in data.items():
            if section in cfg and isinstance(values, dict):
                for k, v in values.items():
                    if k in cfg[section]:
                        cfg[section][k] = v
    for section, values in cfg.items():  # environment overrides
        for k, v in values.items():
            name = f"REPO_AGENT_{section.upper()}_{k.upper()}"
            env = os.environ.get(name)
            if env is not None:
                values[k] = _coerce(name, env, v)
    # backwards-compatible short env names
    for env, (section, k) in {"REPO_AGENT_MODEL": ("llm", "model"), "REPO_AGENT_CHAT_EFFORT": ("llm", "chat_effort"),
                              "REPO_AGENT_REVIEW_EFFORT": ("llm", "review_effort")}.items():
        if os.environ.get(env):
            cfg[section][k] = os.environ[env]
    if os.environ.get("REPO_AGENT_FALLBACKS") is not None:
        cfg["llm"]["fallbacks"] = os.environ["REPO_AGENT_FALLBACKS"] != "0"
    if cfg["llm"]["provider"] not in SUPPORTED_PROVIDERS:
        raise ConfigError(f"Провайдер {cfg['llm']['provider']!r} не поддерживается; доступно: {sorted(SUPPORTED_PROVIDERS)}")
    if int(cfg["agent"]["max_steps"]) < 1:
        raise ConfigError("agent.max_steps должен быть >= 1")
    # TOML values must have the default's type (e.g. post_to_github = "false" is a string, not a bool)
    for section, values in cfg.items():
        for k, v in values.items():
            d = _DEFAULTS[section][k]
            if isinstance(d, bool) and not isinstance(v, bool):
                raise ConfigError(f"{section}.{k} должен быть true/false, получено {v!r}")
            if isinstance(d, int) and not isinstance(d, bool) and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
                raise ConfigError(f"{section}.{k} должен быть положительным целым, получено {v!r}")
    if cfg["llm"]["provider"] == "openai_compatible" and not (cfg["llm"]["base_url"] and cfg["llm"]["api_key_env"]):
        raise ConfigError("для provider = \"openai_compatible\" нужны llm.base_url и llm.api_key_env")
    fault = cfg["demo"]["fault"]
    if fault and not re.fullmatch(r"[a-z_]+:(timeout|error)", fault):
        raise ConfigError(f"demo.fault={fault!r}: формат <инструмент>:<timeout|error>")
    for k in ("chat_effort", "review_effort"):
        if cfg["llm"][k] not in EFFORTS:
            raise ConfigError(f"llm.{k}={cfg['llm'][k]!r}: допустимо {sorted(EFFORTS)}")
    return cfg


load_dotenv()
CONFIG_ERROR: str | None = None
try:
    SETTINGS = load_settings()
except ConfigError as _e:  # reported by the CLI / UI as a short message instead of a traceback
    CONFIG_ERROR = str(_e)
    SETTINGS = {k: dict(v) for k, v in _DEFAULTS.items()}

# Where cloned repositories and their indexes live.
WORKSPACE = Path(os.environ.get("REPO_AGENT_WORKSPACE", Path.cwd() / "workspace")).resolve()

PROVIDER = SETTINGS["llm"]["provider"]
MODEL = SETTINGS["llm"]["model"]
CHAT_EFFORT = SETTINGS["llm"]["chat_effort"]
REVIEW_EFFORT = SETTINGS["llm"]["review_effort"]
LLM_TIMEOUT = float(SETTINGS["llm"]["timeout_seconds"])
LLM_MAX_RETRIES = int(SETTINGS["llm"]["max_retries"])
USE_FALLBACKS = bool(SETTINGS["llm"]["fallbacks"]) and PROVIDER == "anthropic"
BASE_URL = SETTINGS["llm"]["base_url"]
API_KEY_ENV = SETTINGS["llm"]["api_key_env"] or ("ANTHROPIC_API_KEY" if PROVIDER == "anthropic" else "")

MAX_STEPS = int(SETTINGS["agent"]["max_steps"])
MAX_TOOL_OUTPUT = int(SETTINGS["agent"]["max_tool_output_chars"])
MAX_LINE_CHARS = int(SETTINGS["agent"]["max_line_chars"])
TOOL_TIMEOUT = float(SETTINGS["agent"]["tool_timeout_seconds"])
POST_ISSUES_TO_GITHUB = bool(SETTINGS["actions"]["post_to_github"])
DEMO_FAULT = str(SETTINGS["demo"]["fault"] or "")

# Indexing limits
MAX_FILE_BYTES = 400_000          # skip larger files (generated / data)
CHUNK_LINES = 60                  # window size for non-structured chunking
CHUNK_OVERLAP = 15
MAX_COMMITS = 300                 # how much history to index

SKIP_DIRS = {
    ".git", "node_modules", "vendor", "third_party", "dist", "build", "__pycache__",
    ".venv", "venv", "env", ".tox", ".mypy_cache", ".pytest_cache", ".idea", ".vscode",
    "target", "out", ".next", "coverage", "site-packages", ".eggs",
}

CODE_EXT = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".scala": "scala",
    ".rb": "ruby", ".php": "php", ".cs": "csharp", ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".swift": "swift",
    ".sh": "shell", ".bash": "shell", ".sql": "sql",
}
DOC_EXT = {".md": "markdown", ".rst": "rst", ".txt": "text"}
CONFIG_EXT = {
    ".toml": "toml", ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".cfg": "ini",
    ".ini": "ini", ".gradle": "gradle", ".xml": "xml",
}
SPECIAL_FILES = {
    "Dockerfile": "docker", "Makefile": "make", "Procfile": "procfile",
    "Gemfile": "ruby", "Pipfile": "toml", "go.mod": "gomod", "Cargo.toml": "toml",
}


def language_of(path: str) -> str | None:
    name = Path(path).name
    if name in SPECIAL_FILES:
        return SPECIAL_FILES[name]
    ext = Path(path).suffix.lower()
    return CODE_EXT.get(ext) or DOC_EXT.get(ext) or CONFIG_EXT.get(ext)


def is_code(path: str) -> bool:
    return Path(path).suffix.lower() in CODE_EXT
