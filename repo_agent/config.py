"""Central settings: repo_agent.toml (provider, model, limits) + environment overrides.

Secrets are never read from the config file: ANTHROPIC_API_KEY / GITHUB_TOKEN come from the environment.
Override any value with REPO_AGENT_<SECTION>_<KEY> (e.g. REPO_AGENT_LLM_MODEL) or point
REPO_AGENT_CONFIG at another TOML file.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

SUPPORTED_PROVIDERS = {"anthropic"}
_DEFAULTS = {
    "llm": {"provider": "anthropic", "model": "claude-opus-5-5", "chat_effort": "medium", "review_effort": "high",
            "timeout_seconds": 120, "max_retries": 3, "fallbacks": True},
    "agent": {"max_steps": 12, "max_tool_output_chars": 20000, "max_line_chars": 400, "tool_timeout_seconds": 20},
    "actions": {"post_to_github": False},
    "demo": {"fault": ""},
}


class ConfigError(ValueError):
    pass


def _coerce(value: str, like):
    if isinstance(like, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(like, int):
        return int(value)
    return value


def load_settings(path: str | Path | None = None) -> dict:
    path = Path(path or os.environ.get("REPO_AGENT_CONFIG") or Path(__file__).resolve().parent.parent / "repo_agent.toml")
    cfg = {k: dict(v) for k, v in _DEFAULTS.items()}
    if path.exists():
        data = tomllib.loads(path.read_text())
        for section, values in data.items():
            if section in cfg and isinstance(values, dict):
                for k, v in values.items():
                    if k in cfg[section]:
                        cfg[section][k] = v
    for section, values in cfg.items():  # environment overrides
        for k, v in values.items():
            env = os.environ.get(f"REPO_AGENT_{section.upper()}_{k.upper()}")
            if env is not None:
                values[k] = _coerce(env, v)
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
    return cfg


SETTINGS = load_settings()

# Where cloned repositories and their indexes live.
WORKSPACE = Path(os.environ.get("REPO_AGENT_WORKSPACE", Path.cwd() / "workspace")).resolve()

PROVIDER = SETTINGS["llm"]["provider"]
MODEL = SETTINGS["llm"]["model"]
CHAT_EFFORT = SETTINGS["llm"]["chat_effort"]
REVIEW_EFFORT = SETTINGS["llm"]["review_effort"]
LLM_TIMEOUT = float(SETTINGS["llm"]["timeout_seconds"])
LLM_MAX_RETRIES = int(SETTINGS["llm"]["max_retries"])
USE_FALLBACKS = bool(SETTINGS["llm"]["fallbacks"])

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
