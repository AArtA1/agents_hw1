"""Thin wrapper around the Anthropic SDK (Claude Opus 5.5 by default).

* Server-side refusal fallback is on by default (`fallbacks="default"`): if the safety
  classifier declines a request, the API transparently retries it on a fallback model.
  Disable with REPO_AGENT_FALLBACKS=0.
* Top-level automatic prompt caching: the stable system prompt + tool list are cached,
  so a multi-step tool loop doesn't pay full price for the repo summary on every step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import config

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    pass


def has_credentials() -> bool:
    """Best-effort check so the app can degrade to retrieval-only mode instead of crashing."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    if os.environ.get("ANTHROPIC_PROFILE") or os.environ.get("ANTHROPIC_IDENTITY_TOKEN_FILE"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


class LLM:
    def __init__(self, model: str | None = None, client: Any = None):
        self.model = model or config.MODEL
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=config.LLM_TIMEOUT, max_retries=config.LLM_MAX_RETRIES)
        self.client = client
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "calls": 0}

    def create(self, *, messages: list, system: Any = None, tools: list | None = None,
               effort: str = "medium", max_tokens: int = 16000, output_format: dict | None = None,
               tool_choice: dict | None = None):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "cache_control": {"type": "ephemeral"},
            "output_config": {"effort": effort},
        }
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if output_format:
            kwargs["output_config"]["format"] = output_format
        if config.USE_FALLBACKS:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        resp = self._call(kwargs)
        self._track(resp)
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            cat = getattr(details, "category", None) if details else None
            raise LLMError(f"Модель отказалась отвечать (refusal, category={cat}).")
        return resp

    def _call(self, kwargs: dict):
        import anthropic

        try:
            return self.client.beta.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise LLMError("Нет доступа к Claude API: задайте ANTHROPIC_API_KEY.") from e
        except anthropic.BadRequestError as e:
            raise LLMError(f"Некорректный запрос к API: {e.message}") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Превышен лимит запросов к API, попробуйте позже.") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Ошибка API ({e.status_code}): {e.message}") from e
        except anthropic.APITimeoutError as e:
            raise LLMError(f"Claude API не ответил за {config.LLM_TIMEOUT:.0f} с (таймаут).") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Нет соединения с Claude API.") from e

    def _track(self, resp) -> None:
        u = getattr(resp, "usage", None)
        self.usage["calls"] += 1
        if u is None:
            return
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
            self.usage[k] += getattr(u, k, 0) or 0

    @staticmethod
    def text_of(resp) -> str:
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()

    def complete(self, prompt: str, system: str | None = None, effort: str = "medium",
                 max_tokens: int = 16000) -> str:
        resp = self.create(messages=[{"role": "user", "content": prompt}], system=system,
                           effort=effort, max_tokens=max_tokens)
        return self.text_of(resp)

    def json(self, prompt: str, schema: dict, system: str | None = None, effort: str = "high",
             max_tokens: int = 16000) -> dict:
        """Structured output: the response is guaranteed to match `schema`."""
        resp = self.create(messages=[{"role": "user", "content": prompt}], system=system,
                           effort=effort, max_tokens=max_tokens,
                           output_format={"type": "json_schema", "schema": schema})
        if resp.stop_reason == "max_tokens":
            raise LLMError("Ответ обрезан по max_tokens — уменьшите объём входа.")
        return json.loads(self.text_of(resp))
