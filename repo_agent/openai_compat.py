"""Provider adapter: any OpenAI-compatible Chat Completions API (e.g. Silk Compute, vLLM, OpenRouter).

It exposes the same tiny surface the agent uses from the Anthropic SDK — `client.beta.messages.create(**kw)`
returning an object with `.content` blocks (text / tool_use), `.stop_reason` and `.usage` — so the agent loop,
tool validation, confirmation and error handling are identical for every provider.
"""

from __future__ import annotations

import http.client
import json
import socket
import urllib.error
import urllib.request
from types import SimpleNamespace

_STOP = {"tool_calls": "tool_use", "function_call": "tool_use", "stop": "end_turn", "length": "max_tokens",
         "content_filter": "refusal"}


class ProviderError(RuntimeError):
    def __init__(self, message: str, kind: str = "api"):
        super().__init__(message)
        self.kind = kind  # auth | rate_limit | timeout | connection | bad_request | api


def _get(b, key, default=None):
    return b.get(key, default) if isinstance(b, dict) else getattr(b, key, default)


def _system_text(system) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    return "\n\n".join(_get(b, "text", "") for b in system)


def to_openai_messages(messages: list, system=None) -> list[dict]:
    out: list[dict] = []
    sys_text = _system_text(system)
    if sys_text:
        out.append({"role": "system", "content": sys_text})
    for m in messages:
        role, content = m["role"], m["content"]
        if isinstance(content, str):
            out.append({"role": "system" if role == "system" else role, "content": content})
            continue
        if role == "assistant":
            text = "".join(_get(b, "text", "") or "" for b in content if _get(b, "type") == "text")
            calls = [{"id": _get(b, "id"), "type": "function",
                      "function": {"name": _get(b, "name"),
                                   "arguments": json.dumps(_get(b, "input") or {}, ensure_ascii=False)}}
                     for b in content if _get(b, "type") == "tool_use"]
            msg = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        texts = []
        for b in content:  # user turn: tool results become role=tool messages, text stays user text
            if _get(b, "type") == "tool_result":
                c = _get(b, "content")
                c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
                if _get(b, "is_error"):
                    c = "[TOOL ERROR] " + c
                out.append({"role": "tool", "tool_call_id": _get(b, "tool_use_id"), "content": c})
            elif _get(b, "type") == "text":
                texts.append(_get(b, "text", ""))
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})
    return out


def to_openai_tools(tools: list | None) -> list[dict] | None:
    if not tools:
        return None
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": t["input_schema"]}} for t in tools]


def from_openai_response(d: dict):
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    blocks = []
    if msg.get("content"):
        blocks.append(SimpleNamespace(type="text", text=msg["content"]))
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            args = raw  # invalid JSON from the model -> the agent's schema validation reports it back
        blocks.append(SimpleNamespace(type="tool_use", id=tc.get("id") or f"call_{len(blocks)}",
                                      name=fn.get("name"), input=args))
    stop = _STOP.get(ch.get("finish_reason"), "end_turn")
    if stop == "end_turn" and any(b.type == "tool_use" for b in blocks):
        stop = "tool_use"
    u = d.get("usage") or {}
    usage = SimpleNamespace(input_tokens=u.get("prompt_tokens", 0), output_tokens=u.get("completion_tokens", 0),
                            cache_read_input_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    return SimpleNamespace(content=blocks, stop_reason=stop, stop_details=None, usage=usage, model=d.get("model"))


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 120, max_retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.beta = SimpleNamespace(messages=self)
        self.messages = self

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.api_key}",
                                              "Content-Type": "application/json", "User-Agent": "repo-agent"})
        last: ProviderError | None = None
        for _ in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 (configured base_url)
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode("utf-8", "replace")
                if e.code in (401, 403):
                    raise ProviderError(f"HTTP {e.code}: ключ отклонён ({detail})", "auth") from e
                if e.code == 400:
                    raise ProviderError(f"HTTP 400: {detail}", "bad_request") from e
                last = ProviderError(f"HTTP {e.code}: {detail}", "rate_limit" if e.code == 429 else "api")
                if e.code < 500 and e.code != 429:
                    raise last from e
            except (TimeoutError, socket.timeout) as e:
                last = ProviderError(f"сервис не ответил за {self.timeout:.0f} с", "timeout")
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError, OSError, ValueError) as e:
                if isinstance(getattr(e, "reason", None), (TimeoutError, socket.timeout)):
                    last = ProviderError(f"сервис не ответил за {self.timeout:.0f} с", "timeout")
                else:
                    last = ProviderError(f"нет соединения: {e}", "connection")
        raise last  # type: ignore[misc]

    def create(self, *, model, messages, max_tokens=4096, system=None, tools=None, tool_choice=None,
               output_config=None, **_ignored):
        body: dict = {"model": model, "messages": to_openai_messages(messages, system),
                      "max_tokens": min(int(max_tokens), 16000)}
        oai_tools = to_openai_tools(tools)
        if oai_tools:
            body["tools"] = oai_tools
            body["tool_choice"] = "none" if (tool_choice or {}).get("type") == "none" else "auto"
        fmt = (output_config or {}).get("format")
        if fmt and fmt.get("type") == "json_schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "result", "schema": fmt["schema"], "strict": True}}
            try:
                return from_openai_response(self._post(body))
            except ProviderError as e:
                if e.kind != "bad_request":
                    raise
                # provider without json_schema support: ask for JSON and describe the schema in the prompt
                body["response_format"] = {"type": "json_object"}
                body["messages"].insert(0, {"role": "system", "content":
                                            "Ответь ТОЛЬКО JSON-объектом по схеме: " + json.dumps(fmt["schema"])})
        return from_openai_response(self._post(body))
