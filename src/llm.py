"""OpenAI-compatible chat client with tool calling.

Works with any provider that exposes the OpenAI Chat Completions API:
- Gemini (Google AI Studio): LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
- OpenAI:                    LLM_BASE_URL=https://api.openai.com/v1
- OpenRouter:                LLM_BASE_URL=https://openrouter.ai/api/v1
Configuration comes from environment variables or a local `.env` (never committed):
LLM_API_KEY (required), LLM_BASE_URL, LLM_MODEL, LLM_PRICE_IN_PER_M, LLM_PRICE_OUT_PER_M.
Errors are raised, never replaced by fake answers.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-2.5-flash"


def load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines); existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class OpenAICompatibleLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0, max_retries: int = 6):
        from openai import OpenAI
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        key = api_key or os.environ.get("LLM_API_KEY")
        if not key:
            raise RuntimeError("LLM_API_KEY is not set (put it in .env at the repo root)")
        self.model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
        self.temperature = temperature
        self.max_retries = max_retries
        self.price_in = float(os.environ.get("LLM_PRICE_IN_PER_M", "0") or 0)
        self.price_out = float(os.environ.get("LLM_PRICE_OUT_PER_M", "0") or 0)
        self.client = OpenAI(api_key=key, base_url=self.base_url, max_retries=0, timeout=120)

    def chat(self, messages: list[dict], tools: list[dict]) -> dict[str, Any]:
        """One completion. Retries only on rate-limit / transient server errors."""
        from openai import APIConnectionError, APIStatusError, RateLimitError
        delay = 5.0
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools,
                                                           tool_choice="auto", temperature=self.temperature)
                break
            except (RateLimitError, APIConnectionError) as e:
                err = e
            except APIStatusError as e:
                if e.status_code < 500:
                    raise
                err = e
            if attempt == self.max_retries:
                raise err
            time.sleep(delay)
            delay = min(delay * 2, 60)
        msg = resp.choices[0].message
        usage = getattr(resp, "usage", None)
        tin = getattr(usage, "prompt_tokens", 0) or 0
        tout = getattr(usage, "completion_tokens", 0) or 0
        calls = []
        for tc in msg.tool_calls or []:
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments})
        return {"content": msg.content or "", "tool_calls": calls, "finish_reason": resp.choices[0].finish_reason,
                "usage": {"prompt_tokens": tin, "completion_tokens": tout},
                "cost_usd": tin / 1e6 * self.price_in + tout / 1e6 * self.price_out,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "retries": attempt}
