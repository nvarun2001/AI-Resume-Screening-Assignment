"""Thin provider-agnostic wrapper around LLM structured-output calls.

Callers ask for a Pydantic type back and never touch provider SDKs directly.
Two providers are supported:

- ``anthropic``: native Claude via the anthropic SDK, using tool-use to force
  schema-conforming JSON.
- ``openai``: any OpenAI-compatible chat-completions endpoint via httpx (no
  openai SDK dependency), using JSON response mode.

Provider specifics live behind ``_invoke`` so the retry/validation logic is
shared and tests can inject a fake transport instead of hitting the network.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional, TypeVar

from pydantic import BaseModel

from . import config
from .config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# transport(system_prompt, user_prompt, schema_json) -> parsed dict
Transport = Callable[[str, str, dict], dict]


class LLMError(Exception):
    """Raised when a model call fails after retries or is misconfigured."""


class LLMClient:
    def __init__(self, settings: Optional[Settings] = None, transport: Optional[Transport] = None):
        self.settings = settings or Settings.from_env()
        # Injected in tests to avoid real network calls.
        self._transport = transport

    def extract_structured(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
        """Call the model and validate its output against ``schema``.

        Retries once on failure (transient errors or malformed output). Raises
        ``LLMError`` if both attempts fail so callers can degrade gracefully.
        """

        schema_json = schema.model_json_schema()
        last_error: Optional[Exception] = None
        for attempt in range(2):
            try:
                raw = self._invoke(system_prompt, user_prompt, schema_json)
                return schema.model_validate(raw)
            except Exception as exc:  # noqa: BLE001 - any provider error must degrade, not crash the batch
                last_error = exc
                logger.warning("LLM call attempt %d failed: %s", attempt + 1, exc)
        raise LLMError(f"LLM extraction failed after retries: {last_error}") from last_error

    def _invoke(self, system_prompt: str, user_prompt: str, schema_json: dict) -> dict:
        if self._transport is not None:
            return self._transport(system_prompt, user_prompt, schema_json)

        if not self.settings.llm_configured:
            raise LLMError("LLM is not configured (set LLM_PROVIDER, LLM_MODEL, LLM_API_KEY)")

        provider = self.settings.llm_provider.lower()
        if provider == "anthropic":
            return self._invoke_anthropic(system_prompt, user_prompt, schema_json)
        if provider in ("openai", "openai-compatible"):
            return self._invoke_openai(system_prompt, user_prompt, schema_json)
        raise LLMError(f"Unsupported LLM provider: {self.settings.llm_provider}")

    def _invoke_anthropic(self, system_prompt: str, user_prompt: str, schema_json: dict) -> dict:
        import anthropic  # lazy import so the module loads without the SDK

        client = anthropic.Anthropic(api_key=self.settings.llm_api_key)
        tool = {
            "name": "record_result",
            "description": "Return the requested fields as structured data.",
            "input_schema": schema_json,
        }
        response = client.messages.create(
            model=self.settings.llm_model,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            system=system_prompt,
            tools=[tool],
            tool_choice={"type": "tool", "name": "record_result"},
            messages=[{"role": "user", "content": user_prompt}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise LLMError("Anthropic response contained no tool_use block")

    def _invoke_openai(self, system_prompt: str, user_prompt: str, schema_json: dict) -> dict:
        import httpx  # lazy import

        base = self.settings.llm_base_url or "https://api.openai.com/v1"
        url = base.rstrip("/") + "/chat/completions"
        # OpenAI-compatible JSON mode needs the word "json" in the prompt and a
        # schema hint to shape the object.
        system = (
            f"{system_prompt}\n\nReturn a single JSON object matching this schema:\n"
            f"{json.dumps(schema_json)}"
        )
        payload = {
            "model": self.settings.llm_model,
            "temperature": config.LLM_TEMPERATURE,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.settings.llm_api_key}"}
        try:
            resp = httpx.post(
                url, json=payload, headers=headers, timeout=config.GITHUB_REQUEST_TIMEOUT * 3
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI-compatible request failed: {exc}") from exc
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)
