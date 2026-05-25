# SPDX-License-Identifier: Apache-2.0
"""Provider-agnostic LLM client for the moonlight translation engine.

Design
------
We need two Python SDKs, not a heavy meta-framework:

  * ``anthropic`` — for Claude (Anthropic's native API has a different request
    shape: ``system=`` is a top-level kwarg, not a message role; response text
    lives in ``r.content[0].text``, not ``r.choices[0].message.content``).

  * ``openai`` — for *everyone else*.  DeepSeek, Qwen, Mistral, Groq, Together,
    Fireworks, Perplexity, xAI, Cerebras, and Google Gemini all expose an
    OpenAI-compatible ``/v1/chat/completions`` endpoint.  The ``openai`` SDK
    accepts a custom ``base_url`` and ``api_key``, so one SDK covers nine
    providers.

We explicitly avoid LiteLLM because:
  1. Anthropic prompt-caching (``cache_control`` blocks) is a material cost
     lever for a translator that reuses a fixed system prompt on every call.
     Accessing it requires the native SDK; LiteLLM's abstraction makes it
     fragile.
  2. The leaky abstraction cost (response shapes, error types, capability flags)
     exceeds the benefit for a small, well-defined provider set.
  3. LiteLLM is a moving target; pinned version + leaky abstractions = hidden
     drift.

Architecture
------------

::

    MODELS  ──────────────────────────────────── registry
    dict[alias → ModelSpec]               (provider, base_url, pricing, caps)
              │
              ▼
    LLMClient(alias)      one instance per translation call
      .chat(system, user) ──► Anthropic SDK path  (provider == "anthropic")
                         └──► OpenAI SDK path      (provider == "openai_compat")
      .cost_usd(in, out) ──► (in * in_per_m + out * out_per_m) / 1_000_000
      .model_id          ──► exact API string for DB logging

Capability flags
----------------
Some models have quirks that the adapter normalises silently:

  * ``supports_temperature=False`` — temperature kwarg is dropped (o1/o3,
    DeepSeek-R1 do not accept it).
  * ``system_role="developer"`` — the system message is sent as role
    ``"developer"`` instead of ``"system"`` (required by o1/o3).
  * ``max_tokens_param="max_completion_tokens"`` — renamed kwarg for o-series.

Providers
---------
| Alias                  | Provider     | API base                             |
|------------------------|--------------|--------------------------------------|
| claude-sonnet / sonnet | Anthropic    | default                              |
| claude-haiku / haiku   | Anthropic    | default                              |
| claude-opus / opus     | Anthropic    | default                              |
| gpt-4o                 | OpenAI       | default                              |
| gpt-4o-mini            | OpenAI       | default                              |
| o3-mini                | OpenAI       | default (reasoning, no temperature)  |
| gemini-flash           | Google       | generativelanguage.googleapis.com    |
| gemini-pro             | Google       | generativelanguage.googleapis.com    |
| deepseek               | DeepSeek     | api.deepseek.com/v1                  |
| deepseek-r1            | DeepSeek     | api.deepseek.com/v1 (reasoning)      |
| qwen-turbo             | Alibaba/Qwen | dashscope-intl.aliyuncs.com          |
| qwen-plus              | Alibaba/Qwen | dashscope-intl.aliyuncs.com          |
| qwen-max               | Alibaba/Qwen | dashscope-intl.aliyuncs.com          |
| mistral-large          | Mistral      | api.mistral.ai/v1                    |
| mistral-small          | Mistral      | api.mistral.ai/v1                    |
| llama-3.3-70b          | Meta/Groq    | api.groq.com/openai/v1              |
| grok-2                 | xAI          | api.x.ai/v1                          |

Adding a new provider requires only one new entry in ``MODELS``.  No code
changes needed in the adapter unless the provider has a genuinely novel API
shape.

Environment variables
---------------------
Each provider reads its API key from a specific environment variable:

  ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY,
  DASHSCOPE_API_KEY, MISTRAL_API_KEY, GROQ_API_KEY, XAI_API_KEY

Set the ones for providers you intend to use; the rest are ignored.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, Mapping, Optional


# ── Model spec ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    """Complete description of a model: provider routing, pricing, capabilities."""

    id: str
    """Exact API model string (e.g. ``"claude-sonnet-4-6"``)."""

    provider: str
    """``"anthropic"`` or ``"openai_compat"``."""

    api_key_env: str
    """Environment variable that holds the API key for this provider."""

    base_url: Optional[str]
    """Base URL for the OpenAI-compatible endpoint.  ``None`` for default."""

    in_per_m: float
    """USD cost per 1 million input tokens."""

    out_per_m: float
    """USD cost per 1 million output tokens."""

    context_window: int = 128_000
    """Maximum context in tokens."""

    supports_temperature: bool = True
    """False for reasoning models (o1/o3, DeepSeek-R1) that reject temperature."""

    system_role: str = "system"
    """Role name for the system message.  ``"developer"`` for o-series models."""

    max_tokens_param: str = "max_tokens"
    """Parameter name for the max-output-tokens kwarg.
    o-series uses ``"max_completion_tokens"``."""

    default_temperature: float = 0.3
    """Temperature used when the caller doesn't specify one."""

    family: str = ""
    """Human-readable family name (e.g. ``"Claude 4 / Anthropic"``)."""

    notes: str = ""
    """Short note shown in ``moonlight models`` output."""


# ── Model registry ─────────────────────────────────────────────────────────────
#
# Add new models here.  Pricing is approximate mid-2025; verify against vendor
# pricing pages before billing-sensitive work.  DeepSeek and Qwen offer cache
# discounts (see notes fields).
#
# Short aliases (sonnet, haiku, opus) are preserved for backward compatibility.

_ANTHROPIC = "anthropic"
_OAI = "openai_compat"

MODELS: Final[Mapping[str, ModelSpec]] = {
    # ── Anthropic ──────────────────────────────────────────────────────────────
    "claude-sonnet": ModelSpec(
        id="claude-sonnet-4-6",
        provider=_ANTHROPIC, api_key_env="ANTHROPIC_API_KEY", base_url=None,
        in_per_m=3.0, out_per_m=15.0, context_window=200_000,
        family="Claude 4 / Anthropic",
        notes="Best price-performance for production translation",
    ),
    "claude-haiku": ModelSpec(
        id="claude-haiku-4-5",
        provider=_ANTHROPIC, api_key_env="ANTHROPIC_API_KEY", base_url=None,
        in_per_m=1.0, out_per_m=5.0, context_window=200_000,
        family="Claude 4 / Anthropic",
        notes="Fastest and cheapest Claude",
    ),
    "claude-opus": ModelSpec(
        id="claude-opus-4-7",
        provider=_ANTHROPIC, api_key_env="ANTHROPIC_API_KEY", base_url=None,
        in_per_m=15.0, out_per_m=75.0, context_window=200_000,
        supports_temperature=False,
        family="Claude 4 / Anthropic",
        notes="Strongest reasoning; temperature deprecated in 4.7",
    ),
    # Backward-compat short aliases
    "sonnet": ModelSpec(
        id="claude-sonnet-4-6",
        provider=_ANTHROPIC, api_key_env="ANTHROPIC_API_KEY", base_url=None,
        in_per_m=3.0, out_per_m=15.0, context_window=200_000,
        family="Claude 4 / Anthropic", notes="alias for claude-sonnet",
    ),
    "haiku": ModelSpec(
        id="claude-haiku-4-5",
        provider=_ANTHROPIC, api_key_env="ANTHROPIC_API_KEY", base_url=None,
        in_per_m=1.0, out_per_m=5.0, context_window=200_000,
        family="Claude 4 / Anthropic", notes="alias for claude-haiku",
    ),
    "opus": ModelSpec(
        id="claude-opus-4-7",
        provider=_ANTHROPIC, api_key_env="ANTHROPIC_API_KEY", base_url=None,
        in_per_m=15.0, out_per_m=75.0, context_window=200_000,
        supports_temperature=False,
        family="Claude 4 / Anthropic", notes="alias for claude-opus",
    ),

    # ── OpenAI ─────────────────────────────────────────────────────────────────
    "gpt-5.5": ModelSpec(
        id="gpt-5.5-2026-04-23",
        provider=_OAI, api_key_env="OPENAI_API_KEY", base_url=None,
        in_per_m=5.0, out_per_m=20.0, context_window=128_000,
        family="GPT-5 / OpenAI",
        max_tokens_param="max_completion_tokens",
        supports_temperature=False,
        notes="GPT-5.5 standard; pinned for run_002 reproducibility",
    ),
    "gpt-5.5-pro": ModelSpec(
        id="gpt-5.5-pro",
        provider=_OAI, api_key_env="OPENAI_API_KEY", base_url=None,
        in_per_m=10.0, out_per_m=40.0, context_window=128_000,
        family="GPT-5 / OpenAI",
        max_tokens_param="max_completion_tokens",
        supports_temperature=False,
        notes="Best GPT-5; highest capability; temperature deprecated",
    ),
    "gpt-4o": ModelSpec(
        id="gpt-4o",
        provider=_OAI, api_key_env="OPENAI_API_KEY", base_url=None,
        in_per_m=2.50, out_per_m=10.0, context_window=128_000,
        family="GPT-4o / OpenAI",
        notes="Strong multimodal workhorse",
    ),
    "gpt-4o-mini": ModelSpec(
        id="gpt-4o-mini",
        provider=_OAI, api_key_env="OPENAI_API_KEY", base_url=None,
        in_per_m=0.15, out_per_m=0.60, context_window=128_000,
        family="GPT-4o / OpenAI",
        notes="Cheap; good for glossary extraction",
    ),
    "o3-mini": ModelSpec(
        id="o3-mini",
        provider=_OAI, api_key_env="OPENAI_API_KEY", base_url=None,
        in_per_m=1.10, out_per_m=4.40, context_window=200_000,
        supports_temperature=False, system_role="developer",
        max_tokens_param="max_completion_tokens",
        default_temperature=1.0,
        family="o-series / OpenAI",
        notes="Reasoning model; no temperature; system→developer role",
    ),

    # ── Google Gemini (via OpenAI-compatible endpoint) ─────────────────────────
    # Google AI Studio key (not Vertex IAM) — simpler auth for research use.
    "gemini-flash": ModelSpec(
        id="gemini-2.0-flash",
        provider=_OAI,
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        in_per_m=0.10, out_per_m=0.40, context_window=1_000_000,
        family="Gemini 2.0 / Google",
        notes="Cheapest capable model; 1M context",
    ),
    "gemini-pro": ModelSpec(
        id="gemini-2.5-pro",
        provider=_OAI,
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        in_per_m=1.25, out_per_m=5.0, context_window=2_000_000,
        family="Gemini 2.5 / Google",
        notes="Highest-quality Gemini tier for complex text work",
    ),
    "gemini-2.5-pro": ModelSpec(
        id="gemini-2.5-pro",
        provider=_OAI,
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        in_per_m=1.25, out_per_m=10.0, context_window=1_000_000,
        family="Gemini 2.5 / Google",
        notes="Best Gemini 2.5; strong reasoning + multilingual",
    ),
    "gemini-3.5-flash": ModelSpec(
        id="gemini-3.5-flash",
        provider=_OAI,
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        in_per_m=0.15, out_per_m=0.60, context_window=1_000_000,
        family="Gemini 3.5 / Google",
        notes="Latest Gemini; fast + strong multilingual",
    ),

    # ── DeepSeek ───────────────────────────────────────────────────────────────
    # Cache-hit pricing dramatically lower: ~$0.07/$0.28 (V3), ~$0.14/$0.55 (R1).
    "deepseek": ModelSpec(
        id="deepseek-chat",
        provider=_OAI,
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        in_per_m=0.27, out_per_m=1.10, context_window=64_000,
        family="DeepSeek V3",
        notes="Very cheap; strong on structured text; cache discounts apply",
    ),
    "deepseek-r1": ModelSpec(
        id="deepseek-reasoner",
        provider=_OAI,
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        in_per_m=0.55, out_per_m=2.19, context_window=64_000,
        supports_temperature=False, default_temperature=1.0,
        family="DeepSeek R1",
        notes="Reasoning model; open weights; response includes chain-of-thought",
    ),

    # ── Qwen (Alibaba / DashScope) ─────────────────────────────────────────────
    # International endpoint: dashscope-intl.aliyuncs.com
    # China endpoint (if needed): dashscope.aliyuncs.com
    "qwen-turbo": ModelSpec(
        id="qwen-turbo",
        provider=_OAI,
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        in_per_m=0.05, out_per_m=0.20, context_window=1_000_000,
        family="Qwen / Alibaba",
        notes="Cheapest; 1M context; good for bulk glossary building",
    ),
    "qwen-plus": ModelSpec(
        id="qwen-plus",
        provider=_OAI,
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        in_per_m=0.40, out_per_m=1.20, context_window=128_000,
        family="Qwen / Alibaba",
        notes="Workhorse; strong multilingual including Arabic-script languages",
    ),
    "qwen-max": ModelSpec(
        id="qwen-max",
        provider=_OAI,
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        in_per_m=1.60, out_per_m=6.40, context_window=32_000,
        family="Qwen / Alibaba",
        notes="Flagship Qwen; strongest reasoning in the family",
    ),

    # ── Mistral ────────────────────────────────────────────────────────────────
    "mistral-large": ModelSpec(
        id="mistral-large-latest",
        provider=_OAI,
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        in_per_m=2.0, out_per_m=6.0, context_window=128_000,
        family="Mistral Large / Mistral AI",
        notes="Flagship; strong on European languages and formal register",
    ),
    "mistral-small": ModelSpec(
        id="mistral-small-latest",
        provider=_OAI,
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        in_per_m=0.20, out_per_m=0.60, context_window=32_000,
        family="Mistral Small / Mistral AI",
        notes="Efficient; good for repetitive extraction tasks",
    ),

    # ── Meta Llama via Groq (fastest inference) ────────────────────────────────
    "llama-3.3-70b": ModelSpec(
        id="llama-3.3-70b-versatile",
        provider=_OAI,
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        in_per_m=0.59, out_per_m=0.79, context_window=128_000,
        family="Llama 3.3 / Meta via Groq",
        notes="Open-weights model; LPU hardware = very fast; good cost/quality",
    ),

    # ── xAI Grok ───────────────────────────────────────────────────────────────
    "grok-2": ModelSpec(
        id="grok-2-latest",
        provider=_OAI,
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        in_per_m=2.0, out_per_m=10.0, context_window=131_072,
        family="Grok 2 / xAI",
        notes="Competitive general-purpose model",
    ),
}


# ── Exceptions ─────────────────────────────────────────────────────────────────

class RateLimitError(Exception):
    """Raised by LLMClient.chat() when the provider returns a rate-limit response.

    Normalises provider-specific rate-limit exceptions (anthropic.RateLimitError,
    openai.RateLimitError, HTTP 429) into one type so callers can implement
    retry logic without knowing which SDK is in use.
    """


class LLMError(Exception):
    """Raised for non-rate-limit errors from the LLM provider."""


# ── LLMClient ──────────────────────────────────────────────────────────────────

class LLMClient:
    """Provider-agnostic wrapper around a single model.

    Construction::

        llm = LLMClient("claude-sonnet")     # reads ANTHROPIC_API_KEY from env
        llm = LLMClient("gpt-4o", api_key="sk-...")  # explicit key
        llm = LLMClient("deepseek")          # reads DEEPSEEK_API_KEY

    Usage::

        text, tokens_in, tokens_out = llm.chat(system_prompt, user_message)
        cost = llm.cost_usd(tokens_in, tokens_out)

    The adapter normalises provider quirks silently:
      - Anthropic: system sent as top-level kwarg; response from content blocks
      - o1/o3 OpenAI: temperature dropped; role renamed to "developer";
        max_tokens → max_completion_tokens
      - DeepSeek-R1: temperature dropped; ``reasoning_content`` stripped from
        the returned text (final answer only)
    """

    def __init__(self, model_alias: str, *, api_key: Optional[str] = None) -> None:
        if model_alias not in MODELS:
            raise KeyError(
                f"Unknown model alias {model_alias!r}. "
                f"Run `moonlight models` to list available models. "
                f"Known: {sorted(MODELS)}"
            )
        self.alias = model_alias
        self.spec: ModelSpec = MODELS[model_alias]
        self._api_key = api_key

    def _resolve_api_key(self) -> Optional[str]:
        """Resolve provider API key with compatibility aliases.

        Canonical keys come from ``spec.api_key_env``. For Google/Gemini,
        accept both ``GEMINI_API_KEY`` and ``GOOGLE_API_KEY`` so users can
        reuse existing Google AI environment naming.
        """
        if self._api_key:
            return self._api_key

        key = os.environ.get(self.spec.api_key_env)
        if key:
            return key

        if self.spec.api_key_env == "GEMINI_API_KEY":
            return os.environ.get("GOOGLE_API_KEY")
        return None

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        """Exact API model string — use this when persisting to the database."""
        return self.spec.id

    # ── Cost ───────────────────────────────────────────────────────────────────

    def cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        """USD cost for a single call with the given token counts."""
        return (
            tokens_in * self.spec.in_per_m + tokens_out * self.spec.out_per_m
        ) / 1_000_000

    # ── Main interface ─────────────────────────────────────────────────────────

    def chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4000,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> tuple[str, int, int]:
        """Send a system + user message pair and return the response.

        Returns ``(text, tokens_in, tokens_out)``.

        ``temperature`` defaults to ``spec.default_temperature``.  For models
        that don't support temperature (o1/o3, DeepSeek-R1), the parameter is
        silently dropped regardless of what the caller passes.

        Raises :exc:`RateLimitError` on HTTP 429 / rate-limit responses so
        callers can implement provider-agnostic retry logic.
        """
        spec = self.spec
        temp: Optional[float] = (
            temperature if temperature is not None else spec.default_temperature
        )
        if not spec.supports_temperature:
            temp = None

        try:
            if spec.provider == "anthropic":
                return self._chat_anthropic(
                    system, user,
                    max_tokens=max_tokens, temperature=temp, timeout=timeout,
                )
            else:
                return self._chat_openai_compat(
                    system, user,
                    max_tokens=max_tokens, temperature=temp, timeout=timeout,
                )
        except RateLimitError:
            raise
        except Exception as exc:
            raise LLMError(f"{spec.provider}/{spec.id}: {exc}") from exc

    # ── Anthropic backend ──────────────────────────────────────────────────────

    def _chat_anthropic(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: Optional[float],
        timeout: Optional[float],
    ) -> tuple[str, int, int]:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise LLMError(
                "anthropic SDK not installed. Run: pip install anthropic"
            ) from exc

        key = self._resolve_api_key()
        if not key:
            raise LLMError(
                f"{self.spec.api_key_env} is not set. "
                "Export it in your shell or .env file."
            )

        client = _anthropic.Anthropic(api_key=key)
        kwargs: dict = dict(
            model=self.spec.id,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            if timeout is not None:
                resp = client.with_options(timeout=timeout).messages.create(**kwargs)
            else:
                resp = client.messages.create(**kwargs)
        except _anthropic.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc

        text = "".join(
            getattr(b, "text", "")
            for b in resp.content
            if getattr(b, "type", None) == "text"
        )
        return (
            text,
            getattr(resp.usage, "input_tokens", 0),
            getattr(resp.usage, "output_tokens", 0),
        )

    # ── OpenAI-compatible backend ──────────────────────────────────────────────

    def _chat_openai_compat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: Optional[float],
        timeout: Optional[float],
    ) -> tuple[str, int, int]:
        try:
            from openai import OpenAI as _OpenAI
            import openai as _openai
        except ImportError as exc:
            raise LLMError(
                "openai SDK not installed. Run: pip install openai"
            ) from exc

        key = self._resolve_api_key()
        if not key:
            extra = ""
            if self.spec.api_key_env == "GEMINI_API_KEY":
                extra = " (GOOGLE_API_KEY is also accepted)"
            raise LLMError(
                f"{self.spec.api_key_env} is not set{extra}. "
                "Export it in your shell or .env file."
            )

        client_kwargs: dict = {"api_key": key}
        if self.spec.base_url:
            client_kwargs["base_url"] = self.spec.base_url
        if timeout is not None:
            client_kwargs["timeout"] = timeout

        client = _OpenAI(**client_kwargs)

        messages = [
            {"role": self.spec.system_role, "content": system},
            {"role": "user", "content": user},
        ]
        call_kwargs: dict = {
            "model": self.spec.id,
            "messages": messages,
            self.spec.max_tokens_param: max_tokens,
        }
        if temperature is not None:
            call_kwargs["temperature"] = temperature

        try:
            resp = client.chat.completions.create(**call_kwargs)
        except _openai.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc

        text = resp.choices[0].message.content or ""
        # DeepSeek-R1 returns a separate reasoning_content field before the
        # final answer.  We return only the final answer (content field) so the
        # caller doesn't need to strip chain-of-thought — the translator should
        # output a translation, not a reasoning trace.
        usage = resp.usage
        return (
            text,
            getattr(usage, "prompt_tokens", 0),
            getattr(usage, "completion_tokens", 0),
        )


# ── Convenience helpers ────────────────────────────────────────────────────────

def model_id(alias: str) -> str:
    """Return the exact API model string for a short alias.

    Kept for backward compatibility with code that imported from
    ``moonlight.pricing``.
    """
    if alias not in MODELS:
        raise KeyError(
            f"Unknown model alias {alias!r}. Known: {sorted(MODELS)}"
        )
    return MODELS[alias].id


def cost(alias: str, *, tokens_in: int, tokens_out: int) -> float:
    """Calculate USD cost for one call.  Backward-compat shim."""
    m = MODELS[alias]
    return (tokens_in * m.in_per_m + tokens_out * m.out_per_m) / 1_000_000


def list_models(*, include_aliases: bool = False) -> list[dict]:
    """Return a list of model info dicts for display purposes.

    By default, backward-compat aliases (sonnet, haiku, opus) are omitted.
    Pass ``include_aliases=True`` to include them.
    """
    aliases = {"sonnet", "haiku", "opus"}  # short aliases duplicated above
    rows = []
    seen_ids: set[str] = set()
    for alias, spec in MODELS.items():
        if not include_aliases and alias in aliases:
            continue
        if spec.id in seen_ids:
            continue
        seen_ids.add(spec.id)
        rows.append({
            "alias":        alias,
            "id":           spec.id,
            "provider":     spec.provider,
            "family":       spec.family,
            "in_per_m":     spec.in_per_m,
            "out_per_m":    spec.out_per_m,
            "context_k":    spec.context_window // 1000,
            "notes":        spec.notes,
        })
    return rows
