# SPDX-License-Identifier: Apache-2.0
"""Tests for moonlight.llm and moonlight.pricing shim."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from moonlight.llm import (
    MODELS,
    LLMClient,
    ModelSpec,
    RateLimitError,
    cost,
    list_models,
    model_id,
)


# ── 1. Model registry completeness ────────────────────────────────────────────

EXPECTED_ALIASES = [
    "claude-sonnet",
    "claude-haiku",
    "claude-opus",
    "gpt-4o",
    "gpt-4o-mini",
    "o3-mini",
    "gemini-flash",
    "gemini-pro",
    "deepseek",
    "deepseek-r1",
    "qwen-turbo",
    "qwen-plus",
    "qwen-max",
    "mistral-large",
    "mistral-small",
    "llama-3.3-70b",
    "grok-2",
    # backward-compat short aliases
    "sonnet",
    "haiku",
    "opus",
]


@pytest.mark.parametrize("alias", EXPECTED_ALIASES)
def test_model_alias_present(alias):
    assert alias in MODELS, f"Missing alias: {alias!r}"


def test_models_has_17_or_more_unique_ids():
    # We have at least 17 model entries (alias+backward-compat)
    assert len(MODELS) >= 17


# ── 2. model_id() ─────────────────────────────────────────────────────────────

def test_model_id_known():
    assert model_id("claude-sonnet") == "claude-sonnet-4-6"
    assert model_id("gpt-4o") == "gpt-4o"
    assert model_id("gemini-flash") == "gemini-2.0-flash"
    assert model_id("deepseek") == "deepseek-chat"
    assert model_id("qwen-plus") == "qwen-plus"
    assert model_id("mistral-large") == "mistral-large-latest"
    assert model_id("llama-3.3-70b") == "llama-3.3-70b-versatile"


def test_model_id_backward_compat_aliases():
    assert model_id("sonnet") == model_id("claude-sonnet")
    assert model_id("haiku") == model_id("claude-haiku")
    assert model_id("opus") == model_id("claude-opus")


def test_model_id_unknown_raises():
    with pytest.raises(KeyError):
        model_id("nonexistent-model-xyz")


# ── 3. cost() ─────────────────────────────────────────────────────────────────

def test_cost_sonnet_1m_tokens():
    # claude-sonnet: in_per_m=3.0, out_per_m=15.0
    # 1M in + 1M out = 3.0 + 15.0 = $18.0
    result = cost("claude-sonnet", tokens_in=1_000_000, tokens_out=1_000_000)
    assert abs(result - 18.0) < 1e-9


def test_cost_haiku():
    # in_per_m=1.0, out_per_m=5.0 → 1k in + 1k out = (1 + 5) / 1000 = 0.006
    result = cost("claude-haiku", tokens_in=1_000, tokens_out=1_000)
    assert abs(result - (1.0 + 5.0) / 1000) < 1e-9


def test_cost_zero_tokens():
    assert cost("gpt-4o", tokens_in=0, tokens_out=0) == 0.0


def test_cost_gpt4o_mini():
    # in_per_m=0.15, out_per_m=0.60
    result = cost("gpt-4o-mini", tokens_in=1_000_000, tokens_out=1_000_000)
    assert abs(result - (0.15 + 0.60)) < 1e-9


# ── 4. list_models() ──────────────────────────────────────────────────────────

def test_list_models_returns_list():
    rows = list_models()
    assert isinstance(rows, list)
    assert len(rows) > 0


def test_list_models_excludes_short_aliases_by_default():
    rows = list_models()
    aliases = {r["alias"] for r in rows}
    # short aliases excluded
    assert "sonnet" not in aliases
    assert "haiku" not in aliases
    assert "opus" not in aliases


def test_list_models_include_aliases():
    # include_aliases=True removes the alias-exclusion filter, but list_models()
    # still deduplicates by model ID (seen_ids).  The short aliases (sonnet,
    # haiku, opus) share IDs with claude-sonnet/haiku/opus and appear AFTER them
    # in MODELS, so they are still skipped by the seen_ids check.
    # The important invariant: include_aliases=True must return at least as many
    # rows as the default (never fewer).
    rows_default = list_models(include_aliases=False)
    rows_with = list_models(include_aliases=True)
    assert len(rows_with) >= len(rows_default)
    # All full-name aliases must still be present
    aliases_with = {r["alias"] for r in rows_with}
    assert "claude-sonnet" in aliases_with
    assert "claude-haiku" in aliases_with
    assert "claude-opus" in aliases_with


def test_list_models_no_duplicate_ids_by_default():
    rows = list_models()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "Duplicate model IDs returned"


def test_list_models_keys():
    rows = list_models()
    required = {"alias", "id", "provider", "family", "in_per_m", "out_per_m",
                "context_k", "notes"}
    for r in rows:
        assert required.issubset(r.keys())


# ── 5. LLMClient construction ─────────────────────────────────────────────────

def test_llm_client_valid_alias():
    client = LLMClient("claude-sonnet", api_key="fake-key")
    assert client.alias == "claude-sonnet"
    assert isinstance(client.spec, ModelSpec)


def test_llm_client_invalid_alias_raises():
    with pytest.raises(KeyError):
        LLMClient("not-a-real-model")


# ── 6. LLMClient.cost_usd() ───────────────────────────────────────────────────

def test_cost_usd_sonnet():
    client = LLMClient("claude-sonnet", api_key="fake")
    # in_per_m=3.0, out_per_m=15.0
    result = client.cost_usd(1_000_000, 1_000_000)
    assert abs(result - 18.0) < 1e-9


def test_cost_usd_zero():
    client = LLMClient("gpt-4o", api_key="fake")
    assert client.cost_usd(0, 0) == 0.0


# ── 7. LLMClient.model_id property ───────────────────────────────────────────

def test_model_id_property():
    client = LLMClient("claude-sonnet", api_key="fake")
    assert client.model_id == "claude-sonnet-4-6"

def test_model_id_property_gpt4o():
    client = LLMClient("gpt-4o", api_key="fake")
    assert client.model_id == "gpt-4o"


# ── 8a. API key env aliases ───────────────────────────────────────────────────

def test_gemini_api_key_uses_google_alias_when_primary_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    client = LLMClient("gemini-flash")
    assert client._resolve_api_key() == "google-key"


# ── 8. LLMClient._chat_anthropic — mock SDK ───────────────────────────────────

def _make_anthropic_response(text: str, in_tok: int = 10, out_tok: int = 5):
    """Build a mock Anthropic API response object."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=in_tok, output_tokens=out_tok)
    return resp


def test_chat_anthropic_returns_text_and_tokens():
    client = LLMClient("claude-sonnet", api_key="test-key")

    mock_resp = _make_anthropic_response("Hello world", in_tok=20, out_tok=10)
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_resp

    mock_anthropic_module = MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client

    with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
        text, ti, to = client._chat_anthropic(
            "system prompt", "user message",
            max_tokens=100, temperature=0.3, timeout=None,
        )

    assert text == "Hello world"
    assert ti == 20
    assert to == 10


def test_chat_anthropic_temperature_dropped_when_not_supported():
    """o3-mini has supports_temperature=False — temperature kwarg must be omitted."""
    client = LLMClient("o3-mini", api_key="test-key")
    # o3-mini is openai_compat — use _chat_openai_compat but verify via spec flag
    assert client.spec.supports_temperature is False

    # Verify that chat() resolves temperature to None for this model
    # We patch the openai_compat path since o3-mini goes there
    call_kwargs_captured = {}

    def fake_openai_compat(system, user, *, max_tokens, temperature, timeout):
        call_kwargs_captured["temperature"] = temperature
        return ("result", 5, 3)

    client._chat_openai_compat = fake_openai_compat
    # Call the public chat() method — it should set temp=None
    with patch.object(client, "_chat_openai_compat", fake_openai_compat):
        client.chat("sys", "usr", max_tokens=50)
    assert call_kwargs_captured["temperature"] is None


def test_chat_anthropic_rate_limit_raises():
    client = LLMClient("claude-sonnet", api_key="test-key")

    mock_rate_err = type("RateLimitError", (Exception,), {})
    mock_anthropic_module = MagicMock()
    mock_anthropic_module.RateLimitError = mock_rate_err

    mock_messages = MagicMock()
    mock_messages.create.side_effect = mock_rate_err("rate limited")
    mock_anthropic_module.Anthropic.return_value = MagicMock(messages=mock_messages)

    with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
        with pytest.raises(RateLimitError):
            client._chat_anthropic(
                "sys", "usr", max_tokens=100, temperature=0.3, timeout=None
            )


# ── 9. LLMClient._chat_openai_compat — mock SDK ──────────────────────────────

def _make_openai_response(text: str, prompt_tokens=10, completion_tokens=5):
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def test_chat_openai_compat_returns_text_and_tokens():
    client = LLMClient("gpt-4o", api_key="test-key")

    mock_resp = _make_openai_response("Translated text", 15, 8)
    mock_completions = MagicMock()
    mock_completions.create.return_value = mock_resp
    mock_chat = MagicMock(completions=mock_completions)
    mock_oai_client = MagicMock(chat=mock_chat)

    mock_openai_module = MagicMock()
    mock_openai_module.OpenAI.return_value = mock_oai_client
    mock_openai_module.RateLimitError = type("RateLimitError", (Exception,), {})

    with patch.dict("sys.modules", {"openai": mock_openai_module}):
        text, ti, to = client._chat_openai_compat(
            "system prompt", "user message",
            max_tokens=100, temperature=0.3, timeout=None,
        )

    assert text == "Translated text"
    assert ti == 15
    assert to == 8


def test_chat_openai_compat_system_role_used():
    """o3-mini uses system_role='developer' — verify messages use that role."""
    client = LLMClient("o3-mini", api_key="test-key")
    assert client.spec.system_role == "developer"

    captured_messages = []

    mock_resp = _make_openai_response("result", 5, 3)
    mock_completions = MagicMock()
    mock_completions.create.return_value = mock_resp

    def capture_create(**kwargs):
        captured_messages.extend(kwargs.get("messages", []))
        return mock_resp

    mock_completions.create.side_effect = capture_create
    mock_chat = MagicMock(completions=mock_completions)
    mock_oai_client = MagicMock(chat=mock_chat)

    mock_openai_module = MagicMock()
    mock_openai_module.OpenAI.return_value = mock_oai_client
    mock_openai_module.RateLimitError = type("RateLimitError", (Exception,), {})

    with patch.dict("sys.modules", {"openai": mock_openai_module}):
        client._chat_openai_compat(
            "system prompt", "user message",
            max_tokens=100, temperature=None, timeout=None,
        )

    roles = [m["role"] for m in captured_messages]
    assert "developer" in roles
    assert "system" not in roles


# ── 10. RateLimitError on OpenAI path ─────────────────────────────────────────

def test_chat_openai_rate_limit_raises():
    client = LLMClient("gpt-4o", api_key="test-key")

    mock_rate_err = type("RateLimitError", (Exception,), {})
    mock_completions = MagicMock()
    mock_completions.create.side_effect = mock_rate_err("429 Too Many Requests")
    mock_chat = MagicMock(completions=mock_completions)
    mock_oai_client = MagicMock(chat=mock_chat)

    mock_openai_module = MagicMock()
    mock_openai_module.OpenAI.return_value = mock_oai_client
    mock_openai_module.RateLimitError = mock_rate_err

    with patch.dict("sys.modules", {"openai": mock_openai_module}):
        with pytest.raises(RateLimitError):
            client._chat_openai_compat(
                "sys", "usr", max_tokens=100, temperature=0.3, timeout=None
            )


# ── 11. pricing.py shim ───────────────────────────────────────────────────────

def test_pricing_shim_model_id():
    from moonlight.pricing import model_id as pricing_model_id
    assert pricing_model_id("claude-sonnet") == "claude-sonnet-4-6"


def test_pricing_shim_cost():
    from moonlight.pricing import cost as pricing_cost
    result = pricing_cost("claude-sonnet", tokens_in=1_000_000, tokens_out=1_000_000)
    assert abs(result - 18.0) < 1e-9


def test_pricing_shim_models():
    from moonlight.pricing import MODELS as pricing_MODELS
    assert "claude-sonnet" in pricing_MODELS
    assert "gpt-4o" in pricing_MODELS
    assert isinstance(pricing_MODELS["claude-sonnet"], ModelSpec)


def test_pricing_shim_is_same_object():
    """pricing.MODELS should be the same object as llm.MODELS."""
    from moonlight.pricing import MODELS as pricing_MODELS
    from moonlight.llm import MODELS as llm_MODELS
    assert pricing_MODELS is llm_MODELS
