"""Dhivehi fluency scoring via a monolingual GPT-2 language model.

Uses alakxender/dhivehi-gpt2-base (trained on Dhivehi Wikipedia) to compute
perplexity of DV text as a proxy for lexical naturalness and word-choice quality.

Lower perplexity = more predictable/natural Dhivehi = better fluency.

Limitations:
- Trained on Wikipedia (encyclopedic register); PO government press-release
  register will have higher perplexity than informal text even when correct.
- Perplexity is length-sensitive; short segments score lower than long ones.
- Not a replacement for human fluency judgement — use as a relative ranking
  signal across systems on the same source sentences.

Usage::

    from moonlight.dv_fluency import DvFluencyScorer
    scorer = DvFluencyScorer()          # loads model once, ~400 MB
    score = scorer.perplexity("ރައީސުލްޖުމްހޫރިއްޔާ ވިދާޅުވިއެވެ.")
    # → float, lower is better
    fluency = scorer.fluency_score(text)
    # → 0–100, higher is better (100 * exp(-ppl/100))
"""
from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_MODEL_ID = "alakxender/dhivehi-gpt2-base"
_THAANA_MIN, _THAANA_MAX = 0x0780, 0x07BF


def _has_thaana(text: str) -> bool:
    return any(_THAANA_MIN <= ord(c) <= _THAANA_MAX for c in text)


class DvFluencyScorer:
    """Singleton-friendly perplexity scorer for Dhivehi text."""

    def __init__(self, model_id: str = _MODEL_ID, device: str = "cpu") -> None:
        self._model_id = model_id
        self._device = device
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_id)
        self._model.eval()
        self._model.to(self._device)
        self._torch = torch

    def perplexity(self, text: str) -> float | None:
        """Return perplexity of text under the DV language model.

        Returns None if text contains no Thaana characters (wrong script —
        caller should treat as a hard fluency failure).
        """
        if not _has_thaana(text):
            return None

        self._load()
        torch = self._torch

        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = enc.input_ids.to(self._device)

        if input_ids.shape[1] < 2:
            return None

        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            loss = outputs.loss

        return torch.exp(loss).item()

    def fluency_score(self, text: str) -> float:
        """Return 0–100 fluency score: 100 * exp(-ppl / 100).

        Texts with no Thaana get 0. Very natural text (ppl ~5) → ~95.
        Awkward but correct text (ppl ~50) → ~61. Garbled (ppl ~500) → ~1.
        """
        ppl = self.perplexity(text)
        if ppl is None:
            return 0.0
        return round(100.0 * math.exp(-ppl / 100.0), 2)

    def score_batch(self, texts: list[str]) -> list[dict]:
        """Score a list of texts. Returns list of {perplexity, fluency_score}."""
        results = []
        for t in texts:
            ppl = self.perplexity(t)
            results.append({
                "perplexity": round(ppl, 2) if ppl is not None else None,
                "fluency_score": self.fluency_score(t),
                "has_thaana": _has_thaana(t),
            })
        return results
