# SPDX-License-Identifier: Apache-2.0
"""POST /api/fluency — Dhivehi fluency scoring via dhivehi-gpt2-base.

Computes perplexity of DV text using the alakxender/dhivehi-gpt2-base
language model (trained on Dhivehi Wikipedia). Lower perplexity = more
natural Dhivehi. Returns a 0–100 fluency_score alongside raw perplexity.

Requires the `transformers` + `torch` packages and a one-time ~400 MB
model download. Returns a 503 with clear messaging if not available.
This endpoint is called asynchronously by the workbench after translation
so it never blocks the translation response.
"""
from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel, Field

from ..limits import limiter

router = APIRouter()

# Module-level scorer singleton — loaded once on first call
_scorer = None


def _get_scorer():
    global _scorer
    if _scorer is not None:
        return _scorer
    try:
        from ...dv_fluency import DvFluencyScorer
        _scorer = DvFluencyScorer()
        return _scorer
    except ImportError:
        return None


class FluencyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


FluencyRequest.model_rebuild()


@router.post("/fluency")
@limiter.limit("20/minute")
def fluency(request: Request, req: FluencyRequest = Body(...)) -> dict:
    """Score Dhivehi text fluency using dhivehi-gpt2-base perplexity.

    Returns {perplexity, fluency_score (0–100), has_thaana, available}.
    If `transformers`/`torch` are not installed, returns
    {available: false} rather than raising an error — the workbench
    treats this as a soft failure and simply hides the fluency panel.
    """
    scorer = _get_scorer()
    if scorer is None:
        return {
            "available": False,
            "reason": "transformers/torch not installed; run: pip install transformers torch",
        }

    try:
        ppl = scorer.perplexity(req.text)
        fs = scorer.fluency_score(req.text)

        from ...dv_fluency import _has_thaana
        has_thaana = _has_thaana(req.text)
        return {
            "available": True,
            "perplexity": round(ppl, 2) if ppl is not None else None,
            "fluency_score": fs,
            "has_thaana": has_thaana,
        }
    except Exception as e:
        # Model load error, OOM, etc. — degrade gracefully
        return {
            "available": False,
            "reason": str(e)[:200],
        }
