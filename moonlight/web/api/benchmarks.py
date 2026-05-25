# SPDX-License-Identifier: Apache-2.0
"""GET /api/benchmarks — DhivehiMT-Bench reference results.

Serves the pre-computed benchmark JSON files from the results/ directory
so the workbench can show how the current translation compares to frontier
models and to the full moonlight pipeline on the 50-item dev set.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

_RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"

# Benchmark run IDs → friendly metadata
_RUNS = {
    "frontier_baseline": {
        "file": "run_001_frontier_baseline.json",
        "label": "Frontier models (pilot raw baselines)",
        "description": "GPT-4o, Claude Sonnet 4.6, Gemini 2.0 Flash — pure LLM, no RAG or prompt engineering",
    },
    "frontier_best": {
        "file": "run_002_frontier_best.json",
        "label": "Frontier models — best available (raw)",
        "description": "GPT-5.5, Claude Opus 4.7, Gemini 3.5 Flash — latest models without moonlight",
    },
    "moonlight_full": {
        "file": "run_003_moonlight_full.json",
        "label": "moonlight full pipeline",
        "description": "Full moonlight pipeline: RAG exemplars + glossary + phrase contexts + sentence memory",
    },
}


def _load_run(run_id: str) -> dict | None:
    meta = _RUNS.get(run_id)
    if not meta:
        return None
    path = _RESULTS_DIR / meta["file"]
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
        slim_data = {
            "meta": data.get("meta", {}),
            "main_set_aggregate": data.get("main_set_aggregate", {}),
            "challenge_set_aggregate": data.get("challenge_set_aggregate", {}),
        }
        return {**meta, "data": slim_data, "run_id": run_id}
    except Exception:
        return None


@router.get("/benchmarks")
def benchmarks() -> dict:
    """Return all available DhivehiMT-Bench reference results.

    Structure:
    {
      runs: {
        frontier_baseline: { label, description, data: {...} },
        frontier_best:     { ... },
        moonlight_full:    { ... },
      },
      summary: { frontier_chrF, moonlight_chrF, improvement }
    }
    """
    runs = {}
    for run_id in _RUNS:
        loaded = _load_run(run_id)
        if loaded:
            runs[run_id] = loaded

    # Build summary comparison for the header widget
    summary = {}
    if "frontier_best" in runs and "moonlight_full" in runs:
        try:
            fb = runs["frontier_best"]["data"]
            ml = runs["moonlight_full"]["data"]
            fb_systems = fb.get("main_set_aggregate", {})
            ml_systems = ml.get("main_set_aggregate", {})
            # Skip _pairwise_significance and similar meta keys
            fb_chrf = max(
                (v.get("chrf", {}).get("mean", 0)
                 for k, v in fb_systems.items() if not k.startswith("_")),
                default=0,
            )
            ml_chrf = max(
                (v.get("chrf", {}).get("mean", 0)
                 for k, v in ml_systems.items() if not k.startswith("_")),
                default=0,
            )
            # n_items lives inside the system entry
            ml_entry = next(
                (v for k, v in ml_systems.items() if not k.startswith("_")), {}
            )
            summary = {
                "best_frontier_chrf": round(fb_chrf, 1),
                "moonlight_chrf": round(ml_chrf, 1),
                "improvement": round(ml_chrf - fb_chrf, 1),
                "n_items": ml_entry.get("n", 50),
                "dataset": "DhivehiMT-Bench dev / government genre / EN→DV",
            }
        except Exception:
            pass

    return {"runs": runs, "summary": summary}
