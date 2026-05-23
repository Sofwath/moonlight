#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark baseline vs Moonlight-enhanced translation across models.

This script evaluates two conditions per model:
  1) baseline  : retrieval/context layers disabled via ablation
  2) moonlight : full pipeline enabled

It reports BLEU, chrF, a blended score, and per-model improvement.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import sacrebleu

# Allow running as: python scripts/benchmark_models.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moonlight.db import get_connection  # noqa: E402
from moonlight.llm import LLMClient, MODELS  # noqa: E402
from moonlight.translator import translate  # noqa: E402


BASELINE_ABLATIONS = {
    "few_shot",
    "glossary",
    "phrase_contexts",
    "sentence_memory",
    "genre_routing",
    "term_locking",
    "polish",
}


@dataclass
class EvalItem:
    article_id: int
    direction: str
    source_text: str
    reference_text: str
    target_lang: str


class TimeoutLLM:
    """LLM adapter that enforces a timeout per chat call."""

    def __init__(self, alias: str, timeout_seconds: float) -> None:
        self._llm = LLMClient(alias)
        self._timeout = timeout_seconds
        self.model_id = self._llm.model_id
        self.spec = self._llm.spec

    def chat(self, system, user, **kw):
        kw.setdefault("timeout", self._timeout)
        return self._llm.chat(system, user, **kw)

    def cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        return self._llm.cost_usd(tokens_in, tokens_out)


def fetch_pairs(conn, n_pairs: int, max_chars: int) -> list[tuple]:
    rows = conn.execute(
        """SELECT en.id, en.body_text, dv.body_text
           FROM articles en
           JOIN articles dv ON en.paired_id = dv.id
           WHERE en.language='EN' AND dv.language='DV'
             AND en.body_text IS NOT NULL AND en.body_text != ''
             AND dv.body_text IS NOT NULL AND dv.body_text != ''
           ORDER BY en.published_date DESC
           LIMIT ?""",
        (n_pairs,),
    ).fetchall()
    out = []
    for row in rows:
        en_text = (row[1] or "").strip()[:max_chars]
        dv_text = (row[2] or "").strip()[:max_chars]
        if not en_text or not dv_text:
            continue
        out.append((int(row[0]), en_text, dv_text))
    return out


def build_eval_items(pairs: list[tuple], directions: str) -> list[EvalItem]:
    items: list[EvalItem] = []
    for aid, en_text, dv_text in pairs:
        if directions in ("both", "en2dv"):
            items.append(
                EvalItem(
                    article_id=aid,
                    direction="en2dv",
                    source_text=en_text,
                    reference_text=dv_text,
                    target_lang="DV",
                )
            )
        if directions in ("both", "dv2en"):
            items.append(
                EvalItem(
                    article_id=aid,
                    direction="dv2en",
                    source_text=dv_text,
                    reference_text=en_text,
                    target_lang="EN",
                )
            )
    return items


def compute_metrics(hypotheses: list[str], references: list[str], target_lang: str) -> dict:
    if not hypotheses:
        return {"bleu": 0.0, "chrf": 0.0, "score": 0.0}
    tokenize = "char" if target_lang == "DV" else "13a"
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenize).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], char_order=6, word_order=0).score
    score = 0.3 * bleu + 0.7 * chrf
    return {"bleu": bleu, "chrf": chrf, "score": score}


def evaluate_condition(
    conn,
    model: str,
    items: list[EvalItem],
    mode: str,
    *,
    ablate: set[str],
    timeout_seconds: float,
) -> dict:
    llm = TimeoutLLM(model, timeout_seconds)
    by_dir = {"EN": {"hyps": [], "refs": []}, "DV": {"hyps": [], "refs": []}}
    errors: list[str] = []
    for item in items:
        try:
            res = translate(
                conn,
                item.source_text,
                target_lang=item.target_lang,
                llm=llm,
                mode=mode,
                ablate=ablate,
                n_candidates=1,
                style_transfer=(mode == "po_style"),
            )
            by_dir[item.target_lang]["hyps"].append(res["translation"])
            by_dir[item.target_lang]["refs"].append(item.reference_text)
        except Exception as exc:
            errors.append(f"{item.direction}:{item.article_id}:{exc}")

    en = compute_metrics(by_dir["EN"]["hyps"], by_dir["EN"]["refs"], "EN")
    dv = compute_metrics(by_dir["DV"]["hyps"], by_dir["DV"]["refs"], "DV")

    active_scores = []
    if by_dir["EN"]["hyps"]:
        active_scores.append(en["score"])
    if by_dir["DV"]["hyps"]:
        active_scores.append(dv["score"])
    combined = sum(active_scores) / len(active_scores) if active_scores else 0.0
    return {
        "en": en,
        "dv": dv,
        "combined_score": combined,
        "items_total": len(items),
        "items_scored": len(by_dir["EN"]["hyps"]) + len(by_dir["DV"]["hyps"]),
        "errors": errors,
    }


def parse_models(arg: str) -> list[str]:
    if arg.strip().lower() in {"all", "best"}:
        return []
    out = [m.strip() for m in arg.split(",") if m.strip()]
    return out


def has_key_for_model(model: str) -> bool:
    try:
        llm = LLMClient(model)
        return bool(llm._resolve_api_key())  # noqa: SLF001
    except Exception:
        return False


def resolve_best_models() -> list[str]:
    """Pick the strongest available model for each configured provider.

    Priority is based on provider documentation and Moonlight's model registry:
    - Anthropic: Opus > Sonnet > Haiku
    - Google: Pro > Flash
    """
    provider_best_order = {
        "anthropic": ["claude-opus", "claude-sonnet", "claude-haiku"],
        "google": ["gemini-pro", "gemini-flash"],
        "openai": ["gpt-4o", "o3-mini", "gpt-4o-mini"],
    }
    selected: list[str] = []
    for _, aliases in provider_best_order.items():
        for alias in aliases:
            if alias in MODELS and has_key_for_model(alias):
                selected.append(alias)
                break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Moonlight vs baseline across models.")
    parser.add_argument("--db", default=None, help="Path to moonlight.db (default project DB).")
    parser.add_argument("--models", default="best",
                        help="Comma-separated aliases, 'best', or 'all'.")
    parser.add_argument("--pairs", type=int, default=10, help="Number of EN/DV article pairs.")
    parser.add_argument("--directions", choices=["both", "en2dv", "dv2en"], default="both")
    parser.add_argument("--mode", choices=["faithful", "po_style"], default="faithful")
    parser.add_argument("--max-chars", type=int, default=1400, help="Max chars per source/reference text.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-LLM-call timeout seconds.")
    parser.add_argument("--output-json", default=None, help="Write detailed results to JSON path.")
    args = parser.parse_args()

    conn = get_connection(args.db)
    try:
        pairs = fetch_pairs(conn, n_pairs=args.pairs, max_chars=args.max_chars)
        if not pairs:
            raise SystemExit(
                "No EN/DV article pairs found in DB. Import corpus first, then re-run."
            )
        items = build_eval_items(pairs, args.directions)
        models_arg = args.models.strip().lower()
        if models_arg == "best":
            models = resolve_best_models()
        elif models_arg == "all":
            models = [m for m in MODELS.keys() if m not in {"sonnet", "haiku", "opus"}]
        else:
            models = parse_models(args.models)

        results: dict = {}
        skipped: list[str] = []
        for model in models:
            if model not in MODELS:
                skipped.append(f"{model} (unknown alias)")
                continue
            if not has_key_for_model(model):
                skipped.append(f"{model} (missing API key)")
                continue

            # Ensure old runs don't influence this benchmark via translation cache.
            conn.execute("DELETE FROM translation_runs")
            conn.commit()

            baseline = evaluate_condition(
                conn, model, items, args.mode,
                ablate=BASELINE_ABLATIONS, timeout_seconds=args.timeout,
            )
            moonlight = evaluate_condition(
                conn, model, items, args.mode,
                ablate=set(), timeout_seconds=args.timeout,
            )
            delta = moonlight["combined_score"] - baseline["combined_score"]
            results[model] = {
                "baseline": baseline,
                "moonlight": moonlight,
                "delta_combined": delta,
            }

        if not results:
            raise SystemExit(
                "No runnable models found. Set API keys first "
                "(ANTHROPIC_API_KEY and/or GEMINI_API_KEY or GOOGLE_API_KEY)."
            )

        ranked = sorted(
            (
                {
                    "model": model,
                    "moonlight_score": data["moonlight"]["combined_score"],
                    "baseline_score": data["baseline"]["combined_score"],
                    "delta": data["delta_combined"],
                }
                for model, data in results.items()
            ),
            key=lambda x: x["moonlight_score"],
            reverse=True,
        )

        print("\nModel ranking (higher is better):")
        print("model | moonlight | baseline | delta")
        print("-" * 48)
        for row in ranked:
            print(
                f"{row['model']} | "
                f"{row['moonlight_score']:.2f} | "
                f"{row['baseline_score']:.2f} | "
                f"{row['delta']:+.2f}"
            )

        if skipped:
            print("\nSkipped:")
            for s in skipped:
                print(f"- {s}")

        payload = {
            "config": {
                "db": args.db,
                "pairs": args.pairs,
                "directions": args.directions,
                "mode": args.mode,
                "max_chars": args.max_chars,
                "models": models,
            },
            "ranked": ranked,
            "results": results,
            "skipped": skipped,
        }
        if args.output_json:
            out_path = Path(args.output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            print(f"\nSaved: {out_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
