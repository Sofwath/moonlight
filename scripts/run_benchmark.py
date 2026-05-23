#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DhivehiMT-Bench evaluation harness.

Runs all systems under test against the benchmark main set and challenge set,
with segment-level translation caching, bootstrap confidence intervals, and
per-category challenge set accuracy.

Systems under test (benchmark-design-spec.md §6):
  google_translate   Commercial MT baseline (requires GOOGLE_TRANSLATE_API_KEY)
  gpt4o_raw          Raw GPT-4o, no domain context
  claude_raw         Raw Claude Sonnet 4.6, no domain context
  gemini_raw         Raw Gemini 1.5 Flash, no domain context
  moonlight_nocorp   Moonlight pipeline, empty DB (prompt engineering only)
  moonlight_full     Moonlight pipeline, full corpus
  moonlight_po_style Moonlight in po_style mode (register-optimised)

Usage::

    # Run government genre, dev split, EN→DV direction
    python scripts/run_benchmark.py \\
        --db data/moonlight.db \\
        --benchmark-dir data/benchmark/ \\
        --split dev \\
        --direction en_dv \\
        --genre government \\
        --systems gpt4o_raw,moonlight_nocorp,moonlight_full \\
        --output results/bench_2026-05-24_dev_en_dv.json

    # Run challenge set only (no translation cost — uses existing cache)
    python scripts/run_benchmark.py \\
        --challenge-only \\
        --systems gpt4o_raw,moonlight_full \\
        --output results/challenge_2026-05-24.json

    # Full devtest run (all systems, both directions) — ~$60
    python scripts/run_benchmark.py \\
        --split devtest --direction both --genre all \\
        --systems all \\
        --output results/bench_devtest_full.json

Cache:
    Translations are persisted in data/benchmark_cache.db.
    Re-runs skip cached segments — only new/missing translations are sent to APIs.
    To force re-translation: --no-cache

Cost estimates:
    Government genre (100 segs × 2 dirs) × 7 systems ≈ $3–5
    Full benchmark (400 segs × 2 dirs) × 7 systems ≈ $12–20
    Challenge set (160 pairs × 7 systems) ≈ $0.50–1.00
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DB = ROOT / "data" / "moonlight.db"
DEFAULT_CACHE_DB = ROOT / "data" / "benchmark_cache.db"
DEFAULT_BENCHMARK_DIR = ROOT / "data" / "benchmark"
DEFAULT_ENV = ROOT / ".env"

_SYSTEM_IDS = [
    "google_translate",
    "gpt4o_raw",
    "claude_raw",
    "gemini_raw",
    "moonlight_nocorp",
    "moonlight_full",
    "moonlight_po_style",
]

_BASELINE_SYSTEM_PROMPT = (
    "You are a professional translator specialising in Dhivehi (Thaana script) "
    "and English. Translate the input text faithfully and completely. "
    "Preserve all names, numbers, dates, and institutional titles exactly. "
    "Output only the translation — no commentary, no explanations."
)

_CHALLENGE_PASS_MARGIN = 2.0  # chrF margin required to "pass" a contrastive pair


# ── Translation cache ──────────────────────────────────────────────────────────

class TranslationCache:
    """SQLite-backed cache keyed by (segment_id, system_id)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS translations (
               segment_id  TEXT NOT NULL,
               system_id   TEXT NOT NULL,
               hypothesis  TEXT NOT NULL,
               cost_usd    REAL DEFAULT 0,
               elapsed_s   REAL DEFAULT 0,
               model_id    TEXT,
               cached_at   TEXT,
               PRIMARY KEY (segment_id, system_id)
            )"""
        )
        self.conn.commit()

    def get(self, segment_id: str, system_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT hypothesis, cost_usd, elapsed_s, model_id FROM translations "
            "WHERE segment_id=? AND system_id=?",
            (segment_id, system_id),
        ).fetchone()
        if row is None:
            return None
        return {"hypothesis": row[0], "cost_usd": row[1], "elapsed_s": row[2], "model_id": row[3]}

    def put(self, segment_id: str, system_id: str, hypothesis: str,
            cost_usd: float = 0.0, elapsed_s: float = 0.0, model_id: str = "") -> None:
        self.conn.execute(
            """INSERT INTO translations (segment_id, system_id, hypothesis, cost_usd,
               elapsed_s, model_id, cached_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(segment_id, system_id) DO UPDATE SET
                 hypothesis=excluded.hypothesis, cost_usd=excluded.cost_usd,
                 elapsed_s=excluded.elapsed_s, model_id=excluded.model_id,
                 cached_at=excluded.cached_at""",
            (segment_id, system_id, hypothesis, cost_usd, elapsed_s, model_id,
             datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# ── System runners ─────────────────────────────────────────────────────────────

class SystemRunner(Protocol):
    system_id: str

    def translate(
        self, source: str, source_lang: str, target_lang: str,
        exclude_article_ids: set[int] | None = None,
    ) -> tuple[str, float, str]:
        """Return (hypothesis, cost_usd, model_id)."""
        ...


class BaselineRunner:
    """Raw LLM translation — no corpus context."""

    def __init__(self, system_id: str, model: str) -> None:
        self.system_id = system_id
        self._model = model
        self._llm = None  # lazy init

    def _get_llm(self):
        if self._llm is None:
            from moonlight.llm import LLMClient
            self._llm = LLMClient(self._model)
        return self._llm

    def translate(self, source: str, source_lang: str, target_lang: str,
                  exclude_article_ids: set[int] | None = None) -> tuple[str, float, str]:
        llm = self._get_llm()
        tgt_name = "Dhivehi (Thaana script)" if target_lang == "DV" else "English"
        src_name = "Dhivehi" if source_lang == "DV" else "English"
        user = f"Translate the following {src_name} text to {tgt_name}:\n\n{source}"
        t0 = time.time()
        text_out, ti, to = llm.chat(_BASELINE_SYSTEM_PROMPT, user)
        return text_out.strip(), llm.cost_usd(ti, to), llm.model_id


class MoonlightRunner:
    """Moonlight pipeline — configurable DB and mode."""

    def __init__(self, system_id: str, model: str, db_path: Path, mode: str = "faithful") -> None:
        self.system_id = system_id
        self._model = model
        self._db_path = db_path
        self._mode = mode
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from moonlight.llm import LLMClient
            self._llm = LLMClient(self._model)
        return self._llm

    def translate(self, source: str, source_lang: str, target_lang: str,
                  exclude_article_ids: set[int] | None = None) -> tuple[str, float, str]:
        from moonlight.db import get_connection
        from moonlight.translator import translate
        llm = self._get_llm()
        conn = get_connection(str(self._db_path))
        t0 = time.time()
        try:
            res = translate(
                conn, source,
                target_lang=target_lang,
                mode=self._mode,
                llm=llm,
                model_alias=self._model,
                n_candidates=1,
                exclude_article_ids=exclude_article_ids or set(),
            )
        finally:
            conn.close()
        return res["translation"], res["cost_usd"], res["model"]


class GoogleTranslateRunner:
    """Google Cloud Translation API baseline."""

    def __init__(self) -> None:
        self.system_id = "google_translate"
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google.cloud import translate_v2 as gt
                self._client = gt.Client()
            except ImportError:
                raise ImportError(
                    "google-cloud-translate is required for Google Translate. "
                    "Install: pip install google-cloud-translate"
                )
        return self._client

    def translate(self, source: str, source_lang: str, target_lang: str,
                  exclude_article_ids: set[int] | None = None) -> tuple[str, float, str]:
        client = self._get_client()
        tgt = "dv" if target_lang == "DV" else "en"
        src = "en" if source_lang == "EN" else "dv"
        result = client.translate(source, source_language=src, target_language=tgt)
        return result["translatedText"], 0.0, "google_translate"


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_segment(hypothesis: str, reference: str) -> dict:
    import sacrebleu
    return {
        "bleu": round(sacrebleu.corpus_bleu([hypothesis], [[reference]]).score, 2),
        "chrf": round(sacrebleu.corpus_chrf([hypothesis], [[reference]]).score, 2),
    }


def score_challenge_pair(hypothesis: str, correct: str, incorrect: str,
                          category: str) -> dict:
    """Score a contrastive pair. Returns pass/fail + margins."""
    import sacrebleu

    if category == "cat7_thaana_script":
        # Binary: Thaana-only output required for DV target
        thaana_only = all(
            0x0780 <= ord(c) <= 0x07BF
            or c in " \t\n\r!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~0123456789"
            for c in hypothesis
        )
        return {
            "category": category,
            "passed": thaana_only,
            "margin": None,
            "chrf_correct": None,
            "chrf_incorrect": None,
        }

    chrf_correct = sacrebleu.corpus_chrf([hypothesis], [[correct]]).score
    chrf_incorrect = sacrebleu.corpus_chrf([hypothesis], [[incorrect]]).score
    margin = chrf_correct - chrf_incorrect
    return {
        "category": category,
        "passed": margin >= _CHALLENGE_PASS_MARGIN,
        "margin": round(margin, 2),
        "chrf_correct": round(chrf_correct, 2),
        "chrf_incorrect": round(chrf_incorrect, 2),
    }


def bootstrap_ci(
    scores: list[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """95% bootstrap confidence interval over a list of segment-level scores."""
    try:
        import numpy as np
    except ImportError:
        return {"mean": round(sum(scores)/len(scores), 2) if scores else 0,
                "lower": None, "upper": None, "note": "numpy not installed"}

    rng = np.random.default_rng(seed)
    arr = np.array(scores)
    means = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_resamples)]
    alpha = (1 - confidence) / 2
    return {
        "mean": round(float(arr.mean()), 2),
        "lower": round(float(np.percentile(means, 100 * alpha)), 2),
        "upper": round(float(np.percentile(means, 100 * (1 - alpha))), 2),
    }


# ── Setup helpers ──────────────────────────────────────────────────────────────

def load_env(path: Path) -> None:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]


def build_systems(
    system_ids: list[str],
    db_path: Path,
    model_anthropic: str = "claude-sonnet",
    model_openai: str = "gpt-4o",
    model_gemini: str = "gemini-flash",
) -> dict[str, SystemRunner]:
    """Build runner instances for the requested system IDs.

    Empty-DB path for moonlight_nocorp is created on demand.
    """
    from moonlight.db import get_connection

    nocorp_path = db_path.parent / "benchmark_nocorp.db"
    if not nocorp_path.exists():
        conn = get_connection(str(nocorp_path))
        conn.close()
        print(f"  Created empty DB for moonlight_nocorp: {nocorp_path.name}")

    runners: dict[str, SystemRunner] = {}
    for sid in system_ids:
        if sid == "google_translate":
            runners[sid] = GoogleTranslateRunner()
        elif sid == "gpt4o_raw":
            runners[sid] = BaselineRunner("gpt4o_raw", model_openai)
        elif sid == "claude_raw":
            runners[sid] = BaselineRunner("claude_raw", model_anthropic)
        elif sid == "gemini_raw":
            runners[sid] = BaselineRunner("gemini_raw", model_gemini)
        elif sid == "moonlight_nocorp":
            runners[sid] = MoonlightRunner("moonlight_nocorp", model_anthropic, nocorp_path, "faithful")
        elif sid == "moonlight_full":
            runners[sid] = MoonlightRunner("moonlight_full", model_anthropic, db_path, "faithful")
        elif sid == "moonlight_po_style":
            runners[sid] = MoonlightRunner("moonlight_po_style", model_anthropic, db_path, "po_style")
        else:
            print(f"  [warn] unknown system id: {sid} — skipping")
    return runners


# ── Segment loading ────────────────────────────────────────────────────────────

def load_segments(
    benchmark_dir: Path,
    genres: list[str],
    directions: list[str],
    split: str,
) -> list[dict]:
    """Load segments from JSONL files across genres and directions."""
    segments = []
    for genre in genres:
        for direction in directions:
            p = benchmark_dir / "main_set" / genre / direction / f"{split}.jsonl"
            if not p.exists():
                print(f"  [warn] not found: {p} — skipping")
                continue
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        segments.append(json.loads(line))
    return segments


def load_challenge_pairs(benchmark_dir: Path) -> list[dict]:
    p = benchmark_dir / "challenge_set" / "challenge_seed.jsonl"
    if not p.exists():
        return []
    pairs = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


# ── Main evaluation loops ──────────────────────────────────────────────────────

def eval_main_set(
    segments: list[dict],
    runners: dict[str, SystemRunner],
    cache: TranslationCache | None,
    verbose: bool = True,
) -> dict:
    """Translate all segments with all systems. Returns structured results."""
    results: dict[str, list[dict]] = {sid: [] for sid in runners}

    total = len(segments) * len(runners)
    done = 0

    for seg in segments:
        seg_id = seg["id"]
        src_lang = seg["source_lang"]
        tgt_lang = seg["target_lang"]
        source = seg["source"]
        reference = seg["reference"]
        article_id = seg.get("source_article_id")
        exclude = {article_id} if article_id else set()

        for sid, runner in runners.items():
            done += 1
            # Check cache
            cached = cache.get(seg_id, sid) if cache else None
            if cached:
                hypothesis = cached["hypothesis"]
                cost = cached["cost_usd"]
                model_id = cached.get("model_id", "")
                from_cache = True
            else:
                try:
                    hypothesis, cost, model_id = runner.translate(
                        source, src_lang, tgt_lang, exclude_article_ids=exclude
                    )
                except Exception as e:
                    print(f"\n  [error] {sid} on {seg_id}: {e}")
                    hypothesis = f"[ERROR: {e}]"
                    cost = 0.0
                    model_id = ""
                if cache:
                    cache.put(seg_id, sid, hypothesis, cost, 0.0, model_id)
                from_cache = False

            scores = score_segment(hypothesis, reference)
            results[sid].append({
                "segment_id": seg_id,
                "genre": seg.get("genre"),
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "hypothesis": hypothesis,
                "reference": reference,
                "scores": scores,
                "cost_usd": cost,
                "model_id": model_id,
                "from_cache": from_cache,
            })

            if verbose:
                marker = "·" if from_cache else "✓"
                print(
                    f"  [{done:>4}/{total}] {marker} {sid:<22} {seg_id:<30} "
                    f"chrF={scores['chrf']:.1f}",
                    end="\r",
                )

    if verbose:
        print()
    return results


def eval_challenge_set(
    pairs: list[dict],
    runners: dict[str, SystemRunner],
    cache: TranslationCache | None,
    verbose: bool = True,
) -> dict:
    """Evaluate challenge pairs. Returns per-system per-category results."""
    results: dict[str, list[dict]] = {sid: [] for sid in runners}

    for pair in pairs:
        pair_id = pair["id"]
        category = pair["category"]
        source = pair["source"]
        src_lang = pair["source_lang"]
        tgt_lang = pair["target_lang"]
        correct = pair["correct"]
        incorrect = pair["incorrect"]

        for sid, runner in runners.items():
            cache_key = f"challenge:{pair_id}"
            cached = cache.get(cache_key, sid) if cache else None
            if cached:
                hypothesis = cached["hypothesis"]
                cost = cached["cost_usd"]
            else:
                try:
                    hypothesis, cost, model_id = runner.translate(source, src_lang, tgt_lang)
                except Exception as e:
                    hypothesis = f"[ERROR: {e}]"
                    cost = 0.0
                    model_id = ""
                if cache:
                    cache.put(cache_key, sid, hypothesis, cost)

            pair_score = score_challenge_pair(hypothesis, correct, incorrect, category)
            results[sid].append({
                "pair_id": pair_id,
                "category": category,
                "passed": pair_score["passed"],
                "margin": pair_score.get("margin"),
                "chrf_correct": pair_score.get("chrf_correct"),
                "chrf_incorrect": pair_score.get("chrf_incorrect"),
                "hypothesis": hypothesis,
                "correct": correct,
                "incorrect": incorrect,
                "cost_usd": cost,
            })

    return results


# ── Aggregate + reporting ──────────────────────────────────────────────────────

def aggregate_main_set(per_system: dict[str, list[dict]]) -> dict:
    """Compute aggregate metrics + bootstrap CIs per system."""
    agg = {}
    for sid, records in per_system.items():
        if not records:
            continue
        chrf_scores = [r["scores"]["chrf"] for r in records]
        bleu_scores = [r["scores"]["bleu"] for r in records]
        total_cost = sum(r["cost_usd"] for r in records)

        by_genre: dict[str, list[float]] = {}
        for r in records:
            g = r.get("genre", "unknown")
            by_genre.setdefault(g, []).append(r["scores"]["chrf"])

        agg[sid] = {
            "n": len(records),
            "chrf": bootstrap_ci(chrf_scores),
            "bleu": bootstrap_ci(bleu_scores),
            "chrf_by_genre": {
                g: round(sum(v)/len(v), 2) for g, v in by_genre.items()
            },
            "total_cost_usd": round(total_cost, 4),
        }
    return agg


def aggregate_challenge_set(per_system: dict[str, list[dict]]) -> dict:
    """Compute per-category accuracy per system."""
    agg = {}
    for sid, records in per_system.items():
        cats: dict[str, list[bool]] = {}
        for r in records:
            cats.setdefault(r["category"], []).append(r["passed"])
        per_cat = {cat: round(sum(v)/len(v), 3) for cat, v in cats.items()}
        all_passes = [r["passed"] for r in records]
        agg[sid] = {
            "n_pairs": len(records),
            "overall_accuracy": round(sum(all_passes)/len(all_passes), 3) if all_passes else 0,
            "by_category": per_cat,
        }
    return agg


def render_summary(
    main_agg: dict,
    challenge_agg: dict,
    systems: list[str],
    run_meta: dict,
) -> str:
    lines = ["# DhivehiMT-Bench Results", ""]
    ts = run_meta.get("timestamp", "unknown")
    lines += [
        f"**Run**: {ts}",
        f"**Split**: {run_meta.get('split', '?')}  "
        f"**Direction**: {run_meta.get('direction', '?')}  "
        f"**Genre**: {run_meta.get('genre', '?')}",
        "",
        "---",
        "",
    ]

    if main_agg:
        lines += [
            "## Main set results",
            "",
            "| System | N | chrF (mean) | 95% CI | BLEU | Cost |",
            "|--------|---|:-----------:|--------|:----:|-----:|",
        ]
        for sid in systems:
            if sid not in main_agg:
                continue
            a = main_agg[sid]
            ci = a["chrf"]
            ci_str = (f"[{ci['lower']:.1f}–{ci['upper']:.1f}]"
                      if ci.get("lower") is not None else "n/a")
            lines.append(
                f"| {sid:<25} | {a['n']} | **{ci['mean']:.1f}** | {ci_str} "
                f"| {a['bleu']['mean']:.1f} | ${a['total_cost_usd']:.2f} |"
            )
        lines += ["", "> Primary metric: chrF (character n-gram F-score, 0–100).", ""]

    if challenge_agg:
        lines += [
            "## Challenge set accuracy",
            "",
            "| System | Overall | cat1_register | cat2_honorifics | cat3_entities | cat7_script |",
            "|--------|:-------:|:-------------:|:---------------:|:-------------:|:-----------:|",
        ]
        for sid in systems:
            if sid not in challenge_agg:
                continue
            a = challenge_agg[sid]
            cats = a["by_category"]
            def pct(k): return f"{100*cats.get(k, float('nan')):.0f}%" if k in cats else "—"
            lines.append(
                f"| {sid:<25} | {100*a['overall_accuracy']:.0f}% "
                f"| {pct('cat1_politeness_register')} "
                f"| {pct('cat2_honorifics')} "
                f"| {pct('cat3_named_entities')} "
                f"| {pct('cat7_thaana_script')} |"
            )
        lines += [""]

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DhivehiMT-Bench evaluation harness."
    )
    parser.add_argument("--db", default=str(DEFAULT_DB), metavar="PATH",
                        help="moonlight corpus DB.")
    parser.add_argument("--benchmark-dir", default=str(DEFAULT_BENCHMARK_DIR), metavar="DIR")
    parser.add_argument("--split", default="dev", choices=["dev", "devtest"],
                        help="Benchmark split to evaluate (default: dev).")
    parser.add_argument(
        "--direction", default="en_dv",
        choices=["en_dv", "dv_en", "both"],
    )
    parser.add_argument(
        "--genre", default="government",
        help="Comma-separated genres or 'all'. E.g. government,news",
    )
    parser.add_argument(
        "--systems", default="moonlight_nocorp,moonlight_full",
        help=f"Comma-separated system IDs or 'all'. Options: {', '.join(_SYSTEM_IDS)}",
    )
    parser.add_argument("--output", required=True, metavar="PATH",
                        help="Output JSON path for this run's results.")
    parser.add_argument("--summary-md", default=None, metavar="PATH",
                        help="Optional markdown summary output.")
    parser.add_argument("--cache-db", default=str(DEFAULT_CACHE_DB), metavar="PATH",
                        help="Translation cache database.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable translation cache (force re-translation).")
    parser.add_argument("--challenge-only", action="store_true",
                        help="Only run challenge set evaluation (no main set).")
    parser.add_argument("--no-challenge", action="store_true",
                        help="Skip challenge set evaluation.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV), metavar="PATH")
    parser.add_argument("--model-anthropic", default="claude-sonnet",
                        help="Anthropic model alias for claude_raw and moonlight systems.")
    parser.add_argument("--model-openai", default="gpt-4o",
                        help="OpenAI model alias for gpt4o_raw.")
    parser.add_argument("--model-gemini", default="gemini-flash",
                        help="Gemini model alias for gemini_raw.")
    args = parser.parse_args()

    load_env(Path(args.env_file).expanduser())

    # Resolve systems
    system_ids = (
        _SYSTEM_IDS
        if args.systems == "all"
        else [s.strip() for s in args.systems.split(",")]
    )

    # Resolve genres
    all_genres = ["government", "news", "social", "religious"]
    genres = (
        all_genres
        if args.genre == "all"
        else [g.strip() for g in args.genre.split(",")]
    )
    directions = ["en_dv", "dv_en"] if args.direction == "both" else [args.direction]

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    cache_path = Path(args.cache_db).expanduser().resolve()

    print("=" * 68)
    print("  DhivehiMT-Bench evaluation")
    print("=" * 68)
    print(f"  Systems   : {', '.join(system_ids)}")
    print(f"  Split     : {args.split}")
    print(f"  Directions: {', '.join(directions)}")
    print(f"  Genres    : {', '.join(genres)}")
    print(f"  Cache     : {'disabled' if args.no_cache else cache_path.name}")
    print()

    # Build system runners
    print("[1] Initialising system runners …")
    runners = build_systems(
        system_ids, db_path,
        model_anthropic=args.model_anthropic,
        model_openai=args.model_openai,
        model_gemini=args.model_gemini,
    )
    if not runners:
        sys.exit("No valid systems to run.")

    # Translation cache
    cache = None if args.no_cache else TranslationCache(cache_path)

    main_results_raw: dict[str, list[dict]] = {}
    challenge_results_raw: dict[str, list[dict]] = {}

    # Main set
    if not args.challenge_only:
        print(f"\n[2] Loading benchmark segments …")
        segments = load_segments(benchmark_dir, genres, directions, args.split)
        if not segments:
            print("  No segments found for the requested genre/direction/split combination.")
            print("  Run: python scripts/extract_benchmark_segments.py --direction both")
        else:
            print(f"  {len(segments)} segments × {len(runners)} systems = "
                  f"{len(segments)*len(runners)} translation calls")
            print(f"\n[3] Running main set evaluation …")
            main_results_raw = eval_main_set(segments, runners, cache)
            print(f"  Done.")

    # Challenge set
    if not args.no_challenge:
        print(f"\n[{'4' if not args.challenge_only else '2'}] Loading challenge pairs …")
        pairs = load_challenge_pairs(benchmark_dir)
        if pairs:
            print(f"  {len(pairs)} pairs × {len(runners)} systems")
            print(f"\n[{'5' if not args.challenge_only else '3'}] Running challenge set …")
            challenge_results_raw = eval_challenge_set(pairs, runners, cache)
            print(f"  Done.")
        else:
            print("  No challenge pairs found.")

    # Aggregate
    print("\n[6] Computing aggregate metrics …")
    main_agg = aggregate_main_set(main_results_raw) if main_results_raw else {}
    challenge_agg = aggregate_challenge_set(challenge_results_raw) if challenge_results_raw else {}

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_meta = {
        "timestamp": ts,
        "split": args.split,
        "direction": args.direction,
        "genre": args.genre,
        "systems": system_ids,
    }

    output_data = {
        "meta": run_meta,
        "main_set_aggregate": main_agg,
        "challenge_set_aggregate": challenge_agg,
        "main_set_raw": main_results_raw,
        "challenge_set_raw": challenge_results_raw,
    }

    # Write JSON output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Results → {output_path}")

    # Write markdown summary
    summary_md = args.summary_md or output_path.with_suffix(".md")
    md = render_summary(main_agg, challenge_agg, system_ids, run_meta)
    Path(summary_md).write_text(md, encoding="utf-8")
    print(f"  Summary  → {summary_md}")

    # Print console summary
    print("\n" + "─" * 68)
    if main_agg:
        print(f"  {'System':<25} {'chrF':>8}  {'CI':>18}  {'Cost':>8}")
        print("  " + "─" * 64)
        for sid in system_ids:
            if sid not in main_agg:
                continue
            a = main_agg[sid]
            ci = a["chrf"]
            ci_str = (f"[{ci['lower']:.1f}–{ci['upper']:.1f}]"
                      if ci.get("lower") is not None else "")
            print(f"  {sid:<25} {ci['mean']:>8.1f}  {ci_str:>18}  ${a['total_cost_usd']:>6.2f}")
    if challenge_agg:
        print()
        print(f"  {'System':<25} {'Challenge acc':>13}")
        print("  " + "─" * 40)
        for sid in system_ids:
            if sid not in challenge_agg:
                continue
            acc = challenge_agg[sid]["overall_accuracy"]
            print(f"  {sid:<25} {100*acc:>12.1f}%")
    print("─" * 68)

    if cache:
        cache.close()


if __name__ == "__main__":
    main()
