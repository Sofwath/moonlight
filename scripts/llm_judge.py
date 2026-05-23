#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DhivehiMT-Bench LLM judge panel.

Implements the LLM-as-judge evaluation protocol from benchmark-design-spec.md §5.2:

  - Judge panel: GPT-4o + Gemini 1.5 Pro (Claude is EXCLUDED — Moonlight uses Claude,
    so including Claude would introduce self-preference bias per Zheng et al. 2023)
  - Swap test: every pairwise comparison run twice with A/B order reversed;
    only consistent verdicts count; inconsistent pairs are recorded as ties
  - Dialect-guided prompt: explicitly states language is Dhivehi (Thaana script,
    Maldivian government register), provides one paragraph of PO register norms,
    lists the most relevant error categories for this domain
  - Scalar scoring: 5-point score per hypothesis (not just pairwise preference),
    enabling Spearman correlation with human ESA scores
  - Calibration gate: Spearman ≥ 0.60 against human ESA scores on the 50-segment
    calibration set before judge results appear in comparative claims

Usage::

    # Score a benchmark results file against a reference
    python scripts/llm_judge.py \\
        --results results/bench_gov_dev_en_dv.json \\
        --systems moonlight_full,gpt4o_raw \\
        --output results/judge_gov_dev_en_dv.json

    # Calibrate the judge against human ESA annotations
    python scripts/llm_judge.py \\
        --calibrate \\
        --calibration-set data/benchmark/calibration_set/annotations/ \\
        --results results/bench_gov_dev_en_dv.json \\
        --output results/judge_calibration.json

    # Pairwise preference mode (A vs B, swap test)
    python scripts/llm_judge.py \\
        --mode pairwise \\
        --system-a moonlight_full \\
        --system-b gpt4o_raw \\
        --results results/bench_gov_dev_en_dv.json \\
        --output results/judge_pairwise.json

Judge models used: gpt-4o, gemini-flash (not Claude — see bias note above).
Override with --judge-models if you are evaluating non-Claude systems.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_ENV = ROOT / ".env"
DEFAULT_CACHE_DB = ROOT / "data" / "judge_cache.db"

# Judge panel: excludes Claude because Moonlight is built on Claude.
_DEFAULT_JUDGE_MODELS = ["gpt-4o", "gemini-flash"]

_CALIBRATION_SPEARMAN_THRESHOLD = 0.60

# ── Dialect-guided system prompt ──────────────────────────────────────────────
#
# Following Islam et al. (2025): explicit language context substantially improves
# LLM judge reliability for low-resource and non-Latin-script languages.

_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of machine translation quality for Dhivehi (Thaana \
script, Maldivian government and institutional register).

LANGUAGE CONTEXT
Dhivehi is written in the Thaana script (Unicode block U+0780–U+07BF, right-to-left).
The Maldivian Presidency Office (PO) uses a formal register with specific conventions:
  - Presidential speech is marked with the honorific verb ވިދާޅުވިއެވެ, never the
    informal ބުންޏެވެ/ބުނި forms.
  - The Maldivian president's title is always ރައީސުލްޖުމްހޫރިއްޔާ — never the
    transliterated ހިޒް އެކްސެލެންسީ (H.E.), which is reserved for foreign heads of state.
  - Institutional terms have PO-established Dhivehi forms (e.g. ދައުލަތުގެ ބަޖެޓު
    for state budget; ވަޒީރުލްމާލިއްޔަތު for Minister of Finance).
  - Dates use Western Arabic numerals with Dhivehi month names, not Arabic-Indic \
numerals.
  - Formal sentences end with -ވިއެވެ / -ތެވެ suffixes; colloquial endings are wrong \
in PO text.

EVALUATION CRITERIA (in order of importance)
1. Accuracy: is the meaning faithfully conveyed? No omissions, additions, or \
hallucinations.
2. Fluency: is the output grammatically correct Dhivehi (Thaana script, no Arabic \
letters)?
3. Terminology: are institutional terms, honorifics, titles, and register correct?

OUTPUT FORMAT — respond with ONLY valid JSON on a single line, no explanation:
{"score": <1-5 integer>, "reasoning": "<one sentence>"}

Score scale:
  5 = Perfect or near-perfect — fluent, accurate, correct register and terminology
  4 = Good — minor fluency or terminology issues that do not affect meaning
  3 = Acceptable — one clear accuracy or register error but overall intelligible
  2 = Poor — multiple errors; significant meaning loss or register failure
  1 = Unacceptable — unintelligible, wrong script, or major meaning error
"""

_PAIRWISE_SYSTEM_PROMPT = _JUDGE_SYSTEM_PROMPT + """
PAIRWISE TASK
You will see a source sentence, a reference translation, and two system outputs (A \
and B).
Rate each output 1–5 and state your preference.

OUTPUT FORMAT — respond with ONLY valid JSON on a single line:
{"score_a": <1-5>, "score_b": <1-5>, "preference": "A"|"B"|"tie", \
"reasoning": "<one sentence>"}
"""


# ── LLM calls ────────────────────────────────────────────────────────────────

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


def _call_judge(
    judge_model: str,
    user_prompt: str,
    system_prompt: str = _JUDGE_SYSTEM_PROMPT,
) -> tuple[str, float]:
    """Call a judge model. Returns (raw_response_text, cost_usd)."""
    from moonlight.llm import LLMClient
    llm = LLMClient(judge_model)
    text, ti, to = llm.chat(system_prompt, user_prompt)
    return text.strip(), llm.cost_usd(ti, to)


def _parse_score_response(text: str) -> dict | None:
    """Extract JSON from LLM judge response. Returns None if unparseable."""
    # Try the response directly first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON from within prose
    m = re.search(r'\{[^{}]+\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ── Score cache ───────────────────────────────────────────────────────────────

class JudgeCache:
    def __init__(self, path: Path) -> None:
        import sqlite3
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS judge_scores (
               cache_key   TEXT PRIMARY KEY,
               judge_model TEXT NOT NULL,
               raw_response TEXT,
               score       REAL,
               cost_usd    REAL DEFAULT 0,
               judged_at   TEXT
            )"""
        )
        self.conn.commit()

    def get(self, cache_key: str) -> dict | None:
        row = self.conn.execute(
            "SELECT raw_response, score, cost_usd FROM judge_scores WHERE cache_key=?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return {"raw_response": row[0], "score": row[1], "cost_usd": row[2]}

    def put(self, cache_key: str, judge_model: str, raw_response: str,
            score: float | None, cost_usd: float) -> None:
        self.conn.execute(
            """INSERT INTO judge_scores (cache_key, judge_model, raw_response, score,
               cost_usd, judged_at) VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(cache_key) DO UPDATE SET
                 raw_response=excluded.raw_response, score=excluded.score,
                 cost_usd=excluded.cost_usd, judged_at=excluded.judged_at""",
            (cache_key, judge_model, raw_response, score, cost_usd,
             datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# ── Scalar scoring ────────────────────────────────────────────────────────────

def score_hypothesis(
    judge_model: str,
    source: str,
    hypothesis: str,
    reference: str,
    source_lang: str,
    target_lang: str,
    cache: JudgeCache | None = None,
    segment_id: str = "",
) -> dict:
    """Score a single hypothesis with one judge model. Returns score dict."""
    cache_key = f"scalar:{judge_model}:{segment_id}:{hash(hypothesis)}"

    if cache:
        cached = cache.get(cache_key)
        if cached:
            return {
                "score": cached["score"],
                "cost_usd": cached["cost_usd"],
                "from_cache": True,
                "judge_model": judge_model,
            }

    tgt_name = "Dhivehi (Thaana script)" if target_lang == "DV" else "English"
    user = (
        f"Evaluate this {tgt_name} translation.\n\n"
        f"SOURCE ({source_lang}):\n{source}\n\n"
        f"REFERENCE TRANSLATION:\n{reference}\n\n"
        f"SYSTEM OUTPUT TO EVALUATE:\n{hypothesis}"
    )

    raw, cost = _call_judge(judge_model, user)
    parsed = _parse_score_response(raw)
    score_val = float(parsed["score"]) if parsed and "score" in parsed else None

    if cache:
        cache.put(cache_key, judge_model, raw, score_val, cost)

    return {
        "score": score_val,
        "reasoning": parsed.get("reasoning", "") if parsed else "",
        "raw_response": raw,
        "cost_usd": cost,
        "from_cache": False,
        "judge_model": judge_model,
        "parse_failed": parsed is None,
    }


# ── Pairwise judgment with swap test ─────────────────────────────────────────

class SwapTestResult(NamedTuple):
    preference: str        # "A", "B", or "tie" (includes inconsistent)
    consistent: bool       # True if both orderings agreed
    score_a: float | None
    score_b: float | None
    cost_usd: float
    judge_model: str


def judge_pair_swap(
    judge_model: str,
    source: str,
    hypothesis_a: str,
    hypothesis_b: str,
    reference: str,
    source_lang: str,
    target_lang: str,
    cache: JudgeCache | None = None,
    pair_id: str = "",
) -> SwapTestResult:
    """Run a swap test: judge A vs B, then B vs A. Consistent votes only count."""
    tgt_name = "Dhivehi (Thaana script)" if target_lang == "DV" else "English"

    def _judge(sys_x: str, sys_y: str, order_label: str) -> dict | None:
        cache_key = (
            f"pairwise:{judge_model}:{pair_id}:{order_label}:{hash(sys_x)}:{hash(sys_y)}"
        )
        if cache:
            cached = cache.get(cache_key)
            if cached:
                parsed = _parse_score_response(cached["raw_response"] or "")
                return {"parsed": parsed, "cost_usd": cached["cost_usd"], "from_cache": True}

        user = (
            f"Evaluate these two {tgt_name} translations.\n\n"
            f"SOURCE ({source_lang}):\n{source}\n\n"
            f"REFERENCE TRANSLATION:\n{reference}\n\n"
            f"OUTPUT A:\n{sys_x}\n\n"
            f"OUTPUT B:\n{sys_y}"
        )
        raw, cost = _call_judge(judge_model, user, _PAIRWISE_SYSTEM_PROMPT)
        parsed = _parse_score_response(raw)
        if cache:
            score = float(parsed["score_a"]) if parsed and "score_a" in parsed else None
            cache.put(cache_key, judge_model, raw, score, cost)
        return {"parsed": parsed, "cost_usd": cost, "from_cache": False}

    result_ab = _judge(hypothesis_a, hypothesis_b, "ab")
    result_ba = _judge(hypothesis_b, hypothesis_a, "ba")

    total_cost = (result_ab or {}).get("cost_usd", 0) + (result_ba or {}).get("cost_usd", 0)

    parsed_ab = (result_ab or {}).get("parsed") or {}
    parsed_ba = (result_ba or {}).get("parsed") or {}

    pref_ab = parsed_ab.get("preference", "tie")  # A>B in ab order
    pref_ba = parsed_ba.get("preference", "tie")  # A>B in ba order means ba said B wins
    # In ba order: if model says "A" it means ba's A (=original B) wins → original B wins
    # Map ba pref back to original labels:
    ba_original = {"A": "B", "B": "A", "tie": "tie"}.get(pref_ba, "tie")

    if pref_ab == ba_original and pref_ab != "tie":
        preference = pref_ab
        consistent = True
    else:
        preference = "tie"
        consistent = pref_ab == ba_original  # both said tie = consistent tie

    score_a = (
        (parsed_ab.get("score_a", 0) + parsed_ba.get("score_b", 0)) / 2
        if "score_a" in parsed_ab else None
    )
    score_b = (
        (parsed_ab.get("score_b", 0) + parsed_ba.get("score_a", 0)) / 2
        if "score_b" in parsed_ab else None
    )

    return SwapTestResult(
        preference=preference,
        consistent=consistent,
        score_a=round(score_a, 2) if score_a is not None else None,
        score_b=round(score_b, 2) if score_b is not None else None,
        cost_usd=round(total_cost, 5),
        judge_model=judge_model,
    )


# ── Panel aggregation ─────────────────────────────────────────────────────────

def panel_preference(results: list[SwapTestResult]) -> str:
    """Aggregate preferences from multiple judges. Majority vote; ties if split."""
    votes = {"A": 0, "B": 0, "tie": 0}
    for r in results:
        votes[r.preference] = votes.get(r.preference, 0) + 1
    if votes["A"] > votes["B"]:
        return "A"
    elif votes["B"] > votes["A"]:
        return "B"
    return "tie"


# ── Calibration (Spearman against human ESA) ─────────────────────────────────

def compute_spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation between two score lists."""
    if len(x) != len(y) or len(x) < 4:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        return round(float(rho), 3)
    except ImportError:
        # Manual Spearman if scipy not available
        n = len(x)

        def _rank(lst):
            sorted_idx = sorted(range(n), key=lambda i: lst[i])
            ranks = [0.0] * n
            for rank, idx in enumerate(sorted_idx, 1):
                ranks[idx] = float(rank)
            return ranks

        rx, ry = _rank(x), _rank(y)
        d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
        return round(1 - (6 * d2) / (n * (n * n - 1)), 3)


def calibrate_judge(
    calibration_dir: Path,
    results_path: Path,
    judge_models: list[str],
    cache: JudgeCache | None = None,
) -> dict:
    """Compute Spearman correlation between judge scores and human ESA scores.

    Loads human annotations from calibration_dir/*.jsonl and judge scores
    from the benchmark results file. Segments are matched by segment_id.

    Returns calibration report with Spearman values per judge model.
    """
    # Load human annotations
    human_scores: dict[str, float] = {}
    for ann_file in sorted(calibration_dir.glob("*.jsonl")):
        with ann_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ann = json.loads(line)
                seg_id = ann.get("segment_id", "")
                da = ann.get("da_score")
                if seg_id and da is not None:
                    # Average multiple annotations per segment
                    if seg_id in human_scores:
                        human_scores[seg_id] = (human_scores[seg_id] + da) / 2
                    else:
                        human_scores[seg_id] = float(da)

    if not human_scores:
        return {"error": f"No human annotations found in {calibration_dir}"}

    # Load benchmark results
    if not results_path.exists():
        return {"error": f"Results file not found: {results_path}"}

    with results_path.open(encoding="utf-8") as f:
        benchmark_data = json.load(f)

    # Score calibration segments with each judge model
    calibration_results = {}
    for judge_model in judge_models:
        judge_scores: list[float] = []
        human_da: list[float] = []

        raw_results = benchmark_data.get("main_set_raw", {})
        for system_id, segs in raw_results.items():
            for seg_rec in segs:
                seg_id = seg_rec.get("segment_id", "")
                if seg_id not in human_scores:
                    continue

                result = score_hypothesis(
                    judge_model,
                    source=seg_rec.get("source", ""),
                    hypothesis=seg_rec.get("hypothesis", ""),
                    reference=seg_rec.get("reference", ""),
                    source_lang=seg_rec.get("source_lang", "EN"),
                    target_lang=seg_rec.get("target_lang", "DV"),
                    cache=cache,
                    segment_id=f"{system_id}:{seg_id}",
                )
                if result["score"] is not None:
                    judge_scores.append(result["score"])
                    human_da.append(human_scores[seg_id])

        if len(judge_scores) < 4:
            calibration_results[judge_model] = {
                "n_pairs": len(judge_scores),
                "spearman": None,
                "gate_passed": False,
                "note": "Too few paired observations for calibration.",
            }
            continue

        rho = compute_spearman(judge_scores, human_da)
        gate_passed = rho >= _CALIBRATION_SPEARMAN_THRESHOLD

        calibration_results[judge_model] = {
            "n_pairs": len(judge_scores),
            "spearman": rho,
            "gate_passed": gate_passed,
            "threshold": _CALIBRATION_SPEARMAN_THRESHOLD,
            "note": (
                "Judge results APPROVED for comparative claims."
                if gate_passed
                else f"Judge results EXPLORATORY only (Spearman {rho:.3f} < {_CALIBRATION_SPEARMAN_THRESHOLD}). "
                "Report in appendix with explicit caveat."
            ),
        }

    return {
        "n_human_annotated": len(human_scores),
        "judge_models": calibration_results,
        "panel_gate_passed": all(
            v.get("gate_passed", False) for v in calibration_results.values()
        ),
    }


# ── Main evaluation modes ─────────────────────────────────────────────────────

def run_scalar_mode(
    results_path: Path,
    system_ids: list[str],
    judge_models: list[str],
    cache: JudgeCache | None,
) -> dict:
    """Score each system's translations with each judge model (scalar 1-5)."""
    with results_path.open(encoding="utf-8") as f:
        benchmark_data = json.load(f)

    raw = benchmark_data.get("main_set_raw", {})
    output: dict[str, dict] = {}

    for system_id in system_ids:
        if system_id not in raw:
            print(f"  [warn] system '{system_id}' not in results file — skipping")
            continue

        segs = raw[system_id]
        print(f"\n  System: {system_id} ({len(segs)} segments × {len(judge_models)} judges)")
        system_scores: dict[str, list[float]] = {jm: [] for jm in judge_models}
        records = []

        for i, seg_rec in enumerate(segs, 1):
            seg_id = seg_rec.get("segment_id", f"seg_{i}")
            per_judge = {}
            for judge_model in judge_models:
                r = score_hypothesis(
                    judge_model,
                    source=seg_rec.get("source", ""),
                    hypothesis=seg_rec.get("hypothesis", ""),
                    reference=seg_rec.get("reference", ""),
                    source_lang=seg_rec.get("source_lang", "EN"),
                    target_lang=seg_rec.get("target_lang", "DV"),
                    cache=cache,
                    segment_id=f"{system_id}:{seg_id}",
                )
                per_judge[judge_model] = r
                if r["score"] is not None:
                    system_scores[judge_model].append(r["score"])

            # Panel average (mean of non-None scores)
            valid = [v["score"] for v in per_judge.values() if v["score"] is not None]
            panel_score = round(sum(valid) / len(valid), 2) if valid else None

            records.append({
                "segment_id": seg_id,
                "panel_score": panel_score,
                "per_judge": {jm: v["score"] for jm, v in per_judge.items()},
                "cost_usd": sum(v["cost_usd"] for v in per_judge.values()),
            })

            marker = "·" if all(v.get("from_cache") for v in per_judge.values()) else "✓"
            print(f"  [{i:>4}/{len(segs)}] {marker} panel={panel_score}", end="\r")

        print()
        output[system_id] = {
            "n": len(segs),
            "mean_by_judge": {
                jm: round(sum(scores)/len(scores), 2) if scores else None
                for jm, scores in system_scores.items()
            },
            "panel_mean": round(
                sum(r["panel_score"] for r in records if r["panel_score"] is not None)
                / max(sum(1 for r in records if r["panel_score"] is not None), 1),
                2,
            ),
            "records": records,
        }

    return output


def run_pairwise_mode(
    results_path: Path,
    system_a: str,
    system_b: str,
    judge_models: list[str],
    cache: JudgeCache | None,
) -> dict:
    """Run swap-test pairwise comparisons between two systems."""
    with results_path.open(encoding="utf-8") as f:
        benchmark_data = json.load(f)

    raw = benchmark_data.get("main_set_raw", {})
    segs_a = {r["segment_id"]: r for r in raw.get(system_a, [])}
    segs_b = {r["segment_id"]: r for r in raw.get(system_b, [])}
    common = sorted(set(segs_a) & set(segs_b))

    if not common:
        return {"error": f"No overlapping segments between {system_a} and {system_b}"}

    print(f"\n  Pairwise: {system_a} vs {system_b} ({len(common)} segments)")
    records = []
    panel_votes = {"A": 0, "B": 0, "tie": 0}
    total_cost = 0.0

    for i, seg_id in enumerate(common, 1):
        ra = segs_a[seg_id]
        rb = segs_b[seg_id]
        judge_results = []

        for judge_model in judge_models:
            result = judge_pair_swap(
                judge_model,
                source=ra.get("source", ""),
                hypothesis_a=ra.get("hypothesis", ""),
                hypothesis_b=rb.get("hypothesis", ""),
                reference=ra.get("reference", ""),
                source_lang=ra.get("source_lang", "EN"),
                target_lang=ra.get("target_lang", "DV"),
                cache=cache,
                pair_id=f"{system_a}_vs_{system_b}:{seg_id}",
            )
            judge_results.append(result)
            total_cost += result.cost_usd

        panel_pref = panel_preference(judge_results)
        panel_votes[panel_pref] = panel_votes.get(panel_pref, 0) + 1

        records.append({
            "segment_id": seg_id,
            "panel_preference": panel_pref,
            "per_judge": [
                {"judge": r.judge_model, "preference": r.preference,
                 "consistent": r.consistent, "score_a": r.score_a, "score_b": r.score_b}
                for r in judge_results
            ],
        })
        print(f"  [{i:>4}/{len(common)}] panel={panel_pref}", end="\r")

    print()
    n = len(common)
    return {
        "system_a": system_a,
        "system_b": system_b,
        "n_segments": n,
        "panel_votes": panel_votes,
        "a_win_rate": round(panel_votes["A"] / n, 3),
        "b_win_rate": round(panel_votes["B"] / n, 3),
        "tie_rate": round(panel_votes["tie"] / n, 3),
        "total_cost_usd": round(total_cost, 4),
        "records": records,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DhivehiMT-Bench LLM judge panel.")
    parser.add_argument("--results", required=True, metavar="PATH",
                        help="Benchmark results JSON from run_benchmark.py.")
    parser.add_argument("--output", required=True, metavar="PATH")
    parser.add_argument(
        "--mode", default="scalar", choices=["scalar", "pairwise", "calibrate"],
        help="scalar: 1-5 per hypothesis; pairwise: A-vs-B swap test; calibrate: Spearman check.",
    )
    parser.add_argument("--systems", default=None,
                        help="Comma-separated system IDs to judge (scalar/pairwise mode).")
    parser.add_argument("--system-a", default=None, help="System A for pairwise mode.")
    parser.add_argument("--system-b", default=None, help="System B for pairwise mode.")
    parser.add_argument(
        "--judge-models", default=",".join(_DEFAULT_JUDGE_MODELS),
        help=f"Comma-separated judge model IDs. Default: {','.join(_DEFAULT_JUDGE_MODELS)}. "
             "NOTE: Do not include Claude when judging Moonlight outputs (self-preference bias).",
    )
    parser.add_argument("--calibration-set", default=None, metavar="DIR",
                        help="Directory with human ESA annotation JSONL files (calibrate mode).")
    parser.add_argument("--cache-db", default=str(DEFAULT_CACHE_DB), metavar="PATH")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV), metavar="PATH")
    args = parser.parse_args()

    load_env(Path(args.env_file).expanduser())

    judge_models = [m.strip() for m in args.judge_models.split(",")]

    # Warn if Claude is in the judge panel for Moonlight comparisons
    claude_in_panel = any("claude" in jm.lower() for jm in judge_models)
    if claude_in_panel:
        print(
            "WARNING: Claude is in the judge panel. If comparing Moonlight outputs "
            "(which use Claude), this introduces self-preference bias. "
            "Remove Claude from --judge-models for Moonlight comparisons.\n"
        )

    results_path = Path(args.results).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    cache = None if args.no_cache else JudgeCache(Path(args.cache_db).expanduser())

    output_data: dict = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "judge_models": judge_models,
            "results_source": str(results_path),
        }
    }

    if args.mode == "scalar":
        system_ids = (
            [s.strip() for s in args.systems.split(",")]
            if args.systems
            else list(json.loads(results_path.read_text(encoding="utf-8"))
                      .get("main_set_raw", {}).keys())
        )
        print(f"Scalar scoring: {len(system_ids)} systems × {len(judge_models)} judges")
        output_data["scalar_results"] = run_scalar_mode(
            results_path, system_ids, judge_models, cache
        )

    elif args.mode == "pairwise":
        if not args.system_a or not args.system_b:
            sys.exit("Pairwise mode requires --system-a and --system-b")
        output_data["pairwise_result"] = run_pairwise_mode(
            results_path, args.system_a, args.system_b, judge_models, cache
        )

    elif args.mode == "calibrate":
        if not args.calibration_set:
            sys.exit("Calibrate mode requires --calibration-set DIR")
        cal_dir = Path(args.calibration_set).expanduser().resolve()
        print(f"Calibrating judge against human ESA annotations in {cal_dir} …")
        output_data["calibration"] = calibrate_judge(
            cal_dir, results_path, judge_models, cache
        )

        cal = output_data["calibration"]
        print(f"\n  Human-annotated segments: {cal.get('n_human_annotated', 0)}")
        for jm, r in cal.get("judge_models", {}).items():
            rho = r.get("spearman")
            gate = "PASS" if r.get("gate_passed") else "FAIL"
            print(f"  {jm:<20} Spearman={rho:.3f}  gate={gate}")
        panel_gate = "PASS" if cal.get("panel_gate_passed") else "FAIL (exploratory only)"
        print(f"\n  Panel gate: {panel_gate}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults → {output_path}")

    if cache:
        cache.close()


if __name__ == "__main__":
    main()
