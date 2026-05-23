#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Terminal-based ESA (Error Span Annotation) tool for DhivehiMT-Bench.

Implements the WMT 2024 ESA protocol (Amrhein et al.):
  - Direct Assessment score (0–100) for overall translation quality
  - Error span annotation with MQM categories and severity

Reduced MQM profile (3 categories per benchmark-design-spec.md §4.2):
  - ACC  Accuracy: meaning errors, omissions, additions, hallucinations
  - FLU  Fluency: grammaticality, Thaana script correctness, morphological errors
  - TER  Terminology: institutional terms, register errors, honorific errors

Severity: minor (-1), major (-5), critical (-25)

Usage::

    # Annotate a calibration set file
    python scripts/esa_annotate.py \\
        --input data/benchmark/calibration_set/calibration.jsonl \\
        --annotator annotator_id \\
        --output data/benchmark/calibration_set/annotations/annotator_id.jsonl

    # Resume an interrupted session
    python scripts/esa_annotate.py \\
        --input data/benchmark/calibration_set/calibration.jsonl \\
        --annotator annotator_id \\
        --output data/benchmark/calibration_set/annotations/annotator_id.jsonl \\
        --resume

Each annotated segment is appended to the output file immediately, so
interruptions lose at most one annotation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MQM_CATEGORIES = {"ACC": "Accuracy", "FLU": "Fluency", "TER": "Terminology"}
_SEVERITIES = {"minor": -1, "major": -5, "critical": -25}
_SEVERITY_SHORTCUTS = {"n": "minor", "m": "major", "c": "critical"}


def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _divider(char: str = "─", width: int = 72) -> str:
    return char * width


def _load_segments(path: Path) -> list[dict]:
    segments = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                segments.append(json.loads(line))
    return segments


def _load_completed(output_path: Path) -> set[str]:
    """Return set of segment IDs already annotated (for resume)."""
    if not output_path.exists():
        return set()
    completed = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    completed.add(rec["segment_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


def _prompt_da_score() -> int:
    """Prompt for a Direct Assessment score 0–100."""
    while True:
        raw = input("\nDA score (0–100, where 100 = perfect): ").strip()
        try:
            score = int(raw)
            if 0 <= score <= 100:
                return score
            print("  Enter a number between 0 and 100.")
        except ValueError:
            print("  Enter an integer.")


def _prompt_error_spans(source: str, hypothesis: str) -> list[dict]:
    """Interactive error span annotation. Returns list of span dicts."""
    spans = []
    print("\nError span annotation (press Enter with empty input to finish).")
    print("For each error:")
    print("  Category: ACC / FLU / TER")
    print("  Severity: n=minor  m=major  c=critical")
    print("  Quote the error span from the HYPOTHESIS (or describe if implicit)")

    while True:
        print()
        cat_raw = input("  Category (ACC/FLU/TER, or Enter to finish): ").strip().upper()
        if not cat_raw:
            break
        if cat_raw not in _MQM_CATEGORIES:
            print(f"  Unknown category. Use: {', '.join(_MQM_CATEGORIES)}")
            continue

        sev_raw = input("  Severity (n/m/c): ").strip().lower()
        sev = _SEVERITY_SHORTCUTS.get(sev_raw)
        if not sev:
            print("  Use n (minor), m (major), or c (critical).")
            continue

        span_text = input("  Error span (quote from hypothesis): ").strip()
        description = input("  Short description of the error: ").strip()

        spans.append(
            {
                "category": cat_raw,
                "severity": sev,
                "penalty": _SEVERITIES[sev],
                "span_text": span_text,
                "description": description,
            }
        )
        print(f"  → Added {cat_raw} {sev} span.")

    return spans


def _show_segment(
    seg: dict,
    hypothesis: str | None,
    idx: int,
    total: int,
) -> None:
    _clear()
    print(_divider("═"))
    print(f"  Segment {idx}/{total}  |  ID: {seg['id']}  |  Genre: {seg.get('genre','?')}  |  {seg.get('source_lang','?')}→{seg.get('target_lang','?')}")
    print(_divider())
    print(f"\nSOURCE ({seg.get('source_lang','')}):")
    print(f"  {seg['source']}")
    print(f"\nREFERENCE ({seg.get('target_lang','')}):")
    print(f"  {seg['reference']}")
    if hypothesis:
        print(f"\nHYPOTHESIS (system output):")
        print(f"  {hypothesis}")
    print()
    print(_divider())


def _mqm_total(spans: list[dict]) -> int:
    return sum(s["penalty"] for s in spans)


def annotate_segment(
    seg: dict,
    hypothesis: str | None,
    idx: int,
    total: int,
    annotator_id: str,
) -> dict:
    """Run interactive annotation for a single segment. Returns annotation dict."""
    _show_segment(seg, hypothesis, idx, total)

    if hypothesis is None:
        print("NOTE: No hypothesis provided. Annotating reference quality only.")
        print("      DA score reflects reference translation quality.")

    da_score = _prompt_da_score()
    spans = _prompt_error_spans(seg["source"], hypothesis or seg["reference"])

    mqm_total = _mqm_total(spans)
    mqm_adjusted = max(0, da_score + mqm_total)

    print(f"\n  Summary: DA={da_score}  MQM penalty={mqm_total}  Adjusted={mqm_adjusted}")
    confirm = input("  Accept? (Enter=yes, r=redo): ").strip().lower()
    if confirm == "r":
        return annotate_segment(seg, hypothesis, idx, total, annotator_id)

    return {
        "segment_id": seg["id"],
        "annotator_id": annotator_id,
        "annotated_at": datetime.now(timezone.utc).isoformat(),
        "source_lang": seg.get("source_lang"),
        "target_lang": seg.get("target_lang"),
        "genre": seg.get("genre"),
        "da_score": da_score,
        "mqm_spans": spans,
        "mqm_penalty_total": mqm_total,
        "mqm_adjusted_score": mqm_adjusted,
        "hypothesis": hypothesis,
    }


def _append_annotation(output_path: Path, annotation: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(annotation, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Terminal ESA annotation tool for DhivehiMT-Bench calibration set."
    )
    parser.add_argument("--input", required=True, metavar="PATH",
                        help="Input JSONL file (calibration_set/calibration.jsonl).")
    parser.add_argument("--annotator", required=True, metavar="ID",
                        help="Annotator identifier (e.g. 'ann_01').")
    parser.add_argument("--output", required=True, metavar="PATH",
                        help="Output JSONL file for annotations.")
    parser.add_argument("--hypotheses", default=None, metavar="PATH",
                        help="Optional JSONL of system hypotheses keyed by segment_id.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip segments already in the output file.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    segments = _load_segments(input_path)
    hypotheses: dict[str, str] = {}
    if args.hypotheses:
        hyp_path = Path(args.hypotheses).expanduser().resolve()
        if hyp_path.exists():
            with hyp_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        hypotheses[rec["segment_id"]] = rec.get("hypothesis", "")

    completed: set[str] = set()
    if args.resume:
        completed = _load_completed(output_path)
        print(f"Resuming: {len(completed)} already annotated, {len(segments) - len(completed)} remaining.")

    total = len(segments)
    annotated = 0

    for i, seg in enumerate(segments, 1):
        if seg["id"] in completed:
            continue

        hypothesis = hypotheses.get(seg["id"])
        annotation = annotate_segment(seg, hypothesis, i, total, args.annotator)
        _append_annotation(output_path, annotation)
        annotated += 1

        print(f"\n  Saved. ({annotated} annotated this session, {len(completed) + annotated}/{total} total)")
        cont = input("  Continue? (Enter=yes, q=quit): ").strip().lower()
        if cont == "q":
            break

    print(f"\nSession complete. Annotations saved to: {output_path}")


if __name__ == "__main__":
    main()
