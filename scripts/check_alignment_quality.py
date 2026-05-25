#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Alignment quality checker for extracted benchmark segments.

After running extract_benchmark_segments.py, this script flags likely
sentence-alignment mismatches before the segments are committed to the
benchmark main set.

Heuristics (no ground truth required):
  - Length ratio: DV/EN char ratio should be 1.2–2.5 for well-aligned pairs
  - Thaana presence: DV side must contain Thaana codepoints (U+0780–U+07BF)
  - Shared numbers: any digit string in EN should appear in DV
  - Shared years: 4-digit years in EN must be present in DV
  - Arabic contamination: DV side must not contain Arabic codepoints (U+0600–U+06FF)

Outputs a quality report with per-segment flags and a summary.

Usage::

    python scripts/check_alignment_quality.py \\
        data/benchmark/main_set/government/en_dv/segments_raw.jsonl

    # With JSON output
    python scripts/check_alignment_quality.py \\
        data/benchmark/main_set/government/en_dv/segments_raw.jsonl \\
        --json-out data/benchmark/main_set/government/en_dv/quality_report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_THAANA_RANGE = (0x0780, 0x07BF)
# Arabic LETTER range only — explicitly excludes punctuation (U+0600-U+0620)
# The Arabic comma (U+060C) is standard Dhivehi punctuation and must not be flagged.
_ARABIC_LETTER_MIN = 0x0621  # ARABIC LETTER HAMZA
_ARABIC_LETTER_MAX = 0x06FF  # end of Arabic block (covers extended letters)
# Arabic-Indic numerals are a separate Cat-6 issue, not hard contamination
_ARABIC_INDIC_NUMS = set(range(0x0660, 0x066A))

# Direction-specific length ratio thresholds.
# ratio = len(reference) / len(source)
# EN→DV: DV morphology inflates length; EN is the shorter side (1.0–3.0)
# DV→EN: EN is naturally shorter than DV source (0.35–1.2)
_LEN_RATIO = {
    "en_dv": (1.0, 3.0),
    "dv_en": (0.35, 1.2),
}
_LEN_RATIO_DEFAULT = (0.5, 3.0)  # fallback when direction is unknown


def _has_thaana(text: str) -> bool:
    return any(_THAANA_RANGE[0] <= ord(c) <= _THAANA_RANGE[1] for c in text)


def _has_arabic_letters(text: str) -> bool:
    """True if the text contains Arabic letter codepoints.

    Arabic comma (U+060C) and other Arabic punctuation (U+0600–U+0620) are
    standard in Dhivehi writing and must not be counted as contamination.
    Only Arabic letters (U+0621+) indicate genuine script contamination.
    """
    return any(
        _ARABIC_LETTER_MIN <= ord(c) <= _ARABIC_LETTER_MAX
        and ord(c) not in _ARABIC_INDIC_NUMS
        for c in text
    )


def _thaana_tokens(text: str) -> set[str]:
    return set(re.findall(r"[ހ-޿]+", text))


def _thaana_jaccard(hyp: str, ref: str) -> float:
    """Jaccard overlap of Thaana word-tokens between two DV strings.

    Low overlap (< 0.10) means the two strings share almost no vocabulary —
    strong evidence they are from different sentences of the same article rather
    than a translation pair.  Used as a benchmark misalignment detector.
    """
    h = _thaana_tokens(hyp)
    r = _thaana_tokens(ref)
    if not h and not r:
        return 1.0
    union = len(h | r)
    return len(h & r) / union if union else 0.0


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def _extract_years(text: str) -> set[str]:
    return set(re.findall(r"\b(19|20)\d{2}\b", text))


def check_pair(seg: dict) -> dict:
    src = seg.get("source", "")
    ref = seg.get("reference", "")
    src_lang = seg.get("source_lang", "")
    tgt = seg.get("target_lang", "DV")
    flags: list[str] = []
    signals: dict = {}

    # Length ratio — direction-aware thresholds
    direction = f"{src_lang.lower()}_{tgt.lower()}" if src_lang and tgt else ""
    ratio_min, ratio_max = _LEN_RATIO.get(direction, _LEN_RATIO_DEFAULT)
    src_len = max(len(src), 1)
    ref_len = len(ref)
    ratio = ref_len / src_len
    signals["len_ratio"] = round(ratio, 2)
    if ratio < ratio_min:
        flags.append(f"len_ratio_low:{ratio:.2f}")
    elif ratio > ratio_max:
        flags.append(f"len_ratio_high:{ratio:.2f}")

    # Thaana presence in reference (DV side)
    if tgt == "DV":
        signals["has_thaana"] = _has_thaana(ref)
        if not signals["has_thaana"]:
            flags.append("no_thaana_in_reference")
        signals["has_arabic_letters"] = _has_arabic_letters(ref)
        if signals["has_arabic_letters"]:
            flags.append("arabic_contamination")

    # Thaana token Jaccard overlap between source translation (approximated via
    # a reference-only check) and reference.  For EN→DV pairs: both source and
    # reference are in different scripts, so we compare the DV reference against
    # a naive machine-transliterated proxy — not available here.  Instead, flag
    # when the *hypothesis* (if present) and *reference* share almost no Thaana
    # vocabulary.  When only source+reference are available (pre-translation
    # check), use the DV reference's own self-overlap as a proxy: a very short
    # reference with zero content tokens is suspicious.
    # For post-translation review: check_pair accepts an optional "hypothesis"
    # key so the caller can pass the system output alongside the gold reference.
    hyp = seg.get("hypothesis", "")
    if hyp and tgt == "DV":
        jac = _thaana_jaccard(hyp, ref)
        signals["thaana_jaccard"] = round(jac, 3)
        if jac < 0.10:
            flags.append(f"low_thaana_jaccard:{jac:.3f}")

    # Shared numbers — any number in EN must appear in DV
    if seg.get("source_lang") == "EN" and tgt == "DV":
        en_nums = _extract_numbers(src)
        dv_nums = _extract_numbers(ref)
        missing = en_nums - dv_nums
        if missing:
            signals["missing_numbers"] = sorted(missing)
            flags.append(f"missing_numbers:{','.join(sorted(missing)[:3])}")

    # Shared years
    if seg.get("source_lang") == "EN":
        en_years = _extract_years(src)
        dv_years = _extract_years(ref)
        missing_years = en_years - dv_years
        if missing_years:
            signals["missing_years"] = sorted(missing_years)
            flags.append(f"missing_years:{','.join(sorted(missing_years))}")

    status = "PASS" if not flags else ("WARN" if len(flags) == 1 else "FAIL")
    return {
        "id": seg.get("id", "?"),
        "status": status,
        "flags": flags,
        "signals": signals,
        "source_snippet": src[:80],
        "reference_snippet": ref[:80],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check alignment quality of extracted benchmark segments."
    )
    parser.add_argument("input", metavar="JSONL", help="Segments JSONL file.")
    parser.add_argument("--json-out", default=None, metavar="PATH",
                        help="Write detailed JSON quality report to this path.")
    parser.add_argument("--show-fails", action="store_true",
                        help="Print source/reference for FAIL segments.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        sys.exit(f"File not found: {input_path}")

    segments: list[dict] = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                segments.append(json.loads(line))

    print(f"Checking {len(segments)} segments from {input_path.name} …\n")

    results: list[dict] = []
    pass_n = warn_n = fail_n = 0
    for seg in segments:
        r = check_pair(seg)
        results.append(r)
        if r["status"] == "PASS":
            pass_n += 1
        elif r["status"] == "WARN":
            warn_n += 1
        else:
            fail_n += 1

    # Summary
    total = len(results)
    print(f"  PASS: {pass_n:>3} / {total}  ({100*pass_n/total:.0f}%)")
    print(f"  WARN: {warn_n:>3} / {total}  ({100*warn_n/total:.0f}%)")
    print(f"  FAIL: {fail_n:>3} / {total}  ({100*fail_n/total:.0f}%)")
    print()

    # Flag summary
    all_flags: dict[str, int] = {}
    for r in results:
        for flag in r["flags"]:
            key = flag.split(":")[0]
            all_flags[key] = all_flags.get(key, 0) + 1
    if all_flags:
        print("  Flag breakdown:")
        for flag_type, count in sorted(all_flags.items(), key=lambda x: -x[1]):
            print(f"    {flag_type:<30} {count}")
        print()

    # Show WARNs and FAILs
    problem_segments = [r for r in results if r["status"] != "PASS"]
    if problem_segments:
        print(f"  Segments needing review ({len(problem_segments)}):")
        for r in problem_segments:
            marker = "!" if r["status"] == "FAIL" else "~"
            print(f"  [{marker}] {r['id']:<28} {', '.join(r['flags'])}")
            if args.show_fails or r["status"] == "FAIL":
                print(f"       EN: {r['source_snippet']}")
                print(f"       DV: {r['reference_snippet']}")
        print()

    # JSON output
    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "input": str(input_path),
            "total": total,
            "pass": pass_n,
            "warn": warn_n,
            "fail": fail_n,
            "flag_counts": all_flags,
            "segments": results,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  Quality report → {out_path}")

    # Exit code: non-zero if >15% fail
    fail_rate = fail_n / total if total > 0 else 0
    if fail_rate > 0.15:
        sys.exit(f"Quality check FAILED: {fail_rate:.0%} segments flagged as FAIL (threshold 15%)")


if __name__ == "__main__":
    main()
