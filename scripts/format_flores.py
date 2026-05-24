#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Format the DhivehiMT-Bench devtest as FLORES+/OLDI submission.

FLORES-200 / FLORES+ format:
  - One file per language per domain: <lang>.<domain>
  - Language codes: div_Thaa (Dhivehi/Thaana), eng_Latn (English/Latin)
  - Three domains: news (Wikinews-style), wiki (encyclopaedic), travel (Wikivoyage)
  - 1012 sentences per language (FLORES-200 devtest standard); our initial
    submission covers the government genre (200 segments)
  - One sentence per line; lines aligned across language files
  - License: CC BY 4.0
  - Metadata file: flores_metadata.json with segment IDs, sources, dates

OLDI submission package:
  - flores_submission/
    ├── div_Thaa.devtest        Dhivehi sentences (government domain)
    ├── eng_Latn.devtest        English sentences (government domain)
    ├── flores_metadata.json    Segment provenance, quality flags, license
    └── README.md               Submission description

Usage::

    python scripts/format_flores.py \\
        --input data/benchmark/main_set/government/en_dv/devtest.jsonl \\
        --dv-input data/benchmark/main_set/government/dv_en/devtest.jsonl \\
        --output-dir data/flores_submission/

    # Validate the output (line counts must match across language files)
    python scripts/format_flores.py --validate data/flores_submission/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_FLORES_README = """\
# DhivehiMT-Bench FLORES+ Contribution

Language: Dhivehi (div_Thaa) / English (eng_Latn)
Domain: Government/Institutional (Maldives Presidency Office press releases and speeches)
Source: https://presidency.gov.mv (native Dhivehi text with professional EN translation)
License: CC BY 4.0
Submitted to: OLDI (Open Language Data Initiative)

## File contents

| File | Description |
|------|-------------|
| div_Thaa.devtest | Dhivehi sentences in Thaana script |
| eng_Latn.devtest | Corresponding English sentences |
| flores_metadata.json | Segment provenance, quality flags, publication dates |

## Quality

Segments are sourced from paired EN-DV government press releases and speeches.
Alignment is at sentence level by position index (approximate); 94%+ pass the
DhivehiMT-Bench alignment quality gate (length ratio, shared numbers/years,
Thaana presence checks).

## Notes

- This is a 200-segment devtest subset covering the government/institutional domain.
  The full DhivehiMT-Bench (400 segments, 4 genres) is under development.
- Segments span 2020–2026 publications.
- The Arabic comma (U+060C) appears in Dhivehi text as standard punctuation —
  this is correct and not a script contamination artefact.
- Religious domain text (not included here) may contain Arabic script for
  Quranic passages; that domain is reported separately in benchmark results.
"""


def load_jsonl(path: Path) -> list[dict]:
    segs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                segs.append(json.loads(line))
    return segs


def format_flores_submission(
    en_dv_segs: list[dict],
    dv_en_segs: list[dict] | None,
    output_dir: Path,
) -> None:
    """Write FLORES+-format files from benchmark segment JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use EN→DV segments as the primary source of text pairs.
    # Both directions come from the same underlying articles, so we merge.
    # Deduplicate by (source_article_id, source_sentence_idx).
    seen = set()
    pairs: list[dict] = []

    for seg in en_dv_segs:
        key = (seg.get("source_article_id"), seg.get("source_sentence_idx"))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "segment_id": seg["id"],
            "en": seg["source"],
            "dv": seg["reference"],
            "article_id": seg.get("source_article_id"),
            "sentence_idx": seg.get("source_sentence_idx"),
            "published_date": seg.get("published_date", ""),
            "category": seg.get("category", ""),
            "reference_url": seg.get("reference_url", ""),
            "split": seg.get("split", "devtest"),
        })

    # If dv_en_segs provided, fill in any gaps (DV source = EN reference there)
    if dv_en_segs:
        for seg in dv_en_segs:
            key = (seg.get("source_article_id"), seg.get("source_sentence_idx"))
            if key not in seen:
                seen.add(key)
                pairs.append({
                    "segment_id": seg["id"],
                    "en": seg["reference"],
                    "dv": seg["source"],
                    "article_id": seg.get("source_article_id"),
                    "sentence_idx": seg.get("source_sentence_idx"),
                    "published_date": seg.get("published_date", ""),
                    "category": seg.get("category", ""),
                    "reference_url": seg.get("reference_url", ""),
                    "split": seg.get("split", "devtest"),
                })

    # Sort by published_date, article_id, sentence_idx for reproducibility
    pairs.sort(key=lambda x: (x["published_date"], x["article_id"] or 0, x["sentence_idx"] or 0))

    n = len(pairs)
    print(f"  {n} aligned segment pairs")

    # Write FLORES-format language files (one sentence per line).
    # Normalize embedded newlines — FLORES format requires exactly one sentence per line.
    def _normalize(text: str) -> str:
        return " ".join(text.split())

    dv_lines = [_normalize(p["dv"]) for p in pairs]
    en_lines = [_normalize(p["en"]) for p in pairs]

    (output_dir / "div_Thaa.devtest").write_text(
        "\n".join(dv_lines) + "\n", encoding="utf-8"
    )
    print(f"  wrote {output_dir / 'div_Thaa.devtest'} ({n} lines)")

    (output_dir / "eng_Latn.devtest").write_text(
        "\n".join(en_lines) + "\n", encoding="utf-8"
    )
    print(f"  wrote {output_dir / 'eng_Latn.devtest'} ({n} lines)")

    # Write metadata JSON
    metadata = {
        "benchmark": "DhivehiMT-Bench",
        "license": "CC BY 4.0",
        "language_pair": "eng_Latn–div_Thaa",
        "domain": "government_institutional",
        "source": "presidency.gov.mv",
        "n_segments": n,
        "generated_date": str(date.today()),
        "format": "FLORES+/OLDI devtest",
        "segments": [
            {
                "line": i + 1,
                "segment_id": p["segment_id"],
                "article_id": p["article_id"],
                "sentence_idx": p["sentence_idx"],
                "published_date": p["published_date"],
                "category": p["category"],
                "reference_url": p["reference_url"],
            }
            for i, p in enumerate(pairs)
        ],
    }
    with (output_dir / "flores_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  wrote {output_dir / 'flores_metadata.json'}")

    (output_dir / "README.md").write_text(_FLORES_README, encoding="utf-8")
    print(f"  wrote {output_dir / 'README.md'}")


def validate_submission(submission_dir: Path) -> bool:
    """Validate FLORES+ submission: line counts, encoding, Thaana presence."""
    ok = True
    required = ["div_Thaa.devtest", "eng_Latn.devtest", "flores_metadata.json", "README.md"]
    for fname in required:
        if not (submission_dir / fname).exists():
            print(f"  MISSING: {fname}")
            ok = False

    if not ok:
        return False

    dv_lines = (submission_dir / "div_Thaa.devtest").read_text(encoding="utf-8").splitlines()
    en_lines = (submission_dir / "eng_Latn.devtest").read_text(encoding="utf-8").splitlines()

    if len(dv_lines) != len(en_lines):
        print(f"  LINE COUNT MISMATCH: div_Thaa={len(dv_lines)}, eng_Latn={len(en_lines)}")
        ok = False
    else:
        print(f"  Line counts match: {len(dv_lines)} segments per language")

    # Thaana check
    THAANA_MIN, THAANA_MAX = 0x0780, 0x07BF
    no_thaana = [
        i + 1 for i, line in enumerate(dv_lines)
        if not any(THAANA_MIN <= ord(c) <= THAANA_MAX for c in line)
    ]
    if no_thaana:
        print(f"  WARNING: {len(no_thaana)} Dhivehi lines contain no Thaana characters: "
              f"lines {no_thaana[:5]}{'…' if len(no_thaana) > 5 else ''}")
    else:
        print(f"  All {len(dv_lines)} Dhivehi lines contain Thaana characters")

    # Empty lines
    empty_dv = [i + 1 for i, l in enumerate(dv_lines) if not l.strip()]
    empty_en = [i + 1 for i, l in enumerate(en_lines) if not l.strip()]
    if empty_dv or empty_en:
        print(f"  WARNING: empty lines — DV: {empty_dv}, EN: {empty_en}")

    if ok and not no_thaana and not empty_dv:
        print(f"  Validation PASSED")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format DhivehiMT-Bench devtest as FLORES+/OLDI submission."
    )
    parser.add_argument("--input", default=None, metavar="PATH",
                        help="EN→DV devtest JSONL (data/benchmark/main_set/government/en_dv/devtest.jsonl)")
    parser.add_argument("--dv-input", default=None, metavar="PATH",
                        help="DV→EN devtest JSONL (optional; adds unique pairs).")
    parser.add_argument("--output-dir", default="data/flores_submission/", metavar="DIR")
    parser.add_argument("--validate", default=None, metavar="DIR",
                        help="Validate an existing submission directory instead of generating.")
    args = parser.parse_args()

    if args.validate:
        val_dir = Path(args.validate).expanduser().resolve()
        print(f"Validating FLORES+ submission: {val_dir}")
        ok = validate_submission(val_dir)
        sys.exit(0 if ok else 1)

    if not args.input:
        sys.exit("--input is required (EN→DV devtest JSONL)")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        sys.exit(f"Input not found: {input_path}\nRun extract_benchmark_segments.py first.")

    dv_input = Path(args.dv_input).expanduser().resolve() if args.dv_input else None
    output_dir = Path(args.output_dir).expanduser().resolve()

    print(f"Generating FLORES+ submission …")
    print(f"  EN→DV input: {input_path}")
    if dv_input:
        print(f"  DV→EN input: {dv_input}")
    print(f"  Output dir:  {output_dir}")
    print()

    en_dv_segs = load_jsonl(input_path)
    dv_en_segs = load_jsonl(dv_input) if dv_input else None

    format_flores_submission(en_dv_segs, dv_en_segs, output_dir)

    print(f"\nValidating …")
    validate_submission(output_dir)

    print(f"\nDone. FLORES+ submission ready at: {output_dir}")
    print("Next: submit div_Thaa.devtest + eng_Latn.devtest to OLDI via https://oldi.org/")


if __name__ == "__main__":
    main()
