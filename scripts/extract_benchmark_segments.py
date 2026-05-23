#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract parallel segments from moonlight.db for DhivehiMT-Bench.

Pulls aligned EN-DV sentence pairs from the presidency.gov.mv corpus
(government/institutional genre) and outputs JSONL for the benchmark
main set.

Usage::

    python scripts/extract_benchmark_segments.py \\
        --db data/moonlight.db \\
        --genre government \\
        --n 100 \\
        --out data/benchmark/main_set/government/ \\
        --seed 42

Output files:
    <out>/segments_raw.jsonl      all extracted candidates
    <out>/dev.jsonl               50 segments (dev split)
    <out>/devtest.jsonl           50 segments (devtest split; held-out)

The dev/devtest split mirrors FLORES-200 and makes devtest directly
submittable to OLDI.

Segment selection criteria (per benchmark-design-spec.md §2.3):
- Length: 15–60 words on the English side
- Both EN and DV text available via sentence_idx alignment
- No overlap with the kahzaabu eval article (id=29734)
- Published 2020-01-01 or later (recency filter)
- Deduplicated by content hash (no repeated sentences across articles)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import sys
from pathlib import Path

# Article IDs excluded from all eval sets to prevent eval contamination.
# These are the Moonlight eval articles used in the published ablation study.
_EXCLUDED_IDS: set[int] = {29734}

_WORD_MIN = 15
_WORD_MAX = 60
_CHAR_MIN = int(_WORD_MIN * 4.5)   # conservative char lower bound
_CHAR_MAX = int(_WORD_MAX * 7.0)   # conservative char upper bound

_DV_CHAR_MIN = 40   # Dhivehi is more compact; lower char floor


def _word_count(text: str) -> int:
    return len(text.split())


def _content_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()[:12]


def extract_government(
    conn: sqlite3.Connection,
    n: int,
    seed: int,
    excluded_ids: set[int] = _EXCLUDED_IDS,
) -> list[dict]:
    """Extract n aligned EN-DV sentence pairs from the government corpus."""
    rows = conn.execute(
        """
        SELECT
            sp_en.article_id,
            sp_en.sentence_idx,
            sp_en.text        AS en_text,
            sp_dv.text        AS dv_text,
            a_en.published_date,
            a_en.category,
            a_en.reference
        FROM sentence_pairs sp_en
        JOIN sentence_pairs sp_dv
          ON  sp_en.paired_article_id = sp_dv.article_id
          AND sp_en.sentence_idx      = sp_dv.sentence_idx
          AND sp_en.lang  = 'EN'
          AND sp_dv.lang  = 'DV'
        JOIN articles a_en
          ON  a_en.id       = sp_en.article_id
          AND a_en.language = 'EN'
        WHERE sp_en.text_len BETWEEN ? AND ?
          AND sp_dv.text_len >= ?
          AND a_en.published_date >= '2020-01-01'
          AND a_en.category IN ('press_release', 'speech', 'vp_speech')
        ORDER BY a_en.published_date DESC, sp_en.article_id, sp_en.sentence_idx
        """,
        (_CHAR_MIN, _CHAR_MAX, _DV_CHAR_MIN),
    ).fetchall()

    # Filter: word count, excluded articles, deduplicate by EN content hash
    seen_hashes: set[str] = set()
    candidates: list[dict] = []
    for row in rows:
        article_id, sentence_idx, en_text, dv_text, pub_date, category, reference = row
        if article_id in excluded_ids:
            continue
        wc = _word_count(en_text)
        if not (_WORD_MIN <= wc <= _WORD_MAX):
            continue
        h = _content_hash(en_text)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        candidates.append(
            {
                "article_id": article_id,
                "sentence_idx": sentence_idx,
                "en_text": en_text,
                "dv_text": dv_text,
                "published_date": pub_date,
                "category": category,
                "reference": reference,
                "word_count": wc,
            }
        )

    if len(candidates) < n:
        print(
            f"WARNING: only {len(candidates)} candidates found, requested {n}.",
            file=sys.stderr,
        )
        n = len(candidates)

    rng = random.Random(seed)
    selected = rng.sample(candidates, n)
    selected.sort(key=lambda x: (x["published_date"], x["article_id"], x["sentence_idx"]))
    return selected


def _to_segment(
    item: dict,
    idx: int,
    split: str,
    genre: str = "government",
    direction: str = "en_dv",
) -> dict:
    seg_id = f"{genre}_{direction}_{idx:04d}"
    src_lang, tgt_lang = ("EN", "DV") if direction == "en_dv" else ("DV", "EN")
    source = item["en_text"] if direction == "en_dv" else item["dv_text"]
    reference = item["dv_text"] if direction == "en_dv" else item["en_text"]
    return {
        "id": seg_id,
        "genre": genre,
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "source": source,
        "reference": reference,
        "source_article_id": item["article_id"],
        "source_sentence_idx": item["sentence_idx"],
        "published_date": item["published_date"],
        "category": item["category"],
        "reference_url": item["reference"],
        "split": split,
        "flores_compatible": True,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records)} segments → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract benchmark segments from moonlight.db."
    )
    parser.add_argument("--db", default="data/moonlight.db", metavar="PATH")
    parser.add_argument(
        "--genre",
        default="government",
        choices=["government"],
        help="Corpus genre to extract from (currently only 'government' is automated).",
    )
    parser.add_argument("--n", type=int, default=100, metavar="N",
                        help="Number of segments to extract (default: 100).")
    parser.add_argument("--out", default="data/benchmark/main_set/government/",
                        metavar="DIR")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--direction",
        default="en_dv",
        choices=["en_dv", "dv_en", "both"],
        help="Translation direction: en_dv (default), dv_en, or both.",
    )
    parser.add_argument("--raw-only", action="store_true",
                        help="Write raw candidates only; skip dev/devtest split.")
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        sys.exit(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    print(f"Extracting {args.n} {args.genre} segments (seed={args.seed}) …")
    selected = extract_government(conn, args.n, args.seed)
    conn.close()

    out_dir = Path(args.out)
    directions = ["en_dv", "dv_en"] if args.direction == "both" else [args.direction]

    for direction in directions:
        dir_out = out_dir / direction if args.direction == "both" else out_dir
        raw_records = [
            _to_segment(item, i + 1, "raw", args.genre, direction)
            for i, item in enumerate(selected)
        ]
        _write_jsonl(dir_out / "segments_raw.jsonl", raw_records)

        if args.raw_only:
            continue

        half = args.n // 2
        dev_items = selected[:half]
        devtest_items = selected[half:]

        dev_records = [
            _to_segment(item, i + 1, "dev", args.genre, direction)
            for i, item in enumerate(dev_items)
        ]
        devtest_records = [
            _to_segment(item, i + 1, "devtest", args.genre, direction)
            for i, item in enumerate(devtest_items)
        ]

        _write_jsonl(dir_out / "dev.jsonl", dev_records)
        _write_jsonl(dir_out / "devtest.jsonl", devtest_records)

    total = len(selected) * len(directions)
    print(f"\nDone. {total} segments across {len(directions)} direction(s).")
    print("\nNext steps:")
    print("  1. Manual quality review: verify EN↔DV sentence alignment")
    print("  2. Source news/social/religious genres from external corpora")
    print("  3. Run: python scripts/build_calibration_set.py (once all genres ready)")


if __name__ == "__main__":
    main()
