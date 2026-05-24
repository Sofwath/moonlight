#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replace FAIL-quality segments and produce clean dev/devtest splits.

After running extract_benchmark_segments.py, some segments will be flagged
as FAIL by check_alignment_quality.py (multi-signal alignment mismatches).
This script replaces them by drawing from a reserve pool extracted at a
different seed, then rebuilds the dev/devtest splits.

Usage::

    # Replace FAILs and regenerate clean splits (EN→DV direction)
    python scripts/finalize_benchmark_segments.py \\
        --db data/moonlight.db \\
        --raw data/benchmark/main_set/government/en_dv/segments_raw.jsonl \\
        --out data/benchmark/main_set/government/en_dv/

    # Dry run — show what would be replaced without writing
    python scripts/finalize_benchmark_segments.py --dry-run \\
        --db data/moonlight.db \\
        --raw data/benchmark/main_set/government/en_dv/segments_raw.jsonl \\
        --out data/benchmark/main_set/government/en_dv/

After finalisation, commit the clean dev.jsonl and devtest.jsonl to git.
The segments_raw.jsonl and this script's output log remain gitignored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _run_quality_check(segments: list[dict]) -> dict[str, str]:
    """Return {segment_id: status} for each segment."""
    import re
    THAANA_RANGE = (0x0780, 0x07BF)
    ARABIC_LETTER_MIN = 0x0621
    ARABIC_LETTER_MAX = 0x06FF
    ARABIC_INDIC_NUMS = set(range(0x0660, 0x066A))
    # Direction-specific thresholds (EN→DV: DV is longer; DV→EN: EN is shorter)
    LEN_RATIO_BOUNDS = {"en_dv": (1.0, 3.0), "dv_en": (0.35, 1.2)}

    def _has_thaana(t): return any(THAANA_RANGE[0] <= ord(c) <= THAANA_RANGE[1] for c in t)
    def _has_arabic(t): return any(
        ARABIC_LETTER_MIN <= ord(c) <= ARABIC_LETTER_MAX and ord(c) not in ARABIC_INDIC_NUMS
        for c in t)
    def _nums(t): return set(re.findall(r"\d+", t))
    def _years(t): return set(re.findall(r"\b(19|20)\d{2}\b", t))

    statuses = {}
    for seg in segments:
        src, ref = seg.get("source", ""), seg.get("reference", "")
        src_lang = seg.get("source_lang", "EN")
        tgt = seg.get("target_lang", "DV")
        direction = f"{src_lang.lower()}_{tgt.lower()}"
        ratio_min, ratio_max = LEN_RATIO_BOUNDS.get(direction, (0.5, 3.0))
        flags = []

        ratio = len(ref) / max(len(src), 1)
        if ratio < ratio_min: flags.append("len_ratio_low")
        elif ratio > ratio_max: flags.append("len_ratio_high")

        if tgt == "DV":
            if not _has_thaana(ref): flags.append("no_thaana")
            if _has_arabic(ref): flags.append("arabic_contamination")

        if seg.get("source_lang") == "EN" and tgt == "DV":
            if _nums(src) - _nums(ref): flags.append("missing_numbers")
            if _years(src) - _years(ref): flags.append("missing_years")

        n = len(flags)
        statuses[seg["id"]] = "FAIL" if n >= 2 else ("WARN" if n == 1 else "PASS")
    return statuses


def _load_used_pairs(segments: list[dict]) -> set[tuple[int, int]]:
    return {
        (s["source_article_id"], s["source_sentence_idx"])
        for s in segments
        if "source_article_id" in s and "source_sentence_idx" in s
    }


def _extract_reserve(
    db_path: Path,
    direction: str,
    exclude_pairs: set[tuple[int, int]],
    n: int = 200,
    seed: int = 999,
) -> list[dict]:
    """Extract a reserve pool of candidate segments, excluding already-used pairs."""
    import sqlite3, random, hashlib

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT sp_en.article_id, sp_en.sentence_idx,
               sp_en.text AS en_text, sp_dv.text AS dv_text,
               a_en.published_date, a_en.category, a_en.reference
        FROM sentence_pairs sp_en
        JOIN sentence_pairs sp_dv
          ON sp_en.paired_article_id = sp_dv.article_id
          AND sp_en.sentence_idx = sp_dv.sentence_idx
          AND sp_en.lang = 'EN' AND sp_dv.lang = 'DV'
        JOIN articles a_en ON a_en.id = sp_en.article_id AND a_en.language = 'EN'
        WHERE sp_en.text_len BETWEEN 80 AND 350
          AND sp_dv.text_len >= 40
          AND a_en.published_date >= '2020-01-01'
          AND a_en.category IN ('press_release', 'speech', 'vp_speech')
        ORDER BY a_en.published_date DESC, sp_en.article_id, sp_en.sentence_idx
        """,
    ).fetchall()
    conn.close()

    _EXCLUDED_IDS = {29734}

    def _wc(t): return len(t.split())
    def _hash(t): return hashlib.md5(t.strip().lower().encode()).hexdigest()[:12]

    seen_hashes: set[str] = set()
    candidates = []
    for row in rows:
        aid, sidx, en, dv, pub, cat, ref = row
        if aid in _EXCLUDED_IDS:
            continue
        if (aid, sidx) in exclude_pairs:
            continue
        wc = _wc(en)
        if not (15 <= wc <= 60):
            continue
        h = _hash(en)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        candidates.append({"article_id": aid, "sentence_idx": sidx,
                           "en_text": en, "dv_text": dv,
                           "published_date": pub, "category": cat,
                           "reference": ref})

    rng = __import__("random").Random(seed)
    sample_n = min(n, len(candidates))
    return rng.sample(candidates, sample_n)


def _to_segment(item: dict, seg_id: str, split: str, genre: str, direction: str) -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace FAIL segments and produce clean dev/devtest splits."
    )
    parser.add_argument("--db", default="data/moonlight.db", metavar="PATH")
    parser.add_argument("--raw", required=True, metavar="PATH",
                        help="Raw segments JSONL (from extract_benchmark_segments.py).")
    parser.add_argument("--out", required=True, metavar="DIR",
                        help="Output directory for clean dev.jsonl and devtest.jsonl.")
    parser.add_argument("--genre", default="government")
    parser.add_argument("--reserve-seed", type=int, default=999,
                        help="RNG seed for reserve pool extraction (default: 999).")
    parser.add_argument("--reserve-n", type=int, default=300,
                        help="Reserve pool size to draw replacements from.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be replaced without writing output.")
    args = parser.parse_args()

    raw_path = Path(args.raw).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()

    if not raw_path.exists():
        sys.exit(f"Raw segments file not found: {raw_path}")
    if not db_path.exists():
        sys.exit(f"Database not found: {db_path}")

    # Detect direction from first segment
    with raw_path.open(encoding="utf-8") as f:
        first = json.loads(f.readline())
    direction = f"{first['source_lang'].lower()}_{first['target_lang'].lower()}"

    # Load raw segments
    raw_segments: list[dict] = []
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_segments.append(json.loads(line))

    print(f"Loaded {len(raw_segments)} raw segments ({direction})")

    # Quality check
    statuses = _run_quality_check(raw_segments)
    fail_ids = {sid for sid, s in statuses.items() if s == "FAIL"}
    pass_count = sum(1 for s in statuses.values() if s == "PASS")
    warn_count = sum(1 for s in statuses.values() if s == "WARN")

    print(f"  PASS: {pass_count}  WARN: {warn_count}  FAIL: {len(fail_ids)}")

    if not fail_ids:
        print("  No FAILs — nothing to replace. Output = input.")
        if args.dry_run:
            return
    else:
        print(f"\n  FAILs to replace: {sorted(fail_ids)}")
        if args.dry_run:
            print("  (dry run — no changes written)")
            return

    # Extract reserve pool
    used_pairs = _load_used_pairs(raw_segments)
    print(f"\nExtracting reserve pool (seed={args.reserve_seed}) …")
    reserve = _extract_reserve(db_path, direction, used_pairs, args.reserve_n, args.reserve_seed)
    print(f"  {len(reserve)} reserve candidates")

    # Quality-check reserves, find PASS candidates
    reserve_segs_tmp = []
    for i, item in enumerate(reserve):
        seg = _to_segment(item, f"reserve_{i:04d}", "raw", args.genre, direction)
        reserve_segs_tmp.append(seg)

    reserve_statuses = _run_quality_check(reserve_segs_tmp)
    pass_reserves = [
        (reserve_segs_tmp[i], reserve[i])
        for i, (seg, status) in enumerate(zip(reserve_segs_tmp, reserve_statuses.values()))
        if status == "PASS"
    ]
    print(f"  {len(pass_reserves)} PASS reserves available")

    if len(pass_reserves) < len(fail_ids):
        print(f"  WARNING: only {len(pass_reserves)} replacements available for {len(fail_ids)} FAILs")

    # Build replacement map
    replacements: dict[str, dict] = {}  # fail_id → item from reserve
    reserve_iter = iter(pass_reserves)
    for fail_id in sorted(fail_ids):
        try:
            _, reserve_item = next(reserve_iter)
            replacements[fail_id] = reserve_item
            print(f"  Replace {fail_id} ← article {reserve_item['article_id']}/{reserve_item['sentence_idx']}")
        except StopIteration:
            print(f"  WARNING: no replacement found for {fail_id} — keeping FAIL segment")

    # Rebuild final segment list, preserving original ordering and IDs
    final_segments: list[dict] = []
    for seg in raw_segments:
        if seg["id"] in replacements:
            item = replacements[seg["id"]]
            new_seg = _to_segment(item, seg["id"], "raw", args.genre, direction)
            final_segments.append(new_seg)
        else:
            final_segments.append(seg)

    # Verify no FAILs remain
    final_statuses = _run_quality_check(final_segments)
    remaining_fails = {sid for sid, s in final_statuses.items() if s == "FAIL"}
    if remaining_fails:
        print(f"\nWARNING: {len(remaining_fails)} FAILs remain after replacement: {remaining_fails}")
    else:
        print(f"\nAll FAILs replaced. Final quality:")
    final_pass = sum(1 for s in final_statuses.values() if s == "PASS")
    final_warn = sum(1 for s in final_statuses.values() if s == "WARN")
    final_fail = len(remaining_fails)
    print(f"  PASS: {final_pass}  WARN: {final_warn}  FAIL: {final_fail}")

    # Write clean dev/devtest splits
    n = len(final_segments)
    half = n // 2
    dev_segs = []
    devtest_segs = []
    for i, seg in enumerate(final_segments):
        seg = dict(seg)
        seg["split"] = "dev" if i < half else "devtest"
        if i < half:
            dev_segs.append(seg)
        else:
            devtest_segs.append(seg)

    def _write(path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {len(records)} → {path}")

    print()
    _write(out_dir / "dev.jsonl", dev_segs)
    _write(out_dir / "devtest.jsonl", devtest_segs)
    print(f"\nDone. Clean splits written to {out_dir}")
    print("Review the 6 replaced segments before committing dev.jsonl / devtest.jsonl.")


if __name__ == "__main__":
    main()
