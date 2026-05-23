# DhivehiMT-Bench Data

Evaluation benchmark for English–Dhivehi (EN↔DV) machine translation.

See [`docs/benchmark-design-spec.md`](../../docs/benchmark-design-spec.md) for the full design.

## Structure

```
data/benchmark/
├── main_set/           400 parallel segments (4 genres × 100), FLORES+-compatible
│   ├── government/     presidency.gov.mv sourced; extracted from moonlight.db
│   ├── news/           edition.mv, mihaaru.com (to be sourced)
│   ├── social/         Twitter/X, Facebook DV public posts (to be sourced)
│   └── religious/      Friday sermon excerpts, Quran translations (to be sourced)
├── challenge_set/      160 contrastive pairs across 8 error categories
│   └── challenge_seed.jsonl   initial entries; grow to 160 with native speaker input
└── calibration_set/    50 segments for human ESA annotation
    └── calibration.jsonl      stratified sample; annotated by 2-3 humans
```

## Segment JSONL schema

```json
{
  "id": "gov_en_dv_001",
  "genre": "government",
  "source_lang": "EN",
  "target_lang": "DV",
  "source": "...",
  "reference": "...",
  "source_article_id": 27872,
  "source_sentence_idx": 0,
  "published_date": "2023-01-01",
  "split": "dev",
  "flores_compatible": true
}
```

`split` is `"dev"` or `"devtest"`. The 200-segment devtest is the FLORES+-compatible submission.

## Challenge pair JSONL schema

```json
{
  "id": "cat2_honorific_001",
  "category": "cat2_honorifics",
  "subcategory": "maldivian_president",
  "source_lang": "EN",
  "target_lang": "DV",
  "source": "...",
  "correct": "...",
  "incorrect": "...",
  "error_description": "...",
  "attested_source": "presidency.gov.mv",
  "verified": true
}
```

`verified: true` = confirmed by native DV speaker or attested PO text.

## Building the government genre slice

```bash
python scripts/extract_benchmark_segments.py \
    --db data/moonlight.db \
    --genre government \
    --n 100 \
    --out data/benchmark/main_set/government/ \
    --seed 42
```

This extracts 100 stratified parallel segments from the moonlight corpus.

## Status

| Component | Target | Status |
|-----------|--------|--------|
| Government segments | 100 | Extract with `extract_benchmark_segments.py` |
| News segments | 100 | To be sourced externally |
| Social/informal segments | 100 | To be sourced externally |
| Religious segments | 100 | To be sourced externally |
| Challenge set | 160 pairs | Seed: ~30 verified; 130 pending native review |
| Calibration set | 50 segments | Pending human annotation |
