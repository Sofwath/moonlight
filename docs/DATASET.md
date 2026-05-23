# Dataset Documentation

The Moonlight corpus is a collection of paired English–Dhivehi documents scraped from the Maldives Presidency Office website (presidency.gov.mv). This document describes its structure, provenance, statistics, and quality characteristics.

---

## Source: presidency.gov.mv

The Presidency Office publishes all official communications on their public website. Every article — press releases, speeches, decrees, amendments — appears in both English and Dhivehi. This bilingual publication is the primary source of the corpus.

### URL pattern

Article URLs follow a predictable pattern:

```
English:  https://presidency.gov.mv/Press/Article/{article_id}
Dhivehi:  https://presidency.gov.mv/Press/Article/{article_id}?lang=dv
```

The `article_id` is a numeric string assigned by the Presidency's CMS. The EN and DV versions share the same ID, making article-level pairing trivial: scrape the article list, discover IDs, fetch each ID in both languages.

### Article list discovery

The article listing pages are paginated:

```
https://presidency.gov.mv/Press?page={n}
```

Each page lists article IDs with titles and dates. A complete scrape iterates all pages until no new IDs are found.

---

## Article Categories

The corpus spans five document types:

| Category | Description | Approximate share |
|---|---|---|
| `press_release` | Standard press releases covering cabinet meetings, state visits, appointments, and policy announcements | ~65% |
| `speech` | Full text of presidential speeches (delivered in Dhivehi; EN is a translation) | ~20% |
| `vp_speech` | Vice-Presidential speeches | ~5% |
| `amendment` | Constitutional and statutory amendments gazetted via the Presidency | ~5% |
| `decree` | Presidential decrees and executive orders | ~5% |

The category is inferred from the article title and URL segment, not from a stable API field. The classification heuristics are in `moonlight/corpus/parse.py`.

---

## Corpus Statistics

| Metric | Value |
|---|---|
| Total article pairs | 2,648 |
| Held-out evaluation set | 264 (10%, stratified by category and year) |
| Training/retrieval set | 2,384 |
| EN tokens (whitespace-split) | ~2.1M |
| DV tokens (whitespace-split) | ~1.9M |
| Sentence-pair alignments | ~38,000 |
| Date range | 2019 – present |
| Average EN article length | ~420 words |
| Average DV article length | ~380 whitespace units |

### Year distribution

The corpus is not uniformly distributed over time. Article volume increased significantly after 2020. The held-out split is stratified by year bucket to ensure evaluation covers the full date range:

| Period | Approx pairs |
|---|---|
| 2019 | ~200 |
| 2020 | ~450 |
| 2021 | ~550 |
| 2022 | ~600 |
| 2023 | ~550 |
| 2024+ | ~298 |

The 2020–2022 period is the densest because of elevated presidential communication volume during the COVID-19 response.

---

## Sentence-Level Alignment

Article pairs are aligned at sentence level using a modified Champollion aligner. The alignment process:

1. Splits EN and DV article bodies into sentences
2. Runs the aligner to produce (EN sentence, DV sentence, confidence score) triples
3. Discards pairs below a confidence threshold (default: 0.5)
4. Flags articles where fewer than 60% of sentences produce high-confidence alignments as `quality=low_confidence`

Alignment quality is lower for speeches than for press releases. Speeches are often composed in Dhivehi and translated to English with significant restructuring, so the sentence boundaries do not align as cleanly as a document that was produced in both languages simultaneously.

---

## Data Quality Notes

### Numeric mismatch flag

A paired article is flagged `quality=numeric_mismatch` if any numeric value present in the EN body cannot be found (as digit string) in the DV body. This catches cases where:
- The translator rendered a number in words instead of digits
- An amount was stated in a different currency denomination
- A year was omitted or abbreviated

Flagged pairs are included in the retrieval corpus but excluded from numeric F1 evaluation by default. They can be included with `--include-numeric-mismatch` during evaluation.

### Low-confidence alignment flag

Articles where sentence-level alignment confidence is below the 60% threshold are flagged `quality=low_confidence`. These are disproportionately speeches and older articles (pre-2020) where translation style was less consistent.

Low-confidence articles are included in article-level retrieval (the full article pair is usable as few-shot context even if sentence alignment is poor) but excluded from sentence-level translation memory retrieval.

### Arabic-script passages

Some decree and amendment texts contain passages quoted in Arabic script (Quranic verses, treaty text, legal formulas). The EN version renders these as transliterated text or paraphrase. These passages are not excluded from the corpus but are noted in the database — the `articles.quality` field will contain a JSON note for affected articles.

### Boilerplate contamination

PO press releases often begin with a formulaic opening paragraph (the equivalent of "The President's Office reports that..."). These boilerplate sentences appear many times in the corpus with minor variations. The sentence-level deduplication step in `moonlight/corpus/align.py` collapses near-duplicate sentence pairs to avoid inflating the apparent corpus size.

---

## Building the Corpus

### From a kahzaabu database

If you have a kahzaabu SQLite database with the presidency corpus already scraped:

```bash
python -m moonlight.corpus import \
    --source /path/to/kahzaabu.db \
    --out data/moonlight.db
```

This imports article pairs and their EN/DV content directly.

### From scratch (scrape)

```bash
# Scrape all articles (takes ~2–3 hours; respects robots.txt; 1.5s delay)
python -m moonlight.corpus scrape \
    --out data/moonlight.db \
    --delay 1.5

# Run sentence alignment
python -m moonlight.corpus align --db data/moonlight.db

# Build FTS5 indices
python -m moonlight.corpus index --db data/moonlight.db

# Build embedding vectors (requires sentence-transformers)
python -m moonlight.corpus embed --db data/moonlight.db
```

Building embeddings takes approximately 10 minutes on CPU (MiniLM-L12-v2 is fast). On a machine with a GPU, this drops to under 2 minutes.

---

## Licence and Attribution

The content of presidency.gov.mv is the official output of the Government of the Maldives and is their intellectual property. This project does not redistribute the raw corpus. The code to scrape and process the data is provided for research purposes under the MIT licence.

Researchers using this corpus should:
- Acknowledge the Presidency of the Maldives as the source of the underlying content
- Not redistribute the raw article text
- Respect the website's robots.txt and apply reasonable rate limiting when scraping
- Consider whether their use is consistent with Maldivian law on government publications

There is no explicit open-data licence on presidency.gov.mv content as of this writing. The scraping is done for non-commercial research purposes. Users of this project should make their own legal assessment.

Contact for public correspondence: Sofwathullah.Mohamed@gmail.com
