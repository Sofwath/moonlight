# Architecture

This document describes the internal design of the Moonlight translation engine: data flow, module boundaries, database schema, retrieval pipeline, prompt construction, candidate scoring, and evaluation methodology.

---

## Table of Contents

- [Data Flow](#data-flow)
- [Module Dependency Graph](#module-dependency-graph)
- [Database Schema](#database-schema)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Prompt Construction](#prompt-construction)
- [Candidate Scoring](#candidate-scoring)
- [Web Layer](#web-layer)
- [Evaluation Methodology](#evaluation-methodology)

---

## Data Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│  OFFLINE (corpus build — runs once, or on re-scrape)                   │
│                                                                          │
│  presidency.gov.mv ──scrape──► raw HTML/JSON                            │
│                                     │                                   │
│                                     ▼                                   │
│                            moonlight.corpus.parse                       │
│                            (extract EN + DV text, pair by article ID)  │
│                                     │                                   │
│                                     ▼                                   │
│                            moonlight.corpus.align                       │
│                            (sentence-level alignment, Champollion)      │
│                                     │                                   │
│                                     ▼                                   │
│                            SQLite: articles, sentence_pairs             │
│                                     │                                   │
│                                     ▼                                   │
│                            moonlight.corpus.embed                       │
│                            (MiniLM-L12-v2 → float32 vectors)           │
│                                     │                                   │
│                                     ▼                                   │
│                            SQLite: sentence_pair_embeddings             │
│                            + FTS5 virtual tables (articles_fts,         │
│                              sentence_pairs_fts)                        │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│  ONLINE (translation — per request)                                     │
│                                                                          │
│  Input text                                                              │
│       │                                                                  │
│       ▼                                                                  │
│  moonlight.detect (Thaana codepoint fraction → direction)               │
│       │                                                                  │
│       ▼                                                                  │
│  moonlight.retrieve                                                      │
│  ├── HyDE (EN→DV): generate cheap DV hypothesis → embed DV↔DV         │
│  ├── FTS5 BM25 query against articles_fts + sentence_pairs_fts          │
│  ├── Embedding similarity query against sentence_pair_embeddings        │
│  └── Reciprocal Rank Fusion → ranked list of (article_pair, sent_pair) │
│       │                                                                  │
│       ▼                                                                  │
│  moonlight.placenames                                                    │
│  └── Lookup place name mentions → inject romanisation map               │
│       │                                                                  │
│       ▼                                                                  │
│  moonlight.prompt                                                        │
│  ├── Layer 1: system instruction (mode-dependent)                       │
│  ├── Layer 2: glossary injection (26,771 PO-attested EN↔DV terms)      │
│  ├── Layer 3: sentence TM (top-k sentence pairs from retrieval)         │
│  └── Layer 4: article few-shot (2–3 full article pairs)                 │
│       │                                                                  │
│       ▼                                                                  │
│  LLM API call(s)                                                         │
│  ├── Single model: 1–3 candidates scored via MBR (chrF consensus)      │
│  └── multi_model=True: Claude Sonnet + Gemini Pro parallel → MBR pick  │
│       │                                                                  │
│       ▼                                                                  │
│  Post-generation gates                                                   │
│  ├── MBR selection: pairwise chrF consensus picks the winner            │
│  ├── Foreign script sanitizer: removes stray CJK/Arabic/Thaana chars   │
│  └── Entity check: numbers, place names, titles must survive            │
│       │                                                                  │
│       ▼                                                                  │
│  Best candidate → TranslationResult                                     │
│       │                                                                  │
│       ▼                                                                  │
│  moonlight.store (write to translation_runs if logging enabled)         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
moonlight/
│
├── __init__.py            ← Translator public API
│        depends on: translate, retrieve, prompt, score, detect
│
├── translate.py           ← Orchestration: retrieve → prompt → call → score → return
│        depends on: retrieve, prompt, score, llm, detect, store
│
├── retrieve.py            ← Hybrid retrieval: BM25 + embeddings + RRF
│        depends on: db, embed
│
├── prompt.py              ← Prompt construction (4 layers)
│        depends on: db, glossary, placenames
│
├── score.py               ← Candidate scoring: numeric_f1, entity_recall, length_ratio
│        depends on: (pure: no DB calls)
│
├── detect.py              ← Language/direction detection from input
│        depends on: (pure: regex on Unicode ranges)
│
├── embed.py               ← Sentence embedding wrapper (MiniLM-L12-v2)
│        depends on: sentence_transformers (lazy import)
│
├── llm.py                 ← LLM API wrapper (Anthropic + fallback)
│        depends on: anthropic SDK
│
├── store.py               ← Write translation_runs to DB for logging
│        depends on: db
│
├── placenames.py          ← Place name reference lookup
│        depends on: db
│
├── glossary.py            ← Term matching against translation_glossary
│        depends on: db
│
├── db.py                  ← SQLite connection management
│        depends on: sqlite3 (stdlib)
│
├── corpus/
│    ├── scrape.py         ← Scrape presidency.gov.mv
│    ├── parse.py          ← Extract and clean article text
│    ├── align.py          ← Sentence-level alignment
│    ├── embed.py          ← Batch embedding generation
│    └── import_.py        ← Import from kahzaabu DB
│
└── web/
     ├── app.py            ← FastAPI app: routes, CORS, rate-limit middleware
     ├── db_dep.py         ← FastAPI dependency: per-request SQLite connection
     ├── limits.py         ← slowapi limiter + daily spend cap constants
     └── api/
          ├── translate.py     ← POST /api/translate
          ├── concordance.py   ← GET  /api/concordance
          ├── glossary_api.py  ← GET  /api/glossary
          ├── align_batch.py   ← POST /api/align-batch  (cached word alignment)
          ├── alternatives.py  ← POST /api/alternatives
          ├── ner.py           ← POST /api/ner
          ├── spellcheck.py    ← POST /api/spellcheck
          ├── fluency.py       ← POST /api/fluency
          ├── history.py       ← GET  /api/translate/history
          └── benchmarks.py    ← GET  /api/benchmarks
```

Key principle: `score.py` and `detect.py` have no DB dependencies — they operate on strings only, which makes them trivially testable in isolation.

---

## Database Schema

All state lives in a single SQLite file (default: `data/moonlight.db`). SQLite was chosen over a vector database because: (a) the corpus is small enough for SQLite to handle comfortably, (b) FTS5 is built in, (c) embedding similarity search over ~38,000 vectors is fast enough with numpy dot-product in Python, and (d) one file is easy to back up and share.

### `articles`

Stores paired article content at the document level. It uses a composite primary key `(id, language)` to support the paired bilingual corpus model.

```sql
CREATE TABLE articles (
    id              INTEGER NOT NULL,
    language        TEXT    NOT NULL CHECK(language IN ('EN', 'DV')),
    paired_id       INTEGER,
    category        TEXT,
    category_id     INTEGER,
    title           TEXT,
    body_text       TEXT,
    body_html       TEXT,
    reference       TEXT,
    published_date  TEXT,
    image_urls      TEXT,     -- JSON array of URL strings
    raw_page_html   TEXT,
    scraped_at      TEXT,
    content_hash    TEXT,
    PRIMARY KEY (id, language)
);
CREATE INDEX idx_articles_paired   ON articles(paired_id, language);
CREATE INDEX idx_articles_lang     ON articles(language, published_date);
CREATE INDEX idx_articles_category ON articles(category, language);
```

### `articles_fts`

FTS5 virtual table over article content with automatic synchronization triggers. Used for BM25 retrieval at the article level.

```sql
CREATE VIRTUAL TABLE articles_fts USING fts5(
    article_id UNINDEXED,
    language UNINDEXED,
    title,
    body,
    tokenize='unicode61'
);
```

### `translation_glossary`

Domain-specific term pairs. Populated from a seed list and augmented by automated term extraction from the corpus.

```sql
CREATE TABLE translation_glossary (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    en_term       TEXT NOT NULL,
    dv_term       TEXT NOT NULL,
    domain        TEXT,
    freq          INTEGER NOT NULL,
    confidence    REAL,
    sample_en_ids TEXT,   -- JSON array of article ids where this pair was found
    extracted_at  TEXT NOT NULL,
    extracted_by  TEXT NOT NULL
);
CREATE INDEX idx_glossary_en ON translation_glossary(en_term);
CREATE INDEX idx_glossary_dv ON translation_glossary(dv_term);
```

### `translation_runs`

Logging and auditing table. Every translation request (when logging is enabled) writes here, which enables offline analysis, cost/performance tracking, and caching.

```sql
CREATE TABLE translation_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_lang         TEXT NOT NULL,
    target_lang         TEXT NOT NULL,
    input_text          TEXT NOT NULL,
    output_text         TEXT NOT NULL,
    mode                TEXT NOT NULL DEFAULT 'faithful',
    exemplar_ids        TEXT,    -- JSON array of article ids used as few-shot
    phrase_context_ids  TEXT,    -- JSON array
    glossary_terms_used INTEGER,
    n_candidates        INTEGER DEFAULT 1,
    model               TEXT NOT NULL,
    cost_usd            REAL,
    tokens_in           INTEGER,
    tokens_out          INTEGER,
    created_at          TEXT NOT NULL
);
CREATE INDEX idx_translation_runs_cache ON translation_runs(target_lang, created_at);
```

### `sentence_pairs`

Sentence-level segments extracted from the article pairs. These are the primary retrieval units for translation memory. They include embedding vectors inline for hybrid search.

```sql
CREATE TABLE sentence_pairs (
    id                  INTEGER PRIMARY KEY,
    article_id          INTEGER NOT NULL,
    paired_article_id   INTEGER,
    lang                TEXT NOT NULL,
    sentence_idx        INTEGER NOT NULL,
    text                TEXT NOT NULL,
    text_len            INTEGER NOT NULL,
    embedding           BLOB,
    embedding_model     TEXT,
    UNIQUE(article_id, lang, sentence_idx)
);
CREATE INDEX idx_sentpair_article ON sentence_pairs(article_id);
CREATE INDEX idx_sentpair_lang ON sentence_pairs(lang);
```

### `sentence_pairs_fts`

FTS5 virtual table over sentence pairs. Used for BM25 retrieval at the sentence level (translation memory lookup).

```sql
CREATE VIRTUAL TABLE sentence_pairs_fts USING fts5(
    text,
    content='sentence_pairs',
    content_rowid='id',
    tokenize='unicode61'
);
```

### `alignment_cache`

Caches word-alignment results from the LLM to avoid redundant API calls. Entries expire after 24 hours (enforced by the query in `align_batch.py`, not by a DB trigger).

```sql
CREATE TABLE alignment_cache (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_text  TEXT NOT NULL,
    translation  TEXT NOT NULL,
    source_lang  TEXT NOT NULL,
    target_lang  TEXT NOT NULL,
    alignments   TEXT NOT NULL,  -- JSON array
    created_at   TEXT NOT NULL
);
```

### `place_names`

Reference table for Maldivian place names. Used to inject correct official romanisations and Thaana representations into prompts.

```sql
CREATE TABLE place_names (
    geonameid    INTEGER PRIMARY KEY,
    en_name      TEXT NOT NULL,
    dv_thaana    TEXT,
    dv_latin     TEXT,
    en_name_po   TEXT,
    feature_code TEXT,
    atoll_code   TEXT,
    latitude     REAL,
    longitude    REAL,
    source       TEXT NOT NULL DEFAULT 'geonames'
);
CREATE INDEX idx_place_names_thaana  ON place_names(dv_thaana);
CREATE INDEX idx_place_names_en      ON place_names(en_name);
CREATE INDEX idx_place_names_feature ON place_names(feature_code);
```
```

---

## Retrieval Pipeline

### Overview

Retrieval runs in two stages (sentence-level first, article-level second) and uses Reciprocal Rank Fusion (RRF) to combine BM25 and embedding scores.

### Stage 1: Sentence-level translation memory

**Goal**: Find sentence pairs from the corpus that are most similar to the input text.

**Hybrid BM25 + Semantic Search**:
1. Split the input text into individual sentences.
2. For each sentence, use the FTS5 virtual table `sentence_pairs_fts` to retrieve the top candidate pools (e.g., top-30) via BM25 matching.
3. Compute the cosine-normalized embedding vector for the input sentence using the multilingual model.
4. Fetch precomputed embedding BLOBs for the candidate set from `sentence_pairs` and perform vector similarity (dot-product) scoring.
5. Sort the candidates by similarity score and return the top `k` matching sentence pairs.

If embeddings are not initialized or sentence-transformers is not available, the system gracefully falls back to pure BM25 retrieval.

### Stage 2: Article-level few-shot retrieval

**Goal**: Find full article pairs (EN body + DV body) that are topically similar to the input.

The top-10 sentence pairs from Stage 1 are mapped back to their parent articles. The article-level retrieval reuses the BM25 rankings from Stage 1 (no separate article-level embedding query) and selects the top 3 unique articles.

### Fallback

If embeddings are unavailable (model not installed, or `--no-embed` flag), Stage 1 runs BM25 only. This degrades semantic coverage but keeps the system functional. The fallback is logged at WARNING level.

### Deduplication

Retrieved sentence pairs are deduplicated: if the same EN sentence appears in multiple article pairs (which occasionally happens when the PO re-uses boilerplate), only the highest-scoring instance is kept.

---

## Prompt Construction

Each translation request assembles a prompt from four layers in order. The prompt is assembled as a list of messages for the chat API (system + user turn).

### Layer 1: System instruction

Stable across requests; varies by mode.

```
faithful mode:
  "You are a professional translator between English and Dhivehi.
   Preserve ALL numeric values, dates, amounts, and proper nouns exactly.
   Do not paraphrase, summarise, or add contextual explanation.
   If a term has no direct translation, transliterate it.
   Output only the translation. No preamble, no commentary."

po_style mode:
  "You are a professional translator working in the style of the Maldives
   Presidency Office. Translations should use the formal Dhivehi register
   used in official press releases: [specific honorifics], [date format],
   [amount format]. Prefer idiomatic Dhivehi expression over literal
   word-for-word rendering where the PO convention supports it.
   Output only the translation."
```

### Layer 2: Terminology glossary

Matched terms are injected as a structured block. The glossary lookup queries `translation_glossary` for terms that appear in the input text (case-insensitive for EN, exact-match for DV Thaana).

```
TERMINOLOGY:
- President / ރައީސް
- Cabinet / ވަޒީރުންގެ މަޖިލިސް
- Male' City / މާލެ ސިޓީ
[...matched terms only, not the full glossary...]
```

If the input is DV→EN and place names are detected, the place name reference map is appended here:

```
PLACE NAME REFERENCE:
- ކަނޑިތީމު → Kan'ditheemu (Shaviyani Atoll)
- ހދ. ކުޅުދުއްފުށި → Kulhudhuffushi (HDh. Atoll)
```

### Layer 3: Sentence-level translation memory

Top-10 sentence pairs from retrieval, formatted as parallel examples:

```
TRANSLATION MEMORY (most relevant examples from Presidency Office texts):

EN: The President chaired the meeting of the Cabinet.
DV: ރައީސް ވަނީ ވަޒީރުންގެ މަޖިލިހުގެ ބައްދަލުވުން ރިޔާސަތު ކުރައްވާ ހިންގަވާ ދެއްވާފައެވެ.

EN: The meeting was held at the President's Office.
DV: ބައްދަލުވުން ބޭއްވީ ރައީސް އޮފީހުގައެވެ.

[...up to 10 pairs...]
```

Sentence pairs are ordered by RRF score (highest first). If retrieval returns fewer than 5 pairs, a WARNING is logged and the prompt proceeds with what is available.

### Layer 4: Article-level few-shot exemplars

2–3 full article pairs (truncated at 600 tokens each to stay within context limits). These provide structural context: how a full PO press release is laid out, how it opens and closes, and how it handles transitions.

```
FULL ARTICLE EXAMPLES:

[Article 1]
EN: [full EN article text, truncated]
DV: [full DV article text, truncated]

[Article 2]
EN: [...]
DV: [...]
```

### Retrieval payload contract (load-bearing)

The translator expects a canonical internal shape for prompt assembly:

```
{
  "en_article_id": int,
  "dv_article_id": int | null,
  "en_body": str,
  "dv_body": str,
  "published_date": str,
  "en_title": str
}
```

`moonlight.corpus.select_few_shot()` currently emits a retrieval-oriented shape:

```
{
  "article_id": int,
  "source_body": str,
  "target_body": str,
  "published_date": str,
  "category": str
}
```

`moonlight.translator.translate()` normalizes this via `_coerce_exemplars()` before calling `_compose_prompt()`. This boundary is intentional: retrieval code can evolve independently, while prompt logic remains stable on one schema.

Phrase contexts follow the same rule. Prompt assembly accepts both:
- canonical keys: `source_snippet` / `target_snippet`
- retrieval keys: `snippet` (with missing target snippet rendered as a placeholder note)

Do not bypass this normalization boundary when adding new retrieval fields. If retrieval output changes, update the coercion layer and keep prompt input schema stable.

### User turn

```
Translate the following [English / Dhivehi] text to [Dhivehi / English]:

[input text]
```

### Total prompt size

Typical prompt sizes:

| Component | Approx tokens |
|---|---|
| System instruction | 80–120 |
| Glossary (matched terms only) | 50–200 |
| Sentence TM (10 pairs) | 400–800 |
| Article few-shot (2 articles, truncated) | 800–1200 |
| User turn | 20–500 |
| **Total** | **~1350–2820** |

This comfortably fits within the context window of all current frontier models and keeps per-request API cost reasonable.

---

## Candidate Scoring

Three candidates are requested per translation (configurable with `--n-candidates`). Each candidate is scored independently and the highest-scoring candidate is returned.

### Numeric F1

Extracts numeric tokens from input and checks their presence in the output.

```
Input tokens:  regex match on [0-9,./]+ and Thaana numeral equivalents
Output tokens: same regex on candidate text
precision = |output_nums ∩ input_nums| / |output_nums|
recall    = |output_nums ∩ input_nums| / |input_nums|
numeric_f1 = 2 * precision * recall / (precision + recall)
           = 1.0 if input has no numerics (no penalty)
```

"Numeric tokens" here means the digit strings themselves. The scoring does not check semantic equivalence of numeric expressions (e.g., it will not flag "12,000" vs "12000" as a mismatch — both contain the digits 12000). More sophisticated numeric normalisation is a known improvement area.

### Entity recall

Checks whether place names identified in the input appear in the output.

```
input_entities  = place_name_lookup(input_text)  # uses place_names table
output_entities = place_name_lookup(candidate_text)
entity_recall   = |output_entities ∩ input_entities| / |input_entities|
                = 1.0 if no entities found in input
```

This is recall-only (not F1) because we want to penalise missing entities but not penalise the model for adding entities that appear in the retrieved context.

### Length ratio

```
expected_ratio = {
    "en→dv": 0.75,   # DV is typically shorter by character count
    "dv→en": 1.35,   # EN tends to be longer
}
actual_ratio = len(candidate) / len(input_text)  # character lengths
deviation = abs(actual_ratio - expected_ratio[direction]) / expected_ratio[direction]
length_score = max(0.0, 1.0 - deviation)  # 0.0 if >100% deviation
```

Expected ratios are derived empirically from the corpus. A candidate that is half the expected length is almost certainly truncated; a candidate that is twice the expected length has likely added content.

### Final score

```
score = (
    0.50 * numeric_f1 +
    0.30 * entity_recall +
    0.20 * length_score
)
```

The numeric_f1 weight dominates because numeric errors are the most consequential failures in practice. The best candidate by this score is returned. All candidates and their scores are available on `TranslationResult.candidates` for inspection.

---

## Web Layer

The web layer is a thin FastAPI application that wraps the core translation engine. It adds no business logic — all translation logic lives in `moonlight/translator.py`.

### Middleware stack (applied in order)

1. **SlowAPIMiddleware** — rate limiting via `slowapi`. Limits are per-endpoint (e.g. `10/minute` for `/api/translate`, `30/minute` for `/api/align-batch`). A daily USD spend cap is enforced inside the `/api/translate` handler before calling the LLM.
2. **CORSMiddleware** — allowed origins controlled by the `CORS_ORIGINS` environment variable (comma-separated). Defaults to `http://localhost:8000` for local development.
3. **`_no_cache_html` middleware** — sets `Cache-Control: no-store` on HTML responses so the workbench always reflects the latest build.

### Request lifecycle for POST /api/translate

```
HTTP request
    │
    ▼
CORSMiddleware (preflight / header injection)
    │
    ▼
SlowAPIMiddleware (rate check: 10/minute per IP)
    │
    ▼
translate() handler
    ├── check daily spend cap (translation_runs table)
    ├── validate ablate set
    ├── call moonlight.translator.translate()
    ├── attach glossary_terms (SQL lookup on translation_glossary)
    ├── enrich phrase_contexts with target-side snippets
    └── return JSON response
```

### Static assets

The workbench UI is served from `moonlight/web/static/`. Alpine.js is vendored at `static/js/alpine.min.js` — no build step, no npm. The JS app (`static/js/workbench.js`) is a single Alpine component that drives all tab state, token interaction, and API calls.

### DB dependency injection

FastAPI's `Depends(get_db)` pattern is used for SQLite connections. `get_db()` in `db_dep.py` opens a connection using the path from the `MOONLIGHT_DB` environment variable (default: `data/moonlight.db`) and closes it after the response completes. This keeps connection lifetime scoped to the request, which is correct for SQLite in a single-process server.

---

## Evaluation Methodology

### Split strategy

The held-out evaluation set is constructed at corpus-build time with stratified sampling:

```
stratify by: category × year_bucket (pre-2021, 2021-2022, 2023+)
held-out fraction: 10% (default)
seed: 42 (reproducible)
```

Articles in the held-out set are excluded from the FTS5 index and the embedding matrix during evaluation. This prevents retrieval from returning the reference article as a few-shot exemplar — which would inflate scores artificially.

### Metric computation

BLEU and chrF are computed using sacrebleu. For Dhivehi output:

```python
# sacrebleu chrF call (character n-gram, no word boundary assumptions)
chrf = corpus_chrf(hypotheses, [references], char_order=6, word_order=0)

# sacrebleu BLEU call with character tokenisation for DV
bleu = corpus_bleu(hypotheses, [references], tokenize='char')
```

For English output (DV→EN), standard 13a tokenisation is used.

### Evaluation harness

`moonlight.eval.run` iterates over held-out articles, calls `Translator.translate()` for each, records the result in `translation_runs`, and at the end computes corpus-level metrics.

```
for article in held_out:
    result = translator.translate(article.en_body OR article.dv_body, ...)
    record(result, reference=article.dv_body OR article.en_body)
compute_corpus_metrics(records)
```

Corpus-level BLEU (not sentence-average BLEU) is reported, as sentence-average BLEU is unstable on short segments.

### Ablation harness

The ablation harness runs `moonlight.eval.run` four times with different retrieval configurations:

| Condition | Config override |
|---|---|
| `full` | default |
| `no_retrieval` | `retrieval=None` |
| `bm25_only` | `retrieval=bm25` |
| `embed_only` | `retrieval=embed` |

Results are written to `results/ablation_{condition}_{mode}_{direction}.json` and can be compared with `moonlight eval compare results/ablation_*.json`.
