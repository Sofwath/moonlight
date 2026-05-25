# Moonlight

> *Moonlight* was the Maldives' first English-language daily newspaper, published during the late Nasir era and ceasing publication in December 1978 — after which Haveeru Daily was launched to fill the gap. This project borrows the name to honour that early experiment in English-language publishing in the Maldives, and because the work here is similarly about bridging Dhivehi and English — carefully, in context, with attention to register.

Moonlight is a standalone English ↔ Dhivehi translation engine that uses retrieval-augmented prompting against a paired corpus of ~2,648 Presidency Office press releases. It is extracted from the [kahzaabu](https://github.com/sofwath/kahzaabu) fact-checking pipeline and designed to operate independently.

---

## Demo

<video src="demo/moonlight_workbench_demo.mp4" controls width="100%">
  <a href="demo/moonlight_workbench_demo.mp4">Watch the workbench demo (1:04)</a>
</video>

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Dhivehi Language Notes](#dhivehi-language-notes)
- [Design Philosophy](#design-philosophy)
- [Evaluation](#evaluation)
- [Research Findings](#research-findings)
- [Limitations](#limitations)
- [Citation and Acknowledgements](#citation-and-acknowledgements)

---

## Problem Statement

### The gap generic MT systems leave

Dhivehi is spoken by roughly 500,000 people, almost all of them in the Maldives. It is written in Thaana script (Unicode block U+0780–U+07BF), runs right-to-left, and its formal written register diverges substantially from everyday speech. Arabic and Farsi loanwords are common in formal contexts; the vocabulary for governance, law, and diplomacy is largely borrowed and often transliterated inconsistently across sources.

General-purpose machine translation systems — even large, well-resourced ones — produce Dhivehi output that is technically intelligible but tonally wrong for official contexts. The Presidency Office (PO) of the Maldives uses a consistent, formal register in its press releases: specific honorifics, specific ways of rendering atoll and island names, and specific patterns for dates, amounts, and titles. A translation that deviates from this register sounds unprofessional to a Maldivian reader even if it is semantically accurate.

Moonlight addresses this by grounding every translation request in actual PO output: the real paired EN↔DV press releases the Presidency has published since the site's launch. When translating a sentence about a cabinet reshuffle, the model sees how the PO itself rendered similar sentences. When translating a place name, it consults a reference DB built from GeoNames MV data and a hardcoded atoll supplement.

### Why this was extracted from kahzaabu

The kahzaabu fact-checking pipeline needs translation as a sub-step — it reads DV articles, checks claims, and sometimes needs to produce EN summaries or vice versa. But translation quality requirements differ depending on whether the output goes into an automated pipeline (where claim preservation matters most) or into a human-readable product (where register and idiom matter). Embedding the full translation engine inside kahzaabu made it hard to develop, test, and improve the two modes independently. Moonlight separates that concern cleanly.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Moonlight Translation Engine                 │
│                                                                       │
│  Input text (EN or DV)                                                │
│         │                                                             │
│         ▼                                                             │
│  ┌─────────────────┐                                                  │
│  │  Language detect │  (script detection: Thaana → DV, else EN)      │
│  └────────┬────────┘                                                  │
│           │                                                           │
│           ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                     Retrieval Layer                           │    │
│  │                                                               │    │
│  │  ┌──────────────┐        ┌─────────────────────────────┐    │    │
│  │  │  FTS5 BM25   │        │  Multilingual Embeddings    │    │    │
│  │  │  (SQLite)    │        │  (paraphrase-MiniLM-L12-v2) │    │    │
│  │  └──────┬───────┘        └──────────────┬──────────────┘    │    │
│  │         │                               │                    │    │
│  │         └──────────┬────────────────────┘                    │    │
│  │                    ▼                                          │    │
│  │          Reciprocal Rank Fusion                               │    │
│  │          → top-k article pairs + sentence pairs              │    │
│  └──────────────────────────────────────────────────────────────┘    │
│           │                                                           │
│           ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                     Prompt Construction                        │    │
│  │                                                               │    │
│  │   Layer 1: System instruction + mode (faithful / po_style)   │    │
│  │   Layer 2: Terminology glossary (domain terms + place names) │    │
│  │   Layer 3: Sentence-level TM matches (5–10 pairs)            │    │
│  │   Layer 4: Article-level few-shot exemplars (2–3 full pairs) │    │
│  └──────────────────────────────────────────────────────────────┘    │
│           │                                                           │
│           ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │              Frontier LLM  (N=3 candidates)                   │    │
│  └──────────────────────────────────────────────────────────────┘    │
│           │                                                           │
│           ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    Candidate Scoring                           │    │
│  │                                                               │    │
│  │   • Numeric F1  (dates, amounts, percentages must survive)   │    │
│  │   • Entity recall (place names, person titles)               │    │
│  │   • Length ratio (penalise extreme compression/expansion)    │    │
│  │   → Best candidate selected                                  │    │
│  └──────────────────────────────────────────────────────────────┘    │
│           │                                                           │
│           ▼                                                           │
│  Output translation                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- A [kahzaabu](https://github.com/Sofwath/kahzaabu) SQLite database (for corpus import) **or** a pre-built `moonlight.db`
- An Anthropic API key (or compatible frontier LLM endpoint)

### Installation

```bash
git clone https://github.com/sofwath/moonlight
cd moonlight
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Build the corpus database

> If you have a [kahzaabu](https://github.com/Sofwath/kahzaabu) DB (`kahzaabu.db`), import from it.
> The kahzaabu DB contains the full paired EN+DV Presidency Office corpus and is the fastest way to get started.

```bash
# Clone kahzaabu and follow its README to build kahzaabu.db, then:
python scripts/import_from_kahzaabu.py --source /path/to/kahzaabu.db
# Default output: data/moonlight.db
```

> Or initialise an empty database and populate it manually:

```bash
moonlight db-init
moonlight db-stats   # verify schema is ready
```

> Build the embedding index (required for hybrid BM25+semantic retrieval; ~10 min first run):

```bash
moonlight build-embeddings
# Downloads paraphrase-multilingual-MiniLM-L12-v2 (~278 MB) on first run.
# Without this step the translator falls back to pure BM25 — still functional.
```

> Mine glossary terms from the corpus (~$0.05 per article pair):

```bash
moonlight build-glossary --sample 200 --budget 10.0
```

### Translate a sentence

```python
from moonlight.db import get_connection
from moonlight.translator import translate

conn = get_connection("data/moonlight.db")

# Dhivehi → English, faithful mode (for automated pipelines)
result = translate(
    conn,
    text="ރައީސް މިއަދު ވަނީ ވަޒީރުންގެ މަޖިލީހުގެ ބައްދަލުވުމެއް ބާއްވަވާފައެވެ.",
    target_lang="EN",
    mode="faithful",
)
print(result["translation"])
# → "The President held a meeting of the Cabinet today."

# English → Dhivehi, Presidency Office register
result = translate(
    conn,
    text="The President chaired a meeting of the Cabinet today.",
    target_lang="DV",
    mode="po_style",
)
print(result["translation"])
print(f"exemplars={len(result['exemplars'])}  "
      f"glossary_terms={result['glossary_terms_used']}  "
      f"cost=${result['cost_usd']:.4f}")

conn.close()
```

### CLI usage

```bash
# One-off translation (auto-detects language)
moonlight translate "ދިވެހިރާއްޖެ"

# Explicit target and mode
moonlight translate "The President signed the decree." --target DV --mode po_style

# Multi-candidate (Best-of-3, returns highest-scoring translation)
moonlight translate "..." --target EN --candidates 3

# JSON output with full provenance
moonlight translate "..." --json-output

# List all available models with pricing
moonlight models
```

### Run the evaluation suite

```bash
# Baseline vs. moonlight on a single paired article
python scripts/eval_baseline_vs_moonlight.py
# → docs/EVAL_RESULTS.md, README.md benchmark table

# DB stats
moonlight db-stats
```

---

## Dataset

### Source: presidency.gov.mv

The Maldives Presidency Office publishes every press release, speech, decree, and amendment in both English and Dhivehi on their official site. The URL pattern is predictable:

```
English:  https://presidency.gov.mv/Press/Article/{id}
Dhivehi:  https://presidency.gov.mv/Press/Article/{id}?lang=dv
```

Each article has a canonical ID. The EN and DV versions share the same ID, making pairing trivial once the article list is discovered.

### Article categories

| Category | Description |
|---|---|
| `press_release` | General press releases (largest category) |
| `speech` | Presidential speeches |
| `vp_speech` | Vice-Presidential speeches |
| `amendment` | Constitutional and statutory amendments |
| `decree` | Presidential decrees and executive orders |

### Corpus statistics

| Metric | Value |
|---|---|
| Total article pairs | 2,648 |
| EN tokens (approx) | ~2.1M |
| DV tokens (approx) | ~1.9M |
| Date range | 2019 – present |
| Average article length (EN) | ~420 words |
| Sentence-pair alignments | ~38,000 |

Note: "tokens" here means whitespace-split tokens for EN, and space-equivalent units for DV — Thaana does not use spaces between morphological units in the same way Latin script does. Actual subword token counts (as seen by the LLM) are higher.

### Why this corpus is valuable

Most Dhivehi MT training data comes from religious texts, news aggregators, or crowd-sourced platforms. The Presidency corpus is:

1. **Domain-consistent**: every document is formal government communication
2. **Register-consistent**: produced by a small, professional translation team
3. **Aligned at article level** by default (same ID, same structure)
4. **Dateable**: articles carry publication dates, which matters for terminology drift over time

The paired nature is the key. It is not a monolingual corpus with separate EN and DV text — it is the same content in both languages, which makes it directly usable as translation memory and as few-shot material.

### Data quality notes

- Some older articles (pre-2020) have EN versions that are looser adaptations rather than direct translations. The sentence-level alignment step uses a modified Champollion aligner that discards low-confidence pairs.
- Decree and amendment text sometimes contains Arabic-script passages that the EN version renders as transliterated text. These are flagged in the corpus but not excluded.
- Numeric alignment is checked: if an EN article contains MVR 12,000 and the DV version does not contain the Thaana numeral equivalent or the ASCII digits, the pair is flagged with `quality=numeric_mismatch`.

### Licence and attribution

The content of presidency.gov.mv is the official output of the Government of the Maldives. Scraping and storing this content for research purposes is not endorsed or authorised by the Presidency. Use responsibly, respect robots.txt, apply rate limiting, and do not redistribute the raw corpus. The code in this repository is MIT-licensed; the data is not.

---

## Dhivehi Language Notes

This section is for developers who know Python but are not familiar with Dhivehi or Thaana script. Understanding these points helps explain several implementation decisions.

### Script

Dhivehi is written in **Thaana** (Unicode block U+0780–U+07BF). Thaana is:

- **Right-to-left**: text flows RTL, which affects string handling, display in terminals, and how you measure "length" for length-ratio scoring.
- **An abugida variant**: consonants carry inherent vowels, modified by diacritic vowel markers. This means a single Unicode code point may represent what feels like multiple phonemic units.
- **Compact**: a single Thaana word often encodes what English would express as a phrase. This is the main reason EN→DV translations are shorter by character count.

Unicode range reference:

| Range | Contents |
|---|---|
| U+0780–U+07A5 | Thaana letters |
| U+07A6–U+07B0 | Thaana vowel signs and sukun |
| U+07B1 | Thaana letter NAA (used in some loanwords) |

### Morphology

Dhivehi is **agglutinative**: suffixes stack onto stems to encode tense, aspect, evidentiality, politeness, and grammatical role. The formal written register uses longer suffix chains than spoken Dhivehi, which is why translations produced by systems trained on spoken or informal data sound casual in official contexts.

### Loanwords

A significant portion of formal Dhivehi vocabulary is borrowed from **Arabic** and **Farsi**, often via the Islamic scholarly tradition. Many governance terms are Arabic-origin and are written in Thaana transliteration. The terminology glossary in Moonlight covers the most common of these. A translation that renders an Arabic-origin term using a native Dhivehi root (when the PO convention is to use the Arabic loanword) will be technically understandable but register-wrong.

### Place names

Maldivian place names are unusually sensitive to transliteration variation. Key patterns:

- The **glottal stop**: Kan'ditheemu vs Kanditheemu — these refer to the same island but the apostrophe is part of the official romanisation
- The **final -u**: island names conventionally end in -u in romanised form (Hulhumalé being an exception with the accent)
- **Atoll format**: the North/South prefix system (North Malé Atoll, South Malé Atoll) differs from the Dhivehi atoll code system (Kaafu Atoll)
- **Spelling variation over time**: older PO documents sometimes use different romanisations than current ones

The `place_names` table in the DB and the GeoNames MV + Wikipedia atoll supplement together provide a lookup that is injected into each DV→EN prompt. Without this, LLMs produce inconsistent romanisations.

### RTL handling in evaluation

When computing BLEU and chrF scores for DV output, standard tokenisers may mishandle RTL text. Moonlight uses sacrebleu's `--tokenize char` mode for DV-side scoring, which avoids word-boundary assumptions.

---

## Design Philosophy

### No fine-tuning

Moonlight deliberately does not fine-tune any model. This decision is documented formally in [ADR-0001](docs/adr/0001-corpus-as-retrieval-index.md), but the short version is:

1. **Data volume**: 2,648 article pairs is not enough to reliably improve a frontier model's Dhivehi capability through supervised fine-tuning. It might be enough to overfit to PO style in a narrow way, but that narrow overfit would degrade on atypical inputs.
2. **Vocabulary drift**: PO terminology evolves — new ministers, new policy areas, new place names. A fine-tuned model's weights cannot be cheaply updated; a retrieval index can be rebuilt overnight with a new scrape.
3. **Frontier model quality**: Large models already have substantial multilingual Thaana/Arabic/English structure encoded from pretraining. Retrieval-augmented prompting exploits that structure more efficiently than fine-tuning on 2,648 examples.
4. **Maintenance cost**: Fine-tuned model checkpoints require storage, versioning, and periodic retraining. A SQLite retrieval index requires none of that.

This is not a statement that fine-tuning is always wrong for Dhivehi — it is a statement that for this domain, corpus size, and maintenance budget, retrieval-augmented prompting is the better choice.

### Retrieval over generic prompting

The alternative to fine-tuning is not "just ask the LLM". Generic prompting of a frontier model for DV↔EN translation produces output that is semantically reasonable but register-inconsistent. The retrieval step injects domain-specific context at inference time: real examples from the same domain, the same organisation, often the same topic area.

The four-layer prompt (system + glossary + sentence TM + article few-shot) is designed so each layer compensates for a different failure mode:

- **Glossary**: prevents term-level hallucination on domain-specific vocabulary
- **Sentence TM**: anchors phrase-level patterns in PO output
- **Article few-shot**: provides structural context (how a full press release opens and closes, how dates and amounts are formatted)
- **System instruction**: sets mode (faithful vs po_style) and provides language-level instructions that are stable across requests

### Two modes, not one

`faithful` and `po_style` optimise for different things that cannot both be maximised simultaneously. `faithful` prioritises: numeric accuracy, entity preservation, semantic completeness. `po_style` prioritises: register matching, idiomatic Dhivehi expression, structural conformity to PO conventions.

Trying to do both in one prompt produces a compromise that is mediocre at both. The separation is a deliberate design choice, documented in [ADR-0003](docs/adr/0003-two-translation-modes.md).

---

## Evaluation

### Metrics

| Metric | What it measures | Notes |
|---|---|---|
| BLEU | n-gram overlap with reference | Coarse; paraphrase-sensitive; useful as a baseline comparison |
| chrF | Character n-gram F-score | Better for Thaana; not sensitive to word boundary assumptions |
| Numeric F1 | Precision/recall on numeric tokens | Dates, amounts, percentages, article numbers |
| Entity recall | Recall of proper nouns from reference | Place names, person names, organisation names |
| Composite | Weighted combination of the above | The primary summary metric |

Composite formula:

```
composite = 0.25 * bleu + 0.35 * chrF + 0.25 * numeric_f1 + 0.15 * entity_recall
```

Weights are not sacred. The `--weights` flag on `moonlight eval run` lets you override them.

### Held-out evaluation split

The default evaluation split is 10% of the corpus held out (264 article pairs), stratified by article category and year. This avoids the common failure mode of evaluating only on recent press releases (which would overfit to current terminology).

### Results table

> Actual scores are populated by running `moonlight eval run`. The table below shows the metric names and the evaluation conditions.

| Mode | Direction | BLEU | chrF | Numeric F1 | Entity Recall | Composite |
|---|---|---|---|---|---|---|
| `po_style` | EN→DV | — | — | — | — | — |
| `po_style` | DV→EN | — | — | — | — | — |
| `faithful` | EN→DV | — | — | — | — | — |
| `faithful` | DV→EN | — | — | — | — | — |
| Baseline (no retrieval) | EN→DV | — | — | — | — | — |
| Baseline (no retrieval) | DV→EN | — | — | — | — | — |

Run `python -m moonlight.eval report results/*.json --format markdown` to populate this table from your own evaluation run.

### Ablation conditions

The evaluation suite also runs three ablation conditions automatically:

| Condition | Description |
|---|---|
| `no_retrieval` | System instruction only; no glossary, TM, or few-shot |
| `bm25_only` | BM25 retrieval; embeddings disabled |
| `embed_only` | Embedding retrieval; BM25 disabled |
| `full` | Full hybrid pipeline (default) |

These ablations show the marginal contribution of each retrieval component. See [Research Findings](#research-findings) for what we expect to observe.

---

## Research Findings

### What the ablation suite reveals

The ablations are designed to answer four questions:

**1. Does retrieval help at all?**

`full` vs `no_retrieval`. The expectation is a meaningful chrF improvement, particularly for DV→EN on domain-specific vocabulary. Without retrieval, the LLM invents PO-style phrasing that sounds plausible but deviates from the actual conventions.

**2. Does hybrid retrieval beat single-method retrieval?**

`full` vs `bm25_only` vs `embed_only`. The hypothesis (supported by the design rationale in [ADR-0002](docs/adr/0002-hybrid-retrieval.md)) is that BM25 alone scores well on named entity recall (because named entities are exact-match terms) but poorly on semantic coverage for paraphrased or topic-adjacent queries. Embeddings alone score well on semantic coverage but miss exact named entities, particularly non-Latin Thaana strings that embedding models may not have seen densely in pretraining.

**3. Does mode separation matter?**

Numeric F1 comparison between `faithful` and `po_style` on the same test cases. The prediction is that `po_style` trades some numeric accuracy for better register matching — the PO style occasionally renders numbers in words, restructures sentence order, and uses idiomatic date formats that technically paraphrase the numeric content.

**4. Do place names need the reference DB?**

Entity recall comparison with and without the `place_names` injection. Maldivian atoll and island names are the highest-variance point in DV→EN translation. The expectation is a measurable entity recall improvement from the place name DB, particularly for remote atolls that appear infrequently in training data.

---

## Limitations

Being direct about what Moonlight does not do well:

- **Domain boundary**: The corpus is Presidency Office content. Translations of legal text, academic writing, or informal Dhivehi are outside the retrieval distribution. The system will still produce output but retrieval quality drops.
- **BLEU as a metric**: BLEU is a noisy metric even for well-resourced languages. For Dhivehi, where paraphrase is the norm in formal writing, a low BLEU score does not necessarily mean a bad translation. Always use chrF as the primary automated metric and human evaluation for final judgements.
- **Single reference evaluation**: The corpus provides one reference translation per article (the official PO translation). Multiple valid translations exist; scoring against one reference underestimates true translation quality.
- **Embedding model Thaana coverage**: `paraphrase-multilingual-MiniLM-L12-v2` was trained on 50+ languages but Dhivehi is not prominently represented. The semantic embeddings for Thaana text are likely to be noisier than for well-resourced languages. Hybrid retrieval partially compensates for this.
- **LLM Thaana hallucination**: Even frontier models occasionally produce Thaana characters that are visually similar to the intended output but are technically incorrect codepoints. The candidate scoring step penalises outputs containing characters outside the valid Thaana + punctuation range, but it does not catch all such errors.
- **Corpus coverage bias**: The corpus skews toward government policy and ceremony topics. Disaster response, health, and economic topics from the early COVID period (2020–2021) are heavily represented and may overfit retrieval to that era's terminology.

---

## Citation and Acknowledgements

If you use Moonlight in research, please cite it as:

```
@software{moonlight2024,
  author = {Mohamed, Sofwathullah},
  title  = {Moonlight: A Retrieval-Augmented English–Dhivehi Translation Engine},
  year   = {2024},
  url    = {https://github.com/sofwath/moonlight}
}
```

### Acknowledgements

- The corpus is drawn from the public output of the **Presidency of the Maldives** (presidency.gov.mv). This project is not affiliated with or endorsed by the Presidency.
- Place name data from **GeoNames** (CC BY 4.0) and **Wikipedia** Maldivian atoll articles.
- The `paraphrase-multilingual-MiniLM-L12-v2` sentence embedding model is from the [Sentence-Transformers](https://www.sbert.net/) project by Reimers & Gurevych (2019).
- Evaluation uses [sacrebleu](https://github.com/mjpost/sacrebleu) by Matt Post.
- The project name honours *Moonlight*, the Maldives' first English-language daily newspaper, which ceased publication in December 1978 after a short but meaningful contribution to English-language journalism in the Maldives.


<!-- EVAL_TABLE_START -->

### Benchmark: Baseline → Moonlight (no corpus) → Moonlight (full corpus)

*Test article #29734 — Namibia condolences (2024-02-05) — 2026-05-23*  
*Metric: chrF (character n-gram F-score, 0–100, higher = better)*

**Series 1 — mid-tier models**

| Model | Direction | A: Baseline | B: Moonlight nocorp | C: Moonlight corpus | A→B | B→C | A→C |
|-------|-----------|:-----------:|:-------------------:|:-------------------:|:---:|:---:|:---:|
| Claude Sonnet 4.6 | DV→EN | 62.4 | 62.7 | 64.4 | +0.3 | +1.7 | **+2.0** |
| Claude Sonnet 4.6 | EN→DV | 58.0 | 58.0 | 63.8 | +0.0 | +5.8 | **+5.8** |
| Gemini Flash 2.0 | DV→EN | 58.8 | 60.2 | 61.6 | +1.4 | +1.4 | **+2.8** |
| Gemini Flash 2.0 | EN→DV | 65.5 | 59.6 | 67.4 | −5.9 | +7.8 | **+1.9** |

**Series 2 — upgraded models**

| Model | Direction | A: Baseline | B: Moonlight nocorp | C: Moonlight corpus | A→B | B→C | A→C |
|-------|-----------|:-----------:|:-------------------:|:-------------------:|:---:|:---:|:---:|
| Claude Opus 4.7 | DV→EN | 63.4 | 60.8 | 62.6 | −2.6 | +1.8 | **−0.8** |
| Claude Opus 4.7 | EN→DV | 61.7 | 65.1 | 63.4 | +3.4 | −1.7 | **+1.7** |
| Gemini 3.5 Flash | DV→EN | 63.7 | 62.3 | 61.8 | −1.4 | −0.5 | **−1.9** |
| Gemini 3.5 Flash | EN→DV | 68.7 | 15.7 | 12.5 | −53.0 | −3.2 | **−56.2** ⚠ |
| GPT-5.5 | DV→EN | 60.7 | 59.7 | 61.6 | −1.0 | +1.9 | **+0.9** |
| GPT-5.5 | EN→DV | 31.0 | 48.1 | 55.9 | **+17.1** | **+7.8** | **+24.9** |

> **A→B** = value of moonlight's prompt design alone (no data).  
> **B→C** = value of corpus retrieval (1,000 paired EN+DV articles + glossary).  
> **A→C** = total pipeline gain over raw LLM.  
> ⚠ = prompt-format compatibility failure (phrase-context labels echoed as output).  
> See [docs/EVAL_RESULTS.md](docs/EVAL_RESULTS.md) for full translations and [docs/moonlight-rag-dhivehi-mt.md](docs/moonlight-rag-dhivehi-mt.md) for the research paper.

<!-- EVAL_TABLE_END -->
---

## Contact

Research inquiries: Sofwathullah.Mohamed@gmail.com
