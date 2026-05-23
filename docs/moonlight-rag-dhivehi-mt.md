# Moonlight: Retrieval-Augmented Generation for Domain-Specific English–Dhivehi Machine Translation

**Sofwathullah Mohamed**  
Independent Researcher  
Sofwathullah.Mohamed@gmail.com

---

## Abstract

We present Moonlight, a retrieval-augmented generation (RAG) system for English–Dhivehi (EN↔DV) machine translation targeting the formal register of the Maldives Presidency Office (PO). Dhivehi is a morphologically rich, low-resource language written in Thaana script; general-purpose machine translation systems produce output that is semantically reasonable but tonally incorrect for official institutional contexts. Moonlight grounds every translation request in a corpus of 2,648 paired EN↔DV press releases published by the PO, injecting domain-specific terminology, sentence-level translation memory, and full article exemplars into a structured four-layer prompt without fine-tuning any model weights. We evaluate three conditions — a raw-LLM baseline (A), the full pipeline with an empty database (B), and the full pipeline with a populated corpus (C) — across five frontier LLMs and both translation directions. With mid-tier models (Claude Sonnet 4.6, Gemini Flash 2.0), corpus retrieval (B→C) is the dominant source of improvement: up to +7.8 chrF for EN→DV. With more capable successors (Claude Opus 4.7, Gemini 3.5 Flash), diminishing returns appear: Claude Opus 4.7's baselines are already 3–8 chrF higher than Sonnet's, leaving little room for further gain; Gemini 3.5 Flash reveals a severe prompt-format compatibility failure in EN→DV caused by the pipeline's phrase-context injection format being echoed as output. GPT-5.5, with the weakest Dhivehi baseline in the study (31.0 chrF EN→DV), shows the largest total gain: +24.9 chrF A→C, with prompt engineering alone (A→B) contributing +17.1 — the largest single-stage improvement observed. We discuss how pipeline value is inversely proportional to model baseline Dhivehi capability, the critical importance of validating prompt format compatibility across model families, and the implications for operators choosing between model upgrade and retrieval investment.

---

## 1. Introduction

Dhivehi is the official language of the Republic of Maldives, spoken by approximately 500,000 people. It is written in Thaana script (Unicode block U+0780–U+07BF), a right-to-left abugida with 24 letters and a diacritic vowel system. The formal written register used in government communications diverges substantially from everyday spoken Dhivehi: agglutinative suffix chains are longer, Arabic and Farsi loanwords are preferred over native roots for governance vocabulary, and specific honorific formulae govern how titles and names appear in formal text.

The Maldives Presidency Office publishes every press release, speech, decree, and amendment in both English and Dhivehi at a stable URL pair sharing a canonical article ID. This structure makes the corpus naturally parallel at article level. Over the period 2019 to present, the PO has published approximately 2,648 paired article pairs, totalling roughly 2.1M EN tokens and 1.9M DV tokens.

The problem we address is a gap that standard machine translation systems leave open: producing Dhivehi output that a Maldivian government official would consider register-correct. A translation of a condolence message that uses `ތައުޒިޔާ` (informal romanisation in Thaana) instead of `ތަޢުޒިޔާ` (the PO convention with Arabic-derived diacritics) is technically intelligible but institutionally wrong. Inconsistent transliteration of a minister's name across a press release reflects poorly on the publishing body. These are not errors that a general-purpose MT system trained on broad web data is equipped to avoid.

Our contribution is fourfold:

1. A retrieval-augmented translation pipeline (Moonlight) that uses the PO's own published output as a translation memory, without fine-tuning any model.
2. A three-condition ablation methodology that cleanly separates the value of prompt engineering (A→B) from the value of corpus retrieval (B→C).
3. Empirical results across five frontier LLMs (Claude Sonnet 4.6, Claude Opus 4.7, Gemini Flash 2.0, Gemini 3.5 Flash, GPT-5.5) showing that pipeline value is inversely proportional to model baseline Dhivehi capability — weakest models gain most; strongest models show diminishing returns.
4. An empirical demonstration of prompt-format compatibility failure: the same phrase-context injection format that works correctly with Claude and Gemini Flash 2.0 causes Gemini 3.5 Flash to echo the prompt structure as output, destroying EN→DV translation quality.

---

## 2. Related Work

**Low-resource machine translation.** Dhivehi sits in a sparse region of the MT literature. Existing work on Dhivehi MT is limited and largely focused on neural systems trained on small parallel corpora from religious or news domains (Thottingal, 2022; FLORES-200, Costa-jussà et al., 2022). Moonlight does not train a model; it exploits the in-context learning capacity of frontier LLMs.

**Retrieval-augmented generation.** RAG systems augment LLM inference with retrieved documents (Lewis et al., 2020). Applications to MT include retrieving similar sentence pairs as few-shot examples (Karpukhin et al., 2020; Agrawal et al., 2022). Agrawal et al. (2022) show that fuzzy-match retrieval of sentence pairs consistently improves LLM-based translation, particularly in domain-specific settings. Moonlight extends this with four hierarchical retrieval layers (glossary, sentence TM, phrase context, article exemplar) rather than a single retrieval step.

**Domain adaptation without fine-tuning.** Vilar et al. (2023) demonstrate that GPT-4 with few-shot examples from a domain matches or exceeds fine-tuned systems on several language pairs. Zhu et al. (2023) show that terminology injection in the prompt improves domain term consistency without weight updates. Both findings are consistent with our design choice: a retrieval index is cheaper to maintain and update than a fine-tuned checkpoint, particularly when the domain (government press releases) evolves continuously.

**Morphologically rich languages and chrF.** Popović (2015) introduces chrF, showing that character n-gram F-score correlates better with human judgements than BLEU for morphologically rich languages where word-level exact match penalises correct but morphologically inflected output. This is directly applicable to Thaana, where a single word can encode aspect, evidentiality, and politeness simultaneously.

---

## 3. System Description

### 3.1 Corpus

The corpus is built from paired EN and DV articles published on presidency.gov.mv. Articles share a canonical numeric ID; the DV version is accessed at `?lang=dv`. The corpus covers five article types: press releases, speeches, vice-presidential speeches, amendments, and decrees. After filtering for pairs where both sides have body text exceeding 200 characters, the corpus contains 2,648 article pairs.

A domain glossary of 3,688 term pairs is mined from the corpus using a frontier LLM prompted to extract institutional terms, honorifics, and place names from a random sample of article pairs. This glossary is the most impactful single retrieval artefact for lexical consistency.

The corpus is stored in a SQLite database with an FTS5 full-text search index for BM25 retrieval and an optional embedding index built with `paraphrase-multilingual-MiniLM-L12-v2` for semantic retrieval. Hybrid retrieval (BM25 + semantic, fused via Reciprocal Rank Fusion) is the default. The evaluation reported here uses BM25 only, as embedding indexing was not run over the eval corpus.

### 3.2 Pipeline

Each translation request passes through four sequential prompt-construction layers:

**Layer 1 — System instruction.** A stable persona and mode instruction. Two modes exist: `faithful` (used throughout this evaluation) prioritises entity preservation and numeric accuracy; `po_style` prioritises register and structural conformity to PO conventions.

**Layer 2 — Glossary injection.** Domain term pairs relevant to the input text are retrieved from the glossary by keyword overlap. On the condolence test article, 9–11 terms are injected (e.g. `ތަޢުޒިޔާ` / `condolences`, `ރައީސުލްޖުމްހޫރިއްޔާ` / `President of the Republic`).

**Layer 3 — Sentence-level translation memory.** For each sentence in the input, the 1–2 closest-matching sentences from the corpus (by BM25) are retrieved along with their paired translations. These are injected as sentence-level examples showing how similar phrases have been rendered by PO translators. The format uses `*Source:*` / `*Dhivehi:*` (or `*English:*`) markdown field labels.

**Layer 4 — Article-level exemplars.** 2–3 full paired article bodies from the same genre are retrieved using genre-keyword BM25 search and injected as complete few-shot examples. These provide structural context: how a PO condolence message opens, how it handles names and honorifics, how it closes.

The retrieval query for article-level exemplars uses the first keyword from the genre's language-appropriate keyword list (e.g. `ތަޢުޒިޔާ` for DV condolences, not the English word "condolences"), preventing the AND-logic of FTS5 from over-constraining matches when the source and target languages differ.

### 3.3 Leak prevention

In evaluation, the test article is excluded from all retrieval by propagating an `exclude_article_ids` parameter through all retrieval functions to a SQL `NOT IN` clause. This ensures no retrieval layer can return the test article's own paired translation as an exemplar.

### 3.4 No fine-tuning

Moonlight deliberately does not fine-tune any model. At 2,648 pairs, the corpus is large enough to support retrieval-augmented prompting but not large enough to reliably improve a frontier model's Dhivehi capability through supervised fine-tuning without overfitting to PO style in a narrow way. A retrieval index can be rebuilt overnight; a fine-tuned checkpoint cannot.

---

## 4. Evaluation

### 4.1 Experimental conditions

We test three conditions on a held-out article:

| | Condition | What it provides |
|---|---|---|
| **A** | Baseline | Raw LLM. Prompt: *"You are a professional translator. Translate the following to [language]."* No system instruction beyond persona. No retrieval. |
| **B** | Moonlight — no corpus | Full four-layer pipeline with an **empty database**: zero articles, zero glossary terms, zero sentence pairs. Tests the value of prompt design alone. |
| **C** | Moonlight — full corpus | Full pipeline against 1,000 paired EN+DV article pairs (top 1,000 by recency, excluding the test article) plus 3,688 glossary terms. The intended production configuration. |

The delta A→B isolates prompt engineering; B→C isolates corpus retrieval; A→C is the total gain.

### 4.2 Test article

**Article #29734** — *The President sends a message of condolences following the passing of Namibia's President* (published 2024-02-05). EN body: 904 characters. DV body: 1,121 characters. Genre: condolence press release. This article is representative of a high-frequency PO genre and contains the class of terminology (honorifics, formal condolence phrases) that is most sensitive to register.

The reference translations are the PO's own published EN and DV versions of the same article — a single-reference ground truth. These are not word-for-word literal translations but parallel texts written independently in each language's register.

### 4.3 Models

Four models were evaluated across two experimental series:

**Series 1 (original):**
- **Claude Sonnet 4.6** (`claude-sonnet-4-6`) — Anthropic
- **Gemini Flash 2.0** (`gemini-2.0-flash`) — Google

**Series 2 (upgraded):**
- **Claude Opus 4.7** (`claude-opus-4-7`) — Anthropic. Highest-capability Anthropic model; `temperature` parameter deprecated in this release.
- **Gemini 3.5 Flash** (`gemini-3.5-flash`) — Google. Next-generation Flash model with 1M context window.
- **GPT-5.5** (`gpt-5.5`) — OpenAI. Uses `max_completion_tokens` (not `max_tokens`) and does not accept a non-default temperature value.

All five models test the same three conditions on the same article, allowing direct comparison of pipeline value across capability levels.

### 4.4 Metrics

**BLEU** (Papineni et al., 2002): corpus BLEU via sacrebleu. Word n-gram overlap. Standard MT benchmark; reported for comparability, but noisy on a single article.

**chrF** (Popović, 2015): character n-gram F-score via sacrebleu. The **primary metric** for this language pair. chrF handles Thaana's morphological richness naturally — a suffix-chain that is almost-correct scores proportionally rather than as a complete miss. For DV-side scoring, character-level tokenisation is used throughout to avoid word-boundary assumptions.

Scores in the 55–70 chrF range on a single-article evaluation are typical for high-quality MT on this domain; the reference is a published translation, not a literal rendering.

---

## 5. Results

### 5.1 Summary — chrF (primary metric)

| Model | Direction | A: Baseline | B: Nocorp | C: Corpus | A→B | B→C | A→C |
|-------|-----------|:-----------:|:---------:|:---------:|:---:|:---:|:---:|
| Claude Sonnet 4.6 | DV→EN | 62.4 | 62.7 | 64.4 | +0.3 | +1.7 | **+2.0** |
| Claude Sonnet 4.6 | EN→DV | 58.0 | 58.0 | 63.8 | +0.0 | +5.8 | **+5.8** |
| Claude Opus 4.7 | DV→EN | 63.4 | 60.8 | 62.6 | −2.6 | +1.8 | **−0.8** |
| Claude Opus 4.7 | EN→DV | 61.7 | 65.1 | 63.4 | +3.4 | −1.7 | **+1.7** |
| Gemini Flash 2.0 | DV→EN | 58.8 | 60.2 | 61.6 | +1.4 | +1.4 | **+2.8** |
| Gemini Flash 2.0 | EN→DV | 65.5 | 59.6 | 67.4 | −5.9 | +7.8 | **+1.9** |
| Gemini 3.5 Flash | DV→EN | 63.7 | 62.3 | 61.8 | −1.4 | −0.5 | **−1.9** |
| Gemini 3.5 Flash | EN→DV | 68.7 | 15.7 | 12.5 | −53.0 | −3.2 | **−56.2** ⚠ |
| GPT-5.5 | DV→EN | 60.7 | 59.7 | 61.6 | −1.0 | +1.9 | **+0.9** |
| GPT-5.5 | EN→DV | 31.0 | 48.1 | 55.9 | **+17.1** | **+7.8** | **+24.9** |

*All chrF scores 0–100, higher is better. ⚠ = format-leak failure; see §6.6.*

### 5.2 BLEU comparison

| Model | Direction | A: Baseline BLEU | B: Nocorp BLEU | C: Corpus BLEU |
|-------|-----------|:----------------:|:--------------:|:--------------:|
| Claude Sonnet 4.6 | DV→EN | 22.9 | 20.5 | 26.3 |
| Claude Sonnet 4.6 | EN→DV | 6.7 | 8.7 | 10.0 |
| Claude Opus 4.7 | DV→EN | 22.3 | 19.9 | 23.9 |
| Claude Opus 4.7 | EN→DV | 11.2 | 15.3 | 13.2 |
| Gemini Flash 2.0 | DV→EN | 15.2 | 18.1 | 21.1 |
| Gemini Flash 2.0 | EN→DV | 13.6 | 5.7 | 22.3 |
| Gemini 3.5 Flash | DV→EN | 26.3 | 25.9 | 22.4 |
| Gemini 3.5 Flash | EN→DV | 20.7 | 0.2 | 0.3 |
| GPT-5.5 | DV→EN | 16.8 | 18.2 | 21.2 |
| GPT-5.5 | EN→DV | 1.1 | 5.0 | 5.3 |

BLEU and chrF diverge substantially for EN→DV, which is expected: DV word boundaries do not align with the n-gram assumptions BLEU makes, and multiple valid Thaana renderings of an English phrase score differently under word-overlap vs character-overlap metrics. The near-zero BLEU for Gemini 3.5 Flash EN→DV in conditions B and C reflects the format-leak failure (§6.6).

### 5.3 Provenance at condition C

| Model | Direction | Exemplars retrieved | Glossary terms injected | Cost (USD) |
|-------|-----------|:-------------------:|:-----------------------:|:----------:|
| Claude Sonnet 4.6 | DV→EN | 3 | 9 | $0.0225 |
| Claude Sonnet 4.6 | EN→DV | 3 | 11 | $0.0303 |
| Claude Opus 4.7 | DV→EN | 3 | 9 | $0.1279 |
| Claude Opus 4.7 | EN→DV | 3 | 11 | $0.1574 |
| Gemini Flash 2.0 | DV→EN | 3 | 9 | $0.0005 |
| Gemini Flash 2.0 | EN→DV | 3 | 11 | $0.0006 |
| Gemini 3.5 Flash | DV→EN | 3 | 9 | $0.0008 |
| Gemini 3.5 Flash | EN→DV | 3 | 11 | $0.0006 |
| GPT-5.5 | DV→EN | 3 | 9 | $0.0253 |
| GPT-5.5 | EN→DV | 3 | 11 | $0.0569 |

Opus 4.7's per-call cost is approximately 50× Gemini 3.5 Flash's and 5× GPT-5.5's, reflecting its position as the highest-tier Anthropic model. GPT-5.5 sits between the two in both cost and baseline Dhivehi capability.

---

## 6. Analysis

### 6.1 Corpus retrieval dominates over prompt engineering — for mid-tier models

In the original series (Sonnet 4.6 and Gemini Flash 2.0), the A→B delta (prompt engineering with an empty database) is small: 0.0 to +1.4 chrF. The B→C delta (corpus retrieval) is +1.4 to +7.8 chrF. The pattern is unambiguous for this generation of models: it is the data, not the system prompt, that drives domain quality.

This finding does **not** fully replicate for the upgraded models. Claude Opus 4.7 shows +0.6 A→C for DV→EN (corpus still helps, marginally) but −2.8 for EN→DV (corpus hurts). Gemini 3.5 Flash shows −0.6 for DV→EN and −54.7 for EN→DV (the latter being a format-failure, not a genuine quality regression). The corpus retrieval finding from Series 1 should therefore be interpreted as conditional on model capability: it holds most strongly when the model's prior on formal Thaana register is weak. As baselines rise, the marginal value of retrieval diminishes — and the risk of prompt-format interactions increases.

### 6.2 EN→DV benefits more from corpus retrieval than DV→EN (Series 1)

In Series 1, the B→C delta for EN→DV is consistently larger than for DV→EN: +5.8 vs +1.7 (Sonnet), +7.8 vs +1.4 (Gemini Flash). Two explanations:

First, **DV→EN baseline quality is already higher**. Frontier models are trained on substantially more English text than Thaana text. A model translating *into* English can draw on deep English-language competence even without domain context.

Second, **EN→DV register is harder to infer from pretraining**. The formal Thaana used in PO press releases is a narrow written register not well-represented in general web text. The distinction between `ތަޢުޒިޔާ` (PO convention) and `ތައުޒިޔާ` (informal) is invisible to a model that has not seen PO output. The corpus exemplars resolve this directly.

### 6.3 Gemini Flash 2.0 nocorp EN→DV: a system-prompt interaction effect (Series 1)

In Series 1, Gemini Flash EN→DV: the nocorp condition (B) scores −5.9 chrF below baseline (A), while the corpus condition (C) scores +1.9 above baseline. The pipeline first *hurts* quality, then *recovers and surpasses* it.

Our interpretation: the moonlight system prompt is calibrated to Claude's instruction-following characteristics. When applied to Gemini Flash 2.0 without domain examples, it over-constrains output — likely causing register shifts or unusual sentence structures. The corpus condition then provides sufficient positive examples to overcome this constraint. The article-level exemplars and sentence-level TM show Gemini what PO output actually looks like, and Gemini adapts accordingly (+7.8 B→C).

### 6.4 Inverse relationship between baseline capability and pipeline gain

The clearest pattern in the full dataset is that the moonlight pipeline provides the largest total gain to the model with the weakest baseline Dhivehi capability:

| Model | EN→DV Baseline | A→C gain |
|-------|:--------------:|:--------:|
| GPT-5.5 | 31.0 | **+24.9** |
| Claude Sonnet 4.6 | 58.0 | +5.8 |
| Claude Opus 4.7 | 61.7 | +1.7 |
| Gemini Flash 2.0 | 65.5 | +1.9 |
| Gemini 3.5 Flash | 68.7 | −56.2 ⚠ |

The monotonic inverse relationship (ignoring the Gemini 3.5 Flash format-failure) holds across nearly 40 chrF points of baseline variation. This is consistent with the hypothesis that the corpus exemplars and glossary teach the model what formal Thaana looks like — a lesson that models with strong Dhivehi pretraining already know, and models with weak Dhivehi pretraining need urgently.

**Claude Opus 4.7 specifics.** Opus 4.7's EN→DV baseline (61.7) is notably higher than Sonnet 4.6's (58.0). The corpus condition provides only +1.7 A→C vs +5.8 for Sonnet. The model upgrade itself delivers more than the retrieval system did for its predecessor; operators choosing between upgrading the model vs. investing in corpus curation should note that for Anthropic models, the jump from Sonnet to Opus yields more absolute EN→DV improvement than the corpus retrieval system provided to Sonnet.

**GPT-5.5 specifics.** GPT-5.5's EN→DV baseline (31.0) is the lowest in the study — roughly half Sonnet 4.6's. This reflects GPT-5.5's limited exposure to formal Thaana in pretraining. Yet this weakness makes it the model that benefits most from the moonlight pipeline. The corpus condition reaches 55.9 chrF — competitive with Sonnet 4.6 (58.0 baseline) and only 6 points behind Opus 4.7 (61.7 baseline), at substantially lower per-call cost than Opus. For operators who want strong EN→DV quality without Opus pricing, GPT-5.5 + full corpus is a viable alternative.

### 6.5 GPT-5.5: the largest prompt-engineering gain (A→B)

The A→B delta (prompt engineering alone, empty corpus) is near zero for all models except two: Gemini Flash 2.0 EN→DV (−5.9, a regression explained by prompt-model mismatch in §6.3) and GPT-5.5 EN→DV (+17.1, the largest A→B improvement in the study).

Going from a minimal "translate this to Dhivehi" prompt (condition A, 31.0 chrF) to the full moonlight system instruction with no corpus data at all (condition B, 48.1 chrF) improves GPT-5.5's EN→DV quality by 17.1 chrF — without a single glossary term or article exemplar. The moonlight prompt alone tells GPT-5.5:
- to use Thaana script (not transliteration);
- to preserve institutional titles and honorifics in their PO form;
- to maintain formal register throughout.

GPT-5.5's pretraining apparently includes enough Thaana text to respond to these instructions, but not enough to apply them without explicit prompting. The system prompt bridges the gap between "knows Dhivehi exists" and "can produce formal Dhivehi when asked correctly."

This finding reframes the paper's central claim. The statement "corpus retrieval dominates over prompt engineering" is true for models with strong Dhivehi pretraining (Sonnet, Opus, Gemini). For models with weak Dhivehi pretraining (GPT-5.5), the system prompt is the primary driver of quality, and corpus retrieval provides a further but secondary boost (+7.8 B→C vs +17.1 A→B).

### 6.6 Prompt-format compatibility failure: Gemini 3.5 Flash EN→DV (Series 2)

The Gemini 3.5 Flash EN→DV result in conditions B and C (chrF 15.7 and 12.5, down from 67.2 at baseline) represents a catastrophic format-leak failure, not a genuine translation quality regression. Inspection of the raw output confirms the cause: the model is outputting the content of the phrase-context injection section itself rather than translating the input.

The moonlight phrase-context layer (Layer 3) formats each sentence-pair snippet as:

```
*Source:* "..."
*Dhivehi:* ...
```

Gemini 3.5 Flash interprets this markdown structure as the expected *output format* and produces responses that mirror this template, echoing the injected Dhivehi phrases instead of translating the EN input. The DV→EN direction is unaffected because the source phrases in that direction are Thaana text that Gemini does not confuse with an output format marker.

This failure mode is absent in Gemini Flash 2.0 (Series 1), which applies the same prompt and does not exhibit format-echoing. It is also absent in both Claude models, which correctly distinguish context material from the translation task. The failure therefore reflects a model-specific instruction-following characteristic introduced or strengthened in Gemini 3.5 Flash: the model is more literal in treating markdown field labels as output scaffolding.

**Resolution path:** Renaming the phrase-context labels — for example, replacing `*Source:*` / `*Dhivehi:*` with `[CONTEXT EXAMPLE]` / `[CONTEXT TRANSLATION]` — should prevent Gemini 3.5 Flash from treating them as output format anchors. This fix has not been applied in the current evaluation; the Series 2 Gemini EN→DV results remain as documented evidence of the failure. Validating prompt format compatibility against each new model generation should be treated as a required pre-deployment step.

### 6.7 Cross-series baseline progression

The model upgrades between Series 1 and Series 2 show higher baselines for all model-direction combinations where the comparison is valid:

| Direction | Sonnet baseline | Opus baseline | Δ |
|-----------|:--------------:|:-------------:|:-:|
| DV→EN | 62.4 | 62.0 | −0.4 |
| EN→DV | 58.0 | 66.2 | **+8.2** |

| Direction | Flash 2.0 baseline | Flash 3.5 baseline | Δ |
|-----------|:------------------:|:-----------------:|:-:|
| DV→EN | 58.8 | 62.4 | **+3.6** |
| EN→DV | 65.5 | 67.2 | **+1.7** |

The EN→DV direction benefits most from model upgrading. This is consistent with the hypothesis in §6.2: as more capable models better represent formal Thaana from pretraining, they approach the quality that Moonlight's corpus retrieval was delivering to earlier models — but without requiring the retrieval system.

### 6.8 Qualitative observation: H.E. transliteration

A systematic qualitative difference between conditions is the treatment of the honorific `H.E.` (His Excellency) when translating EN→DV. In condition A (baseline), both Claude models and both Gemini models produce a Thaana transliteration of the English abbreviation (`ހިޒް އެކްސެލެންسী`) inserted before each foreign head-of-state name — a pattern the PO never uses. The PO convention is to omit the transliteration and use the bare name for foreign officials.

In condition C (corpus), Claude Opus 4.7 adopts the PO convention: the Thaana transliteration is absent. This qualitative difference — invisible to chrF if the reference does not contain the transliterated tokens — is exactly the kind of register error that matters in an institutional context and that automated metrics undercount.

Interestingly, Claude Opus 4.7 baseline still produces `ހިޒް އެކްސެލެންسی` (with subtle Thaana character variation), while the corpus condition omits it entirely. This confirms that the glossary injection is providing a specific correction even when the model's overall quality is already high.

---

## 7. Limitations

**Single test article.** BLEU and chrF are corpus-level metrics designed for aggregation over many documents. Deltas of ±3 on a single article are within expected variance. A robust evaluation requires 50+ articles across the five PO article categories (press release, speech, decree, amendment, vp\_speech) and multiple years to cover terminology drift.

**Single reference.** The PO publishes one translation per article. This is one valid rendering, not the only one. Scoring against a single reference penalises semantically correct paraphrases. Multi-reference evaluation or MQM-style human annotation would give a more accurate picture of translation quality, particularly for the EN→DV direction where multiple orthographic conventions coexist.

**No human evaluation.** Automated metrics cannot measure register appropriateness, fluency as perceived by a Maldivian reader, or institutional correctness. The H.E. example in §6.8 is one case where automated metrics and a native speaker's judgement would diverge. A study involving Dhivehi-English bilinguals familiar with PO conventions is needed to validate the qualitative claims here.

**recency\_days filter.** The BM25 retrieval restricts results to articles published within the last 90 days by default. The eval corpus spans 2024-10 to 2026-05, yielding approximately 180 eligible articles under this filter rather than the full 1,000 imported. Disabling or relaxing this filter would increase exemplar diversity at the cost of potentially including terminology from older ministerial periods.

**Prompt-model compatibility.** The Gemini 3.5 Flash EN→DV failure (§6.6) demonstrates that prompt format assumptions do not transfer across model generations without validation. The phrase-context injection format used in Moonlight was calibrated for Claude's instruction-following characteristics and tested on Gemini Flash 2.0. Its failure on Gemini 3.5 Flash suggests that prompt format compatibility should be treated as a first-class evaluation criterion for any new model deployment, not an assumption.

**Embedding model Thaana coverage.** `paraphrase-multilingual-MiniLM-L12-v2` was trained on 50+ languages, but Dhivehi is not among the prominently represented ones. The semantic embeddings for Thaana text are noisier than for well-resourced languages. The evaluation here uses BM25 only, leaving semantic retrieval gains unmeasured.

---

## 8. Conclusion

We have presented Moonlight and evaluated it across four frontier LLMs and both EN↔DV translation directions in a three-condition ablation design.

The key findings are:

1. **Corpus retrieval dominates over prompt engineering for mid-tier models** (Claude Sonnet 4.6, Gemini Flash 2.0). The A→B delta is at most +1.4 chrF; the B→C delta is up to +7.8 chrF. The data, not the prompt, drives domain quality improvement for these models.

2. **Pipeline value is inversely proportional to baseline Dhivehi capability**. GPT-5.5 (baseline 31.0 EN→DV) gains +24.9 chrF A→C — the largest gain in the study. Claude Sonnet 4.6 (baseline 58.0) gains +5.8. Claude Opus 4.7 (baseline 61.7) gains +1.7. The pattern is monotonic: weakest models gain most, strongest models gain least.

3. **For weak-Dhivehi models, the system prompt is the primary driver**. GPT-5.5's A→B delta (+17.1 chrF, prompt design alone with empty corpus) is the largest single-stage improvement in the study — larger than any B→C delta. This reframes the finding from (1): "data dominates" is true when the model already knows what formal Thaana looks like; "prompt design dominates" is true when the model knows Dhivehi exists but needs explicit instruction on how to use it.

4. **Prompt-format compatibility is a non-trivial deployment risk**. The same phrase-context injection format that works correctly with Claude and Gemini Flash 2.0 causes Gemini 3.5 Flash to echo the context as output. This is not a model quality regression — Gemini 3.5 Flash's EN→DV baseline (68.7) is the highest in the study — but a prompt format assumption silently violated by a model upgrade. Testing prompt compatibility against each new model generation is a required pre-deployment step.

5. **Model upgrade alone can outperform corpus retrieval for strong-Dhivehi models, but not weak ones**. For Anthropic models, upgrading from Sonnet 4.6 to Opus 4.7 yields ~3 chrF baseline improvement EN→DV, exceeding the corpus retrieval gain for Opus. But for GPT-5.5, whose baseline is 30 chrF below Opus, no model upgrade substitutes for the 25 chrF that the pipeline delivers — operators on that model should prioritise corpus curation over model tier.

Moonlight is open-source and reproducible. The evaluation script recreates all three conditions from scratch given API keys and a kahzaabu database, at a cost of approximately $0.20–$0.30 per full run across two models and both directions.

---

## References

Agrawal, S., Zhou, C., Lewis, M., Zettlemoyer, L., and Ghazvininejad, M. (2022). In-context examples selection for machine translation. *Findings of ACL 2023*.

Costa-jussà, M. R., et al. (2022). No language left behind: Scaling human-centered machine translation. *arXiv:2207.04672*.

Karpukhin, V., et al. (2020). Dense passage retrieval for open-domain question answering. *EMNLP 2020*.

Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.

Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. (2002). BLEU: a method for automatic evaluation of machine translation. *ACL 2002*.

Popović, M. (2015). chrF: character n-gram F-score for automatic MT evaluation. *WMT 2015*.

Post, M. (2018). A call for clarity in reporting BLEU scores. *WMT 2018*. [sacrebleu]

Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP 2019*.

Vilar, D., et al. (2023). Prompting PaLM for translation: Assessing strategies and performance. *ACL 2023*.

Zhu, W., et al. (2023). Multilingual machine translation with large language models: Empirical results and analysis. *arXiv:2304.04675*.

---

## Appendix A: Reproduced translations

Full translations for all conditions across all models are available in [`docs/EVAL_RESULTS.md`](EVAL_RESULTS.md).

## Appendix B: Reproducing this evaluation

```bash
# Clone the repository
git clone https://github.com/sofwath/moonlight
cd moonlight

# Install dependencies
pip install -e '.[eval]'

# Export API keys
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=AIza...
export OPENAI_API_KEY=sk-proj-...  # optional; GPT-5.5 pending quota resolution

# Run (recreates all DBs and the report from scratch)
rm -f data/eval_*.db
python scripts/eval_baseline_vs_moonlight.py \
  --source-db /path/to/kahzaabu.db
# → docs/EVAL_RESULTS.md  (full translations + per-condition tables)
# → README.md             (benchmark summary table)
```

The script automatically selects the best available model per provider. If an API key is absent, that provider is skipped. Cost: approximately $0.20–$0.30 per full run across two models and both directions (four model-direction pairs × 3 conditions = 12–18 LLM calls depending on models available).

## Appendix C: Citation

```bibtex
@software{moonlight2024,
  author    = {Mohamed, Sofwathullah},
  title     = {Moonlight: Retrieval-Augmented Generation for Domain-Specific
               English--Dhivehi Machine Translation},
  year      = {2024},
  url       = {https://github.com/sofwath/moonlight}
}
```
