# ADR-0001: Use the Paired PO Corpus as a Retrieval Index Rather Than Fine-Tuning

**Status**: Accepted

**Date**: 2024

---

## Context

### The problem

The Maldives Presidency Office (PO) publishes every press release, speech, decree, and amendment in both English and Dhivehi on presidency.gov.mv. As of this writing, this gives approximately 2,648 paired article documents — the same content, translated between the two languages by a professional team that applies a consistent house style.

This corpus is the best available source of high-quality, domain-specific EN↔DV translation data. The question is: what is the most effective way to use it?

### Option 1: Fine-tuning

The straightforward MT research approach is to use parallel data to fine-tune a translation model — either a dedicated sequence-to-sequence model (e.g., NLLB-200, mBART-50) or a frontier LLM via supervised fine-tuning (SFT).

Several factors make this unappealing for this project:

**Data volume**: 2,648 article pairs is a small dataset by MT standards. Sentence-level alignment produces approximately 38,000 sentence pairs, which sounds larger but is still modest. At this scale, SFT on a large model is more likely to overfit to surface patterns in the corpus than to genuinely improve translation quality. The risk is a model that reproduces PO boilerplate well but fails on novel phrasings.

**Vocabulary drift**: PO terminology is not static. New ministers are appointed, new policy domains emerge, new place names appear as islands are developed. A fine-tuned model's weights cannot be updated without a full retraining run. A retrieval index can be rebuilt overnight with a new corpus scrape.

**Dhivehi pretraining coverage**: Frontier models (Claude, GPT-4 class) have been exposed to substantial multilingual content that includes Thaana script, Arabic (which shares phonological structure with Dhivehi loanword vocabulary), and Farsi. The models already have a representation of Dhivehi. Fine-tuning on 2,648 pairs risks corrupting that representation in unpredictable ways.

**Maintenance cost**: Fine-tuned checkpoints require storage, versioning, and periodic retraining infrastructure. For a research project maintained by one or two people, this is a significant overhead.

### Option 2: Retrieval-augmented prompting

Instead of modifying model weights, inject relevant corpus examples at inference time. Given an input text, retrieve the most relevant sentence pairs and article pairs from the corpus, include them in the prompt as few-shot examples, and let the frontier model's existing multilingual capability handle the translation in that context.

This approach has different tradeoffs:

- It adds per-request latency for retrieval (~50–100ms for the DB queries)
- It adds per-request token cost for the few-shot examples
- It requires a retrieval index to be built and maintained
- It requires careful prompt engineering to use the retrieved examples effectively

But it avoids the vocabulary drift problem entirely: the retrieval index always contains the most recent corpus content. It avoids the overfitting risk. And it is far cheaper to maintain.

### Option 3: No retrieval, raw prompting

A third option is to simply prompt a frontier model for DV↔EN translation without any corpus retrieval — relying entirely on the model's pretraining knowledge of Dhivehi.

This was tested informally during early development. The output is semantically reasonable but consistently wrong on PO register. The model produces translations that a Maldivian reader would recognise as coming from a generic MT system, not from an official publication. Specific failure patterns:

- Honorific forms are inconsistent or missing
- Date format does not match PO convention
- Atoll names are romanised with inconsistent patterns
- Sentence endings in Dhivehi do not use the characteristic PO formal suffix chains

This option was rejected.

---

## Decision

Use the paired PO corpus as a **retrieval index** rather than fine-tuning. At inference time, retrieve the most relevant sentence pairs and article pairs from the corpus and inject them into the prompt as in-context examples.

The corpus data never goes into model weights. It lives in a SQLite database and is queried at translation time.

---

## Consequences

**Expected gains**:
- Translations match PO register far more closely than raw prompting
- Terminology is consistent with actual PO usage (retrieved from real examples)
- Adding new articles to the corpus immediately improves coverage for related topics
- No retraining required when terminology evolves

**Known costs**:
- Each translation request requires a retrieval step (adds latency)
- Each translation request sends more tokens to the LLM (few-shot examples in context)
- Quality is bounded by retrieval quality: if no relevant examples exist in the corpus, the few-shot signal is weak and the system degrades toward raw prompting
- The system works best for press-release-style content; performance on other Dhivehi text types is not guaranteed

**Risks**:
- Retrieval could return a near-duplicate article that causes the model to copy rather than translate. Deduplication and score thresholds in the retrieval pipeline mitigate this, but it is worth monitoring in translation_runs logs.

---

## Alternatives Considered

| Alternative | Why rejected |
|---|---|
| Fine-tune NLLB-200 on the corpus | Data volume too small; vocabulary drift problem; maintenance cost |
| Fine-tune a frontier LLM via SFT | Same reasons; also expensive; checkpoint management overhead |
| No retrieval (raw prompting) | Register quality unacceptable; tested informally and rejected |
| Retrieval + fine-tuning combined | Would combine the costs of both without proportional benefit at 2,648 pairs |

---

## Related ADRs

- [ADR-0002](0002-hybrid-retrieval.md): The retrieval mechanism (how BM25 and embeddings are combined)
- [ADR-0003](0003-two-translation-modes.md): How retrieval context is used differently in the two modes
