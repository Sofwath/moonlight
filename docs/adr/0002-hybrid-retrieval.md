# ADR-0002: BM25 + Multilingual Sentence Embeddings for Hybrid Retrieval

**Status**: Accepted

**Date**: 2024

---

## Context

Given the decision in [ADR-0001](0001-corpus-as-retrieval-index.md) to use the corpus as a retrieval index, the next question is: how to retrieve the most relevant examples for a given input?

Two main retrieval methods are available: lexical (BM25) and semantic (dense vector embeddings). Each has complementary failure modes, which motivates combining them.

### What BM25 does well

BM25 (implemented via SQLite FTS5) is a term-frequency/inverse-document-frequency retrieval method. It scores documents by how much overlap they have with the query terms, weighted by term rarity.

For this application, BM25 excels at:

- **Named entity recall**: If the input contains the Thaana characters for "Addu City" (ފުވައްމުލައް ސިޓީ), BM25 will find documents containing those exact characters. Named entities are usually rare terms with high IDF, so BM25 ranks entity-matching documents highly.
- **Exact terminology matching**: Domain terms like specific legislation names, ministry names, or decree numbers will match exactly against corpus documents that use the same terms.
- **Speed**: FTS5 BM25 queries run in a few milliseconds on the 38,000-pair corpus.

### What BM25 misses

BM25 requires token overlap. It will not retrieve a relevant sentence pair if the input uses a paraphrase, a synonym, or a different sentence structure that happens to avoid the specific terms in the reference. Example:

- Input: "The President met with senior officials."
- Corpus: "The President convened a meeting with cabinet members."

These sentences are semantically equivalent for translation purposes, but BM25 will score them low if "met" and "convened" appear in different documents.

BM25 also has reduced effectiveness for Thaana morphological variants. Dhivehi uses agglutinative morphology — a root can appear in many suffix forms depending on case, tense, and politeness level. BM25 treats each surface form as a separate token, so a query containing the base form of a verb will not match corpus sentences containing inflected forms. (Porter stemming covers English but not Dhivehi.)

### What embeddings do well

Dense vector embeddings from a multilingual sentence model map semantically similar sentences to nearby vectors in the embedding space, regardless of surface form. This captures:

- **Paraphrase matching**: Sentences with different wording but the same meaning map to similar vectors
- **Cross-lingual matching**: A multilingual model can place similar EN and DV sentences near each other in the same space, enabling cross-lingual retrieval (find DV sentences similar to an EN query, or vice versa)
- **Topic-level similarity**: Sentences about the same subject area cluster together even without term overlap

### What embeddings miss

Dense retrieval struggles with:

- **Rare named entities**: A place name, person name, or decree number that appears infrequently in the embedding model's training data may be poorly encoded. The model has no mechanism to guarantee exact character-level recall.
- **Out-of-vocabulary terms**: Thaana strings that did not appear in the embedding model's training data will be encoded based on surrounding context, which may map them to incorrect semantic neighbourhoods.
- **Precision on exact terms**: An embedding model might score "the President" and "the Prime Minister" as nearly identical (both are government leader titles). For translation, this distinction matters.

### The embedding model: `paraphrase-multilingual-MiniLM-L12-v2`

Several embedding models were considered:

| Model | Languages | Dim | Size | Notes |
|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 50+ | 384 | ~120MB | Open, local, no API cost, reasonable quality |
| `text-embedding-3-large` (OpenAI) | Many | 3072 | API only | Higher quality, but API cost per request adds up |
| `multilingual-e5-large` | 100 | 1024 | ~560MB | Better multilingual coverage, but larger and slower |
| `LaBSE` | 109 | 768 | ~470MB | Strong cross-lingual, but slower than MiniLM |

`paraphrase-multilingual-MiniLM-L12-v2` was chosen because:
1. It runs locally with no API cost — embedding the full corpus and all queries runs offline
2. It is small enough (120MB) to keep resident in memory for the online retrieval step
3. It has documented multilingual coverage including Arabic, which shares phonological territory with Dhivehi formal vocabulary
4. 384 dimensions fit comfortably in SQLite BLOBs without special storage infrastructure

Dhivehi is not a prominently covered language in this model's training data. The embeddings for Thaana text are noisier than for well-resourced languages. Hybrid retrieval compensates: BM25 handles exact Thaana matching; embeddings handle semantic similarity when exact match is insufficient.

If Dhivehi embedding quality becomes a bottleneck (identifiable by low retrieval recall in ablation), upgrading to `multilingual-e5-large` or an API-based model is the most direct lever.

### Combining the two: Reciprocal Rank Fusion

BM25 produces one ranked list; embeddings produce another. To combine them without tuning interpolation weights (which would require a labelled retrieval evaluation set we don't have), Reciprocal Rank Fusion (RRF) is used:

```
RRF_score(doc) = Σ  1 / (k + rank_i(doc))
                 i
```

Where `k=60` is a smoothing constant (standard RRF default). A document that ranks highly in both lists scores higher than one that ranks highly in only one. A document absent from one list still contributes via its ranking in the other.

RRF was chosen over learned interpolation because:
- It requires no training data
- It is parameter-free (k=60 is the well-established default from the original RRF paper)
- It is interpretable — the contribution of each source to the final ranking is transparent

---

## Decision

Use **hybrid retrieval combining FTS5 BM25 and `paraphrase-multilingual-MiniLM-L12-v2` sentence embeddings**, merged with Reciprocal Rank Fusion.

Both methods run at query time. Embeddings are pre-computed offline for all corpus sentence pairs and loaded into memory at Translator initialisation. BM25 queries run against the FTS5 virtual tables in SQLite.

When the sentence-transformers library is not installed, or when the `--no-embed` flag is passed, the system falls back to BM25-only retrieval with a logged warning.

---

## Consequences

**Expected gains**:
- Better recall on paraphrased or topic-adjacent queries (embedding contribution)
- Better precision on named entities and exact terminology (BM25 contribution)
- Hybrid consistently outperforms either method alone in MT retrieval benchmarks (established result in the retrieval literature)

**Known costs**:
- Startup cost: embedding matrix load takes ~1–2s and ~110MB RAM
- Offline build cost: embedding 38,000 sentence pairs takes ~10 minutes on CPU first run
- Slightly more complex retrieval code than pure BM25
- If the embedding model produces poor Dhivehi representations, its contribution may be net-negative for DV-query retrieval (BM25-only fallback exists for this reason)

**Monitoring**:
- The ablation suite runs `bm25_only`, `embed_only`, and `full` conditions and reports metric deltas. If `embed_only` scores consistently below `bm25_only` on the Dhivehi corpus, revisiting the embedding model choice is warranted.
- Named entity recall in `full` vs `bm25_only` is the primary signal for whether embeddings help or hurt on entity-sensitive inputs.

---

## Alternatives Considered

| Alternative | Why not chosen |
|---|---|
| BM25 only | Misses semantic similarity; tested in ablation to confirm the gap |
| Embeddings only | Misses named entities; tested in ablation to confirm the gap |
| API embedding model (e.g., OpenAI) | Adds per-query API cost and network dependency; offline-first preferred |
| Learned interpolation (linear combination with tuned weights) | Requires labelled retrieval eval set; RRF avoids this with negligible performance penalty |
| Re-ranking with a cross-encoder | Adds latency; retrieval quality at top-10 is sufficient for few-shot purposes without re-ranking |

---

## Related ADRs

- [ADR-0001](0001-corpus-as-retrieval-index.md): Why retrieval rather than fine-tuning
- [ADR-0003](0003-two-translation-modes.md): How retrieved context is used in prompts
