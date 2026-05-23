# ADR-0004: Evaluation with sacrebleu (BLEU + chrF) + Numeric F1 + Composite Score

**Status**: Accepted

**Date**: 2024

---

## Context

Evaluating translation quality automatically is genuinely hard. For a low-resource language like Dhivehi, it is harder than usual. This ADR documents which metrics were chosen, why, and what their known weaknesses are. It also documents what was explicitly rejected and why.

### The problem with BLEU for Dhivehi

BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between a candidate translation and one or more reference translations. It is the dominant metric in MT research for historical reasons — it was one of the first automatic metrics to correlate reasonably with human judgements, and the community has decades of comparative results using it.

BLEU has several well-known problems that are amplified for Dhivehi:

**Paraphrase insensitivity**: BLEU gives zero credit for a correct paraphrase that uses different words. In Dhivehi, the same meaning can be expressed with meaningfully different morphological forms depending on register and style. A `po_style` translation that uses the correct PO idiom but differs from the reference in surface form may score poorly on BLEU even if a Maldivian reader would rate it highly.

**Single-reference evaluation**: Moonlight evaluates against the official PO translation as the single reference. Multiple valid translations of the same text exist; BLEU does not account for this. Scores are systematically lower than they would be with multiple references, and comparisons across papers using different reference sets are unreliable.

**Word boundary dependence**: BLEU tokenises on whitespace and punctuation. Dhivehi, written in Thaana, does not use spaces between morphological units in the same way as Latin script — spaces separate words, but Thaana words encode what English renders as multi-word phrases. This affects the n-gram counting in non-trivial ways.

Despite these problems, BLEU is retained because:
1. It is the standard, and cross-comparison with other reported results requires it
2. It is computed by sacrebleu, which ensures reproducibility through standardised tokenisation
3. It contributes to the composite score with relatively low weight

### Why chrF is better suited for Thaana

chrF (character n-gram F-score) was designed for morphologically rich languages. It counts character n-gram overlap between candidate and reference, not word n-gram overlap. This has two advantages for Dhivehi:

1. **No word boundary assumption**: character n-grams span across what would be word boundaries in the Latin-script view, which is more natural for Thaana
2. **Partial credit for morphological variants**: if the candidate uses a different suffix form of the same Dhivehi root as the reference, chrF gives partial credit for the shared character n-grams, where BLEU gives none

chrF is computed using sacrebleu with `char_order=6, word_order=0`. Word-order component is disabled because Thaana word segmentation is unreliable.

### Why a numeric F1 component is necessary

Neither BLEU nor chrF specifically tracks whether numeric values survive translation. For the kahzaabu fact-checking use case, this matters enormously: a translation that changes "MVR 2.4 billion" to "approximately two billion rufiyaa" has destroyed a claim that might be factually significant.

Numeric F1 measures precision and recall on numeric tokens (digit strings and Thaana numeral equivalents) extracted from the input and output. It is a targeted, interpretable metric for the specific failure mode of numeric distortion.

Numeric F1 is most important for `faithful` mode. In `po_style` mode, some numeric variation is acceptable (PO style sometimes writes small numbers in words), so the composite weight for numeric F1 is slightly lower in `po_style` evaluation.

### Why a composite score

Individual metrics are noisy and emphasise different things. A single number that summarises overall quality is useful for:
- Comparing ablation conditions (is hybrid retrieval better than BM25-only overall?)
- Tracking progress across versions
- Summarising results in a table

The composite score is:
```
composite = 0.25 * bleu + 0.35 * chrF + 0.25 * numeric_f1 + 0.15 * entity_recall
```

Weights reflect the priority ordering: chrF is the most informative automatic signal for Dhivehi, so it carries the most weight. BLEU and numeric F1 are roughly equal in importance. Entity recall is included but carries less weight because the entity extractor is imperfect.

These weights are not the result of optimisation on a development set — they are informed judgements. The `--weights` flag on `moonlight eval run` allows them to be overridden.

### sacrebleu for reproducibility

sacrebleu enforces standardised tokenisation and normalisation, which means BLEU and chrF scores computed by Moonlight are directly comparable to scores reported by other papers that also use sacrebleu. Without standardisation, BLEU scores are not comparable across implementations — different tokenisation choices can move scores by several points.

### What was rejected: learned metrics (COMET, BERTScore)

Learned evaluation metrics like COMET and BERTScore use a neural model to score translation quality. They generally correlate better with human judgements than BLEU on well-resourced languages.

They were rejected for Moonlight for two reasons:

1. **Dhivehi coverage**: COMET and BERTScore models are typically trained on quality estimation data from well-resourced language pairs. Their quality signal for Dhivehi is uncertain — there is no evidence they have been validated for DV↔EN translation.

2. **Oracle contamination risk**: COMET requires a reference translation and uses it along with both source and hypothesis to score quality. If the COMET model has been trained on data that includes PO translations, evaluation results would be inflated. sacrebleu's n-gram methods have no such contamination risk.

COMET or a Dhivehi-specific QE model would be a valuable addition if Dhivehi quality estimation models become available.

### What was rejected: human evaluation as primary

Human evaluation by Maldivian native speakers is the gold standard. It is not used as the primary metric because:
- It is expensive and slow
- It does not scale to the ablation conditions (4 conditions × 2 modes × 2 directions × 264 test pairs = ~4,200 translations to evaluate)
- It cannot be run automatically in a CI pipeline

Human evaluation should be run on a sample of results before any public report of Moonlight's translation quality. Automated metrics are a development signal, not a publication-quality quality claim.

---

## Decision

Use **sacrebleu BLEU + sacrebleu chrF + numeric F1 + entity recall**, combined into a **composite score** as the primary automated evaluation suite.

BLEU is computed with sacrebleu's character tokenisation for Dhivehi output and 13a tokenisation for English output.

chrF is computed with `char_order=6, word_order=0`.

Numeric F1 is computed using a custom extractor that handles both ASCII digits and Thaana numeral characters.

Composite weights:
- `composite = 0.25 * bleu + 0.35 * chrF + 0.25 * numeric_f1 + 0.15 * entity_recall`

---

## Consequences

**Expected gains**:
- chrF provides a more meaningful primary signal for Dhivehi than BLEU alone
- Numeric F1 directly tracks the most consequential error type for the automated pipeline use case
- sacrebleu standardisation ensures scores are reproducible and comparable
- Composite score provides a single summary number for ablation comparison

**Known weaknesses**:
- BLEU is retained despite its limitations, because cross-comparison with other work requires it
- All automatic metrics are noisy for Dhivehi given the paraphrase sensitivity issue
- Single-reference evaluation systematically underestimates quality
- Numeric F1 does not catch semantic numeric errors (e.g., changing a year from 2023 to 2024 — both contain four digits, both match the pattern)
- No learned quality metric is used (COMET, BERTScore), which would improve human-correlation at the cost of coverage uncertainty

**Guidance**:
- Use chrF as the primary headline metric in any result reports
- Use numeric F1 as the headline metric when evaluating `faithful` mode specifically
- Treat BLEU scores as supplementary context, not primary quality claims
- Run human evaluation before drawing strong conclusions from automated metrics

---

## Related ADRs

- [ADR-0003](0003-two-translation-modes.md): Mode separation affects which metrics matter most
