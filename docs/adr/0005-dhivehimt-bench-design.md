# ADR 0005: DhivehiMT-Bench benchmark design

**Status**: Accepted  
**Date**: 2026-05-24

---

## Context

Moonlight's ablation study (ADR 0004, paper §5) showed that standard automatic metrics (chrF, BLEU) cannot distinguish high-quality DV outputs from each other when both contain correct Thaana script and reasonable vocabulary. Claude Opus and Moonlight-full both score in the 60–65 chrF range, but qualitative inspection reveals systematic differences in honorific forms, register, and named-entity handling that the metric misses entirely.

This is not a Moonlight-specific problem — it reflects the absence of any rigorous EN↔DV evaluation benchmark in the literature (documented in the literature review: FLORES-200 has no Dhivehi; no peer-reviewed Dhivehi MT papers exist as of 2025).

A publishable benchmark must address three failure modes of existing MT benchmarks when applied to low-resource morphologically rich languages:

1. **Metric saturation**: aggregate chrF cannot separate systems once both produce grammatical Thaana
2. **Register blindness**: no existing benchmark evaluates politeness-register accuracy at verb-form granularity
3. **Cultural bias in source text**: translated Wikipedia (FLORES method) introduces English-world cultural bias and named-entity copying inflation

---

## Decision

Design DhivehiMT-Bench as a three-component benchmark:

### Component 1: Main evaluation set (400 segments)
- 4 genres × 100 segments: government, news, social/informal, religious
- Source principle: native Dhivehi text, not translated Wikipedia
- FLORES+-compatible split: 200 dev + 200 devtest (CC BY 4.0)
- Both directions: EN→DV and DV→EN evaluated independently

### Component 2: Challenge set (160 contrastive pairs)
- 8 error categories targeting Dhivehi-specific phenomena
- Each pair: correct variant vs. targeted error; scored by chrF preference margin
- Novel contribution: Cat-1 (politeness register, 40 pairs) — first MT benchmark to evaluate this at verb-suffix granularity

### Component 3: Calibration set (50 segments)
- Human ESA annotation (WMT 2024 protocol): DA score + MQM error spans
- Gates LLM judge usage: Spearman ≥ 0.60 required before judge results appear in comparative claims
- Inter-annotator agreement: Krippendorff's alpha ≥ 0.60 threshold

### Evaluation stack
- **Primary**: chrF (sacrebleu, char order 6) — tokenisation-free, handles Thaana morphology
- **Secondary**: BLEU (for comparability with NLLB-200 published scores only)
- **Indicative**: COMET, xCOMET (zero-shot extrapolation; explicit caveat)
- **Human ground truth**: ESA on calibration set anchors all ranking claims
- **LLM judge**: GPT-4o + Gemini panel; Claude excluded from any panel that includes Moonlight output

### LLM judge exclusion of Claude
Moonlight is built on Claude (Anthropic models). Self-preference bias in LLM-as-judge has been documented (Zheng et al. 2023; Islam et al. 2025). Including Claude in the judge panel for Moonlight comparisons would introduce systematic bias. GPT-4o and Gemini are used as the judge panel instead.

---

## Alternatives considered

### Single-component aggregate benchmark (rejected)
A 500-segment aggregate set with chrF as the sole metric would be faster to build but suffers metric saturation at the top of the quality range — exactly where Moonlight operates. It would not provide discrimination between systems that are all "good."

### Wikipedia-based source text (rejected)
FLORES-200 used Wikipedia as source material. For Dhivehi, this introduces English-world cultural bias (most Wikipedia DV articles are direct translations of EN articles), and allows named-entity copying to inflate automatic scores. Native text from presidency.gov.mv, edition.mv, and mihaaru.com is more representative of actual DV usage.

### BLEU as primary metric (rejected)
BLEU requires tokenisation. Thaana has no established word tokeniser; whitespace tokenisation produces inconsistent results. chrF operates at character level and avoids this problem entirely.

### 4-category MQM profile (not adopted)
WMT full MQM has dozens of categories. For this domain, 3 categories (Accuracy, Fluency, Terminology) cover ≥95% of observed errors in the Moonlight evaluation. Reducing to 3 makes annotation tractable for non-professional annotators while maintaining reliability.

---

## Consequences

**Gained**:
- Challenge set discriminates where aggregate metrics saturate
- Register evaluation (Cat-1) is a novel publishable contribution
- FLORES+-compatible devtest enables direct comparison with future systems
- ESA calibration provides rigorous human ground truth for ranking claims
- LLM judge exclusion of Claude removes the most likely source of systematic bias

**Cost / risks**:
- Sentence-level alignment in the government corpus is approximate (by position index, not semantic alignment): the extracted raw segments require manual quality review before use
- Annotator recruitment (2–3 native DV speakers with institutional domain familiarity) is the hardest practical constraint
- Social media and religious source access involves scraping decisions and licensing considerations not yet resolved
- LLM judge calibration may not reach Spearman ≥ 0.60 on the first calibration set; fallback is to report LLM judge as exploratory

---

## Implementation

- `data/benchmark/` — JSONL data directory structure
- `scripts/extract_benchmark_segments.py` — government genre extraction from moonlight.db
- `data/benchmark/challenge_set/challenge_seed.jsonl` — initial challenge seed entries (mixed verification status; Cat-1/Cat-4/Cat-5 contain unverified scaffold items)
- `scripts/esa_annotate.py` — terminal ESA annotation tool
- `docs/benchmark-design-spec.md` — full benchmark design (Phase 2 deliverable)
