# ADR-0003: Two Distinct Translation Modes — `faithful` and `po_style`

**Status**: Accepted

**Date**: 2024

---

## Context

Moonlight was extracted from the kahzaabu fact-checking pipeline, where translation is used as an internal step. But translation output serves two very different audiences in that pipeline — and in downstream uses more broadly.

### The two use cases

**Use case A — Automated pipeline input (fact-checking)**

The kahzaabu pipeline reads a Dhivehi article, translates it to English, and then runs claim-extraction and fact-checking on the English text. In this use case:

- The translation is **never read by a human** as a final product
- Semantic accuracy is paramount: every factual claim in the source must be present in the translation
- Numeric values (dates, amounts, statistics, article numbers) must be preserved exactly
- Register, idiom, and stylistic elegance are irrelevant — the downstream claim extractor does not care whether the sentence sounds natural
- Paraphrase is dangerous: if "MVR 12,000" becomes "twelve thousand rufiyaa" and the claim extractor looks for digit strings, the claim is invisible

**Use case B — Human-readable output (newsroom, UI)**

A Maldivian journalist or editor wants to read an English article translated to Dhivehi (or vice versa) in publication-ready form. In this use case:

- The translation is **read by a Maldivian reader** who has expectations about formal Dhivehi register
- Idiomatic expression matters: the PO writes in a specific formal style that readers of official documents recognise and expect
- Exact preservation of every numeric value is less important than overall quality — a human proofreader will catch numeric errors before publication
- Boilerplate phrases ("The President held a meeting..." opening paragraphs) should match PO conventions, not be translated word-for-word from English if there is a more natural Dhivehi equivalent

### The fundamental tension

These two use cases make contradictory demands on the translation:

| Dimension | Automated pipeline | Human-readable |
|---|---|---|
| Numeric values | Must be preserved exactly as digits | Prefer written-out words in some contexts |
| Sentence structure | Literal, preserving claim structure | Idiomatic restructuring acceptable |
| Honorifics | Not important | Critical to register correctness |
| Date format | Preserve exact digits | Match PO date convention |
| Boilerplate | Translate literally | Use PO-style standard phrases |
| Completeness | Every clause must be present | Minor condensation acceptable |

A single prompt cannot optimise for both simultaneously. If the system instruction says "preserve all numerics as digits," the model will not use the PO convention of writing some numbers in words. If it says "use idiomatic PO style," the model will sometimes paraphrase numeric expressions.

### Evidence from early testing

During development, a single "best effort" prompt was tested before the two-mode split. Observations:

- **Numeric F1 in po_style prompts was lower than in faithful prompts**: The model occasionally wrote "ބާރަ ހާސް ރުފިޔާ" (twelve thousand rufiyaa) when the input contained "MVR 12,000" — correct Dhivehi rendering, but invisible to a digit-searching claim extractor.
- **Register quality in faithful prompts was lower than in po_style prompts**: The literal-preservation instruction caused the model to translate sentence-by-sentence in a way that violated Dhivehi clause-ordering conventions, producing text that sounded awkward to a native reader.
- No weighting of the two objectives in a single prompt resolved this cleanly.

---

## Decision

Implement two explicitly separate translation modes with distinct system prompts:

**`faithful` mode**:
- System instruction emphasises: preserve all numerics as digits, preserve all proper nouns, do not paraphrase, do not restructure for idiom
- Used by: kahzaabu pipeline, any automated downstream consumer
- Evaluated primarily by: numeric F1, entity recall

**`po_style` mode**:
- System instruction emphasises: match Presidency Office formal register, use PO-specific honorifics and date conventions, prefer idiomatic Dhivehi expression over literal rendering
- Used by: human-facing output, UI display, editorial review
- Evaluated primarily by: chrF, human evaluation

The mode is a required parameter on the `Translator.translate()` call. There is no default, by design — callers must choose explicitly. This prevents accidental use of the wrong mode by making the choice visible.

---

## Consequences

**Expected gains**:
- Numeric F1 is meaningfully higher in `faithful` mode for automated pipeline use
- Register quality is meaningfully higher in `po_style` mode for human-facing output
- The system is honest about the tradeoff rather than hiding it in a mediocre single mode

**Known costs**:
- Doubles the number of prompt variants to maintain (two system instructions)
- Evaluation must be run separately for each mode, doubling evaluation compute
- New users may be confused about which mode to use — documentation and required-parameter design help here
- The boundary is not always clear: a journalist fact-checking a specific number while also wanting readable output falls between the modes

**Edge cases**:
- If a caller needs both semantic accuracy and readable output (e.g., a publication that will both publish and fact-check their own content), running `faithful` first for fact-checking and `po_style` separately for publication is the recommended pattern. Running both on the same input and comparing is a valid workflow.

---

## Alternatives Considered

| Alternative | Why rejected |
|---|---|
| Single prompt with tunable weights | Tested; found no weighting resolves the fundamental tradeoff |
| Single prompt defaulting to faithful | Would produce poor register quality for all human-facing use |
| Single prompt defaulting to po_style | Would produce poor numeric F1 for automated pipelines |
| Three or more modes | Adds complexity without clear benefit at this stage; `faithful` and `po_style` cover the two primary use cases |
| Letting callers write their own system prompts | Too much complexity pushed to callers; they would need to understand PO translation conventions |

---

## Related ADRs

- [ADR-0001](0001-corpus-as-retrieval-index.md): Retrieval provides context for both modes
- [ADR-0004](0004-evaluation-with-sacrebleu.md): Evaluation metrics reflect the different priorities of each mode
