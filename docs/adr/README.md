# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the Moonlight translation engine. ADRs follow the [Nygard format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions): title, status, context, decision, consequences.

Each ADR documents a non-obvious design choice: what was decided, why, what alternatives were considered, and what we expect to gain or lose. They are written to be useful to someone joining the project six months later, not as a justification exercise.

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-corpus-as-retrieval-index.md) | Use the paired PO corpus as a retrieval index rather than fine-tuning | Accepted |
| [0002](0002-hybrid-retrieval.md) | BM25 + multilingual sentence embeddings for hybrid retrieval | Accepted |
| [0003](0003-two-translation-modes.md) | Two distinct translation modes: `faithful` and `po_style` | Accepted |
| [0004](0004-evaluation-with-sacrebleu.md) | Evaluation with sacrebleu (BLEU + chrF) + numeric F1 + composite | Accepted |

## What qualifies as an ADR here

An ADR is warranted when:
- A design choice has non-obvious tradeoffs
- A reasonable alternative exists and was actively considered
- The decision is likely to be questioned or revisited as the project evolves

Routine implementation choices (which library, which field name, how to structure a loop) do not need ADRs.

## Adding new ADRs

Copy an existing ADR as a template. Increment the number. Set status to `Proposed` until the decision is settled. Add it to the table above.
