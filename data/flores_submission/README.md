# DhivehiMT-Bench FLORES+ Contribution

Language: Dhivehi (div_Thaa) / English (eng_Latn)
Domain: Government/Institutional (Maldives Presidency Office press releases and speeches)
Source: https://presidency.gov.mv (native Dhivehi text with professional EN translation)
License: CC BY 4.0
Submitted to: OLDI (Open Language Data Initiative)

## File contents

| File | Description |
|------|-------------|
| div_Thaa.devtest | Dhivehi sentences in Thaana script |
| eng_Latn.devtest | Corresponding English sentences |
| flores_metadata.json | Segment provenance, quality flags, publication dates |

## Quality

Segments are sourced from paired EN-DV government press releases and speeches.
Alignment is at sentence level by position index (approximate); 94%+ pass the
DhivehiMT-Bench alignment quality gate (length ratio, shared numbers/years,
Thaana presence checks).

## Notes

- This is a 200-segment devtest subset covering the government/institutional domain.
  The full DhivehiMT-Bench (400 segments, 4 genres) is under development.
- Segments span 2020–2026 publications.
- The Arabic comma (U+060C) appears in Dhivehi text as standard punctuation —
  this is correct and not a script contamination artefact.
- Religious domain text (not included here) may contain Arabic script for
  Quranic passages; that domain is reported separately in benchmark results.
