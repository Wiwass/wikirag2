"""PDF extraction: structure first, then clean text per section.

Ported concepts from v1 (they worked):
- PyMuPDF TOC as the primary structure source; font-size heuristics with
  per-page title aggregation as fallback.
- Boilerplate removal by repeated-line detection across pages (not fixed
  geometric crops — v1 learned this on sparse slide decks).

New in v2:
- Output is the `sections` structure consumed by the store: ordered,
  hierarchical (level from TOC depth), with page ranges — because chunk
  IDs derive from the structural path (docs/storage.md).
- Extraction quality is inspectable standalone (debug CLI like v1's
  pdf_tester) since everything downstream depends on it.

Open (docs/ingestion.md): ColPali-class visual indexing for layout-heavy
sourcebooks would bypass this module entirely for the index; extraction
would still run for the vault/synthesis side.

Planned API
-----------
- extract(pdf_path) -> ExtractedDoc(title, sections=[Section(...)])
"""
