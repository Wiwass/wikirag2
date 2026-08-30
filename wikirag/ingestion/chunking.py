"""Deterministic, structure-aware chunker. Local, no LLM.

Contract (docs/ingestion.md):
- Input: an ExtractedDoc's sections. Output: chunks with stable IDs
  derived from (doc_id, structural path, ordinal).
- Target size ~400-600 tokens (embedder.max_seq_tokens minus margin is
  the binding constraint — v1 silently truncated embeddings by ignoring
  it). Split on section, then paragraph boundaries; small tail-merge to
  avoid fragment chunks; modest overlap only where a split severs a
  paragraph mid-argument.
- Re-running on the same file yields byte-identical chunks and IDs.

Chunk *enrichment* (contextual headers etc.) is NOT here — that's the
enrich stage, which may rewrite what gets embedded but never what the
canonical chunk text is.

Planned API
-----------
- chunk_document(doc, sections, max_tokens, token_counter) -> [Chunk]
"""
