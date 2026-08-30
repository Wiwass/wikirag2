"""Enrich stage — OPEN RESEARCH AREA (docs/ingestion.md).

What is fixed is only the contract:
- Input: a document's chunks (numbered) + the KB's existing keyword index
  (embedding-shortlisted, never the full index in a prompt — v1's O(index)
  cost and runaway-filter failure mode must not return).
- Output: chunk_keywords tags, optional per-chunk context strings for
  embedding, new keyword proposals. All by reference; no copied text.

Candidate methods to implement and benchmark (each behind this interface):
- contextual chunk enrichment (Anthropic-style), possibly via prompt
  caching or a local small model;
- local 4-8B model tagging with API spot-checks;
- deferred/lazy variants that tag coarsely at ingest and refine on demand.

Planned API
-----------
- class EnrichMethod(Protocol): enrich(kb, doc_id, chunks) -> EnrichResult
- registry of implementations selectable per-KB in config
"""
