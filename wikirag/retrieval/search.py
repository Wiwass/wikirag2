"""First-stage hybrid search: BM25 + dense, fused by RRF.

- Lexical: FTS5 over chunks and notes (BM25 ranking built in).
- Dense: VectorIndex search over the same content, one embedding space.
- Fusion: Reciprocal Rank Fusion — rank-based, so no score-scale
  calibration between channels (v1's min-max normalizations and
  round-robin interleaving disappear). Multi-anchor queries (R2
  sub-questions) fuse the same way: RRF across all anchor rankings.

Returns candidates as (owner_kind, owner_id, fused_rank/score) with
enough provenance for assembly to fetch text and cite sources.

Planned API
-----------
- hybrid_search(kb, queries: list[str], k: int) -> [Candidate]
"""
