"""Graph channel (R2): wikilink expansion that contributes content.

From the top-scoring keywords, walk 1-hop neighbors via note_links (a SQL
join, not markdown regex) and inject the neighbors' notes and top chunks
as *scored candidates* into the rerank pool. v1 built the graph and used
it as a name list at lowest priority; v2 uses it as a retrieval channel
or not at all.

Degrades silently on KBs without notes/links (S0/S1): the channel simply
yields nothing (S/R independence, docs/architecture.md).

Planned API
-----------
- expand(kb, seed_keywords, per_neighbor_limit) -> [Candidate]
"""
