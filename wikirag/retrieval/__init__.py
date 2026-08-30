"""Retrieval: the R slider, R0-R2 (R3 lives in wikirag.agent).

Branches on what exists in the store, never on the KB's configured S
level. One scoring scale per stage: RRF ranks at R0, absolute reranker
scores at R1+ (they also drive the no-answer threshold).

Modules:
    search      lexical (FTS5/BM25) + dense + RRF fusion         [R0]
    rerank      cross-encoder second stage                        [R1]
    router      query classification: depth + sub-questions      [R2 gate]
    graph       1-hop note_links expansion with content          [R2]
    assembly    merged candidate list -> context under budget
"""
