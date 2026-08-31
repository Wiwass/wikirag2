"""Retrieval: the R slider, R0-R2 (R3 lives in wikirag.agent).

Default shape: coarse-to-fine with verification (D12). Branches on what
exists in the store, never on the configured S level. One scoring scale
per stage: RRF ranks for fusion, absolute reranker scores for inclusion
and the no-answer signal. Retrieve small, read big.

Modules:
    search      BM25 (FTS5) + dense + RRF fusion, multi-anchor     [R0]
    rerank      cross-encoder scores + NO_ANSWER_THRESHOLD         [R0 floor]
    orient      active regions from the gist layer (boost, no LLM) [R1]
    router      simple/complex classification, sub-questions,
                depth auto-routing                                 [R1+]
    graph       experimental: 1-hop note_links content             [R2, gated]
    verify      coverage judge + repair loop + consolidation
                triggers (CRAG-lite)                               [R2]
    assembly    merged pool -> context under budget, structural
                (section-window) expansion, provenance
"""
