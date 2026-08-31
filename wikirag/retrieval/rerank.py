"""Second stage: cross-encoder reranking. Part of the R0 floor (D12):
local and cheap, there is no reason to make it a level.

Takes a wide first-stage pool (e.g. top-50), scores each candidate's text
against the query with the local reranker, returns absolute scores.
Highest-leverage upgrade over v1 per the 2026 literature.

Absolute scores matter beyond ordering: a below-threshold top score is
the "nothing relevant found" signal (docs/retrieval.md) — a state v1's
per-query normalization made unrepresentable.

Planned API
-----------
- rerank(kb, query, candidates, reranker) -> [ScoredCandidate]
- NO_ANSWER_THRESHOLD (calibrated on the benchmark, per reranker model)
"""
