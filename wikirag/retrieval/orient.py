"""Orientation (R1+): identify active regions from the gist layer. No LLM.

Before fine retrieval, a cheap pass over L1/L3 content (summary and
overview embeddings + FTS, keyword names) identifies the *active regions*
for the query: which documents, sections, and concepts are in play.

Contract (D12, confidence prior 55 — the weakest bet, designed to cap its
own downside):
- Regions BOOST fine-stage candidates (a bounded score bonus applied
  before/within reranking) and select which L2 notes deserve context.
- Regions NEVER filter the global channels: an orientation error must not
  blind the system. Ablation flag to disable entirely (evaluation.md Q3).

Planned API
-----------
- orient(kb, query) -> Regions(doc_ids, section_ids, keyword_ids, scores)
- Regions.boost(candidates) -> candidates with adjusted scores
"""
