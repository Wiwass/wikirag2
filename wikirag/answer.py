"""Answering: turn a query + retrieved context into a grounded reply.

Shared by `wikirag query` and `wikirag chat` (and by eval runners):
- system prompts: grounding rules (facts only from context, cite sources,
  say so when the context is insufficient), complex-question addendum
  listing sub-questions (v1 prompts, ported);
- history is continuity-only, never evidence; fresh retrieval per turn;
  history windowed to a cap (v1 let it grow unboundedly);
- the no-answer path: when rerank's top score is below threshold, reply
  "not in the KB" without stuffing weak context;
- overview primer prepended when the KB has one (S3).

Depth dispatch: R0-R2 -> retrieval pipeline + single generate call;
R3 -> wikirag.agent.loop.

Planned API
-----------
- answer(kb, query, depth, client, history=None) -> Answer(text, provenance, usage)
"""
