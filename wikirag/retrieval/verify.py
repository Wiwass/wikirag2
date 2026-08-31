"""Verification (R2): coverage judge + repair queries (CRAG-lite).

After assembly, a light LLM call (role: classify) judges whether the
assembled evidence covers the question; if not, it emits 1-2 targeted
queries for the missing aspects, and ONE more retrieval round runs.
Hard-capped (default: a single repair iteration) so cost and latency stay
predictable. Confidence prior 65 (D12); benchmark question Q4.

Also owns the consolidation trigger: concepts touched by the query that
lack a fresh L2 note are pushed to consolidation_queue (D11) — deep
queries leave the KB better than they found it.

Planned API
-----------
- check_coverage(client, query, context) -> Coverage(ok, missing_aspects,
  repair_queries)
- trigger_consolidation(kb, hit_keywords) -> queued count
"""
