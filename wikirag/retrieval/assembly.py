"""Context assembly: one merged candidate list -> context under budget.

- Single pool (chunks, notes, summaries — whatever scored), one scale.
- Retrieve small, read big (D12): included chunks are expanded to their
  section window (structural neighbors from the store), deduplicated by
  chunk ID so expansion never double-counts budget.
- Greedy fill under token budget with per-document and per-keyword
  diversity caps (v1's MAX_CHUNKS_PER_KEYWORD insight, generalized).
- Dedup is structural: a chunk exists once, referenced by ID
  (v1's text-fingerprint dedup has nothing left to do).
- Every included entry keeps provenance (doc, section, chunk IDs) so the
  answer can cite.
- ONE token estimator shared package-wide (model tokenizer or a single
  calibrated heuristic) — v1 had three that disagreed.
- Overview primer (if S3) is prepended outside the budget, its cost
  reserved like the answer allowance.

Planned API
-----------
- assemble(kb, scored_candidates, budget_tokens) -> Context(text, provenance)
"""
