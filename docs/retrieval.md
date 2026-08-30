# Retrieval — the R slider

Per-query depth setting. Explicit (`--depth`) or auto-routed by a lightweight classifier (v1's query classifier performed well and its contract carries over). Retrieval branches on **what exists in the store**, never on the KB's configured S level.

## Levels

**R0 — Lookup.** Hybrid first stage only:

- BM25 over `chunks_fts` (and notes) + dense search over embeddings.
- Fusion by **Reciprocal Rank Fusion**. RRF is rank-based, so it removes v1's fragile score-scale calibration (min-max normalizations, per-anchor round-robin justifications) in one move.
- Top-k into context, ordered by fused rank, under token budget.

**R1 — Ranked (default).** R0 with a wider candidate pool (e.g. top-50) passed through a **local cross-encoder reranker** (bge-reranker class). Candidates enter context by reranker score. This is the single highest-leverage upgrade over v1 per the 2026 literature.

**R2 — Multi-hop.** For complex/comparative questions:

- Query decomposition into sub-questions (ported classifier), each sub-question runs R1; results fuse by RRF across anchors (rank-based fusion makes v1's interleave machinery unnecessary).
- **Graph expansion with content**: top keywords' 1-hop neighbors via `note_links` contribute their *notes and top chunks* as scored candidates — not a name list. v1 built the graph and then used it as a tag cloud; v2 uses it as a retrieval channel.
- Larger context budget (config-set multiplier).

**R3 — Agentic.** An agent loop (answering model or a cheaper driver) with tools:

- `search(query, k)` → fused+reranked hits (id, snippet, score)
- `read_chunk(id)` / `read_note(keyword)` / `neighbors(keyword)`
- `read_section(doc, section)` for local context around a hit

The agent iterates until it can answer or concludes the KB lacks the information; hard caps on tool calls and tokens. Requires nothing beyond stable IDs and read APIs, which the storage design already guarantees. Highest quality ceiling, highest latency; intended for hard questions and interactive research sessions.

## Context assembly

- Single merged candidate list (chunks, notes, summaries — whatever scored), one scale (reranker score at R1+, RRF at R0).
- Greedy fill under budget with per-document and per-keyword diversity caps (v1's `MAX_CHUNKS_PER_KEYWORD` insight, generalized).
- Verbatim dedup by chunk ID is structural (a chunk exists once); v1's fingerprint dedup disappears.
- Provenance kept per entry: `(doc, section, chunk ids)` so the answer can cite.
- Token accounting via **one** shared estimator (the model tokenizer or a single calibrated heuristic), not three.

## Answering

- Chat keeps v1's clean separation: history for continuity only, fresh retrieval per turn, grounding rules in the system prompt.
- The overview note (if S3) is prepended as a primer, its cost reserved from the budget (v1 behavior, kept).
- A **no-answer path**: if top reranker scores fall below a threshold, say so rather than stuffing weak context. v1's min-max normalization made "nothing relevant" literally unrepresentable; absolute reranker scores fix that.
