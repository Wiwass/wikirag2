# Retrieval — the R slider

Per-query depth, explicit (`--depth`) or auto-routed. The default shape is **coarse-to-fine with verification**, not one-shot channel stacking: orient on the gist layer, retrieve fine, expand structurally, check coverage. Retrieval branches on what exists in the store, never on the configured S level.

Three rules baked in everywhere:

- **Retrieve small, read big.** Search runs over small precise chunks; context receives the surrounding section window (structural neighbors from L0). Search precision, reading coherence.
- **One scoring scale per stage**: RRF ranks for fusion, absolute reranker scores for inclusion and for the no-answer signal.
- **The reranker is part of the floor** (R0, not a level): it is local and costs milliseconds.

## Levels

**R0 — Lookup.** Single pass, no extra LLM calls:

- BM25 (FTS5) + dense search over chunks and notes; RRF fusion (rank-based, no score calibration).
- Cross-encoder rerank of a wide pool (e.g. top-50).
- Structural expansion of top hits (section window), assembly under budget.

**R1 — Default.** R0 plus:

- **Orient**: a cheap pass over the gist layer (L1 summaries, L3 atlas, keyword names — no LLM) identifies *active regions* (documents, sections, concepts). Regions **boost** fine candidates and decide which L2 notes enter context; they never filter the global channels (an orientation error must not blind the system).
- **Coverage gate**: if the top reranker score is below the absolute threshold, answer "not in the KB" instead of stuffing weak context.

**R2 — Deep.** R1 plus, for complex questions:

- Query decomposition into sub-questions (router); every anchor runs the R1 flow; RRF fuses across anchors.
- **Repair loop (CRAG-lite)**: after assembly, a light LLM call judges whether the evidence covers the question; if not, it emits 1–2 targeted queries for the missing aspects and one more retrieval round runs. Hard-capped, so cost stays predictable.
- **Consolidation trigger**: concepts touched without an L2 note are queued for lazy consolidation (docs/synthesis.md) — deep queries leave the KB better than they found it.
- Wikilink graph expansion (1-hop `note_links` content) is an **experimental channel** here, benchmark-gated with low expectations (decision D10).

**R3 — Agentic.** An agent navigates with tools until it can answer or concludes the KB lacks the information; hard caps on tool calls and tokens. Tools wrap retrieval primitives and store read APIs: `search`, `read_chunk`, `read_note`, `neighbors`, `read_section`. Investment priority over R2's experimental channels (D10): links become valuable when something can traverse them.

## Module map

| Module | Stage |
|---|---|
| `search.py` | BM25 + dense + RRF, multi-anchor |
| `rerank.py` | cross-encoder scores, NO_ANSWER_THRESHOLD |
| `orient.py` | active regions from the gist layer |
| `router.py` | simple/complex classification, sub-questions, depth auto-routing |
| `graph.py` | R2 experimental: 1-hop note_links content |
| `verify.py` | R2 coverage judge + repair queries |
| `assembly.py` | merged pool → context under budget, structural expansion, provenance |

## Context assembly

- Single merged candidate pool (chunks, notes, summaries), reranker scale.
- Greedy fill under budget with per-document/per-keyword diversity caps.
- Structural expansion applied to included chunks (section window), deduplicated by ID — expansion must not double-count budget.
- Provenance per entry: (doc, section, chunk IDs) so answers cite.
- ONE token estimator shared package-wide.
- Overview primer (if present) prepended with reserved budget.

## Answering

- History is continuity only, windowed; fresh retrieval per turn; grounding rules in the system prompt.
- The no-answer path is a first-class outcome, not a failure.
- Depth dispatch lives in `answer.py`: R0–R2 → pipeline + one generate call; R3 → `agent.loop`.
