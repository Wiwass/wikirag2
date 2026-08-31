# Decision log

Running record of design decisions. Newest first. Format: date, decision, why, alternatives set aside. Confidence annotations (0-100) are honest priors to be checked against the benchmark, not promises.

## 2026-08-31 — Implementation, Wave 0

**D13. Vector index: npz + numpy brute force; sqlite-vec deferred.** Micro-benchmark (the open question in docs/storage.md), random vectors at dim 1024, float32, single `index.npz` per KB, 12-core WSL2 box, median of 10 top-20 searches after warm-up:

| vectors | search median | search p90 | filtered (`owner_kind`) | save | load | file / RAM |
|---|---|---|---|---|---|---|
| 10k | 2.3 ms | 7.1 ms | 1.5 ms | 0.5 s | 0.06 s | 41 MB |
| 50k | 2.5 ms | 5.3 ms | 2.3 ms | 0.9 s | 0.2 s | 207 MB |
| 200k | 10.8 ms | 17.0 ms | 9.5 ms | 3-6 s | 1.5-5 s | 830 MB |

Brute force is a single BLAS matrix-vector product: latency is ~1-2 orders of magnitude below the reranker and the answering LLM at every scale a personal KB reaches (v1: tens of thousands of vectors), so an ANN or sqlite-vec index would buy nothing measurable while adding a dependency outside the closed set. The binding constraint is *memory*, not latency: the matrix is resident (~4 KB/vector at dim 1024), which is comfortable at 50k and noticeable at 200k. Revisit `sqlite-vec` (or memory-mapping the matrix) when a KB passes ~200k vectors or when index load time enters query latency. Reproduce with `WIKIRAG_BENCH=1 pytest tests/test_vectors.py -k benchmark -s`.

Persistence: mutations are in-memory, `save()` writes temp file + fsync + `os.replace`, so a crash leaves either the previous complete index or the new one — never a torn file. Unsaved mutations are lost by design; the store is the source of truth and `stale_owners()` makes re-embedding them incremental. An index opened with a different model or dim discards the stored vectors (different embedding space, derived data) and reports every owner as stale.

## 2026-08-31 — Memory model and retrieval redesign

Motivated by the graphs-vs-agents discussion (Cursor-style "no graphs for coding" positions, v1's own benchmark, GraphRAG-Bench): explicit LLM-built graphs compete with two free association mechanisms (parametric knowledge, embedding neighborhoods) and only pay off as *navigable structure*, not as flattened context.

**D12. Default retrieval shape: coarse-to-fine with verification.** Reranker moves into the R0 floor (local, milliseconds). R1 adds gist-layer orientation (boost, never filter — confidence 55) and the absolute no-answer threshold (75). R2 adds decomposition, a hard-capped CRAG-lite repair loop (65), and consolidation triggers. "Retrieve small, read big": search over small chunks, context gets section windows (85). Rejected: one-shot channel stacking as the default.

**D11. Usage-driven consolidation.** The S slider means *consolidation eagerness*, not upfront work. L2 notes are built lazily (first query that needs them, or background jobs on hot/central concepts, driven by query_log) and cached with staleness tracking. Ingest cost collapses; the vault grows around actual use. Confidence: 85 on cost, 45 on answer-quality gain over S0 — the 45 is v1's honest lesson, and the design holds either way (the vault is a product in itself). Default for first-touch latency: defer to background; the current query proceeds with chunks.

**D10. Graphs demoted to navigation and human product.** The only eagerly-built graph is the corpus's *native* one: section adjacency + explicit textual cross-references (xrefs, pattern-extracted, no LLM). Semantic wikilinks exist only typed and grounded (note_links.type, grounding_chunk_id), created during synthesis from shortlisted candidates. As a retrieval channel, graph expansion is experimental at R2, benchmark-gated, expected weak; investment priority goes to R3 agentic navigation, where links are traversed instead of flattened. Confidence in the demotion itself: 85 (v1 benchmark + literature agree).

## 2026-08-30 — Plan established

**D9. Plan-only repo first.** The repo starts as documentation; implementation follows the roadmap in README. Keeps the design reviewable before code exists.

**D8. No backwards compatibility with v1.** All KBs will be re-ingested from scratch. Frees the schema and pipeline from v1's data layout entirely. (v1 stays untouched as reference and as the source of the benchmark harness.)

**D7. Ingestion methods deliberately left open.** Contracts fixed (stable chunk IDs, no verbatim echo, per-stage cost metering); implementations are a measured research program targeting much better cost/performance than v1, possibly hybrid local/API. See ingestion.md.

**D6. Two independent sliders.** Synthesis level S (per-KB, ingestion time) and retrieval depth R (per-query). Chosen over single presets or one continuous knob: the two costs land at different times (build vs query) and serve different products (vault vs answers).

**D5. Provider-agnostic LLM layer.** Minimal client protocol with role→model mapping in config; Gemini first implementation. Chosen over Gemini-only to enable per-stage provider mixing and local models for cheap ingest.

**D4. SQLite as per-KB source of truth; vault is an export.** Replaces v1's coupled JSON files and sync machinery with transactions, FKs, FTS5. Markdown vault becomes a regenerable view.

**D3. Chunks are canonical; LLMs return references, never echo source text.** Kills v1's largest complexity source (JSON escaping/repair/retry around verbatim extraction) and its retrieval-time dedup burden.

**D2. Retrieval floor: BM25 + dense + RRF + cross-encoder rerank.** The 2026 production standard; every fancier mechanism must beat it in the benchmark to stay.

**D1. Docs in English.** Repo may outlive its private phase; working discussions remain in Italian.

## Deferred / open

- Embedding model: long-context, multilingual candidates (bge-m3, gte-modernbert, EmbeddingGemma).
- Two-way vault sync (user edits flowing back into the store): out of scope for now.
- Auto-routing R by classifier vs manual depth flag as default UX.
