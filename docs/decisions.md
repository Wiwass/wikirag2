# Decision log

Running record of design decisions. Newest first. Format: date, decision, why, alternatives set aside.

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

- Vector index: npz-in-RAM vs sqlite-vec (micro-benchmark at skeleton time).
- Embedding model: long-context, multilingual candidates (bge-m3, gte-modernbert, EmbeddingGemma).
- Two-way vault sync (user edits flowing back into the store): out of scope for now.
- Auto-routing R by classifier vs manual depth flag as default UX.
