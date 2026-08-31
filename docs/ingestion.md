# Ingestion

**Status: contracts fixed, methods deliberately open.** The explicit v2 goal is ingest that is *much cheaper* than v1 (which spent LLM calls proportional to corpus size × keyword index size, with expensive verbatim output) while producing an index that beats or matches v1's benchmark scores. Implementations will be researched and measured, not designed up front.

## Fixed: the stage contract

Ingestion is a sequence of idempotent stages per document, tracked in the `jobs` table. Each stage reads/writes only through the store, can be retried safely, and a batch interrupted at any point resumes exactly where it stopped.

```
extract   PDF → sections (structure-aware, no LLM)
chunk     sections → chunks with stable IDs (deterministic, local, sized to embedding model)
xrefs     chunks → native cross-references (pattern-extracted, no LLM)
index     chunks → FTS rows + embeddings                    [this alone = S0]
summarize documents → L1 gist summaries                     [S1+]
enrich    chunks → chunk_keywords tags, chunk context       [S2+; LLM or local — open]
```

Note synthesis is NOT an ingest stage anymore: L2 notes are consolidated lazily at query time and in background (D11, docs/synthesis.md). Ingest pays only for extract/chunk/xrefs/index/summarize/tagging.

Non-negotiable rules (the lessons v1 paid for):

1. **No verbatim echo.** LLM stages receive numbered chunks and return IDs + new prose. Never source text.
2. **Chunking is deterministic and local.** Re-running on the same file yields the same chunks and IDs.
3. **Chunk size fits the embedding model** with margin (target ~400–600 tokens). No unbounded concatenation.
4. **Every stage records cost** (tokens in/out, calls, wall time) via the LLM layer's meter, so method comparisons are grounded.

## Open: methods to research

Ranked roughly by expected cost/benefit; each candidate gets measured against the S0/R1 floor on the v2 benchmark before adoption.

- **Contextual chunk enrichment** (Anthropic-style contextual retrieval): prepend 2–3 generated sentences of document context to each chunk before embedding. Big reported retrieval gains; cost is one cheap call per chunk, and prompt caching of the surrounding document can push it near zero. Could also be produced by a *local* small model.
- **Local/small-model enrichment**: keyword tagging and chunk contextualization are shallow tasks; a local 4–8B model (or a very cheap API tier: flash-lite class) may match the quality of v1's setup at a fraction of cost. Hybrid split: local for tagging, API for synthesis prose.
- **Deferred synthesis (LazyGraphRAG pattern)**: don't build notes for every keyword at ingest; build/refresh a note the first time a query actually needs that concept, then cache it. Turns S2 cost from "pay for everything up front" into "pay for what gets used". Fits the slider vision naturally.
- **Embedding-shortlisted tagging**: when tagging chunks against the existing keyword index, shortlist candidate keywords by embedding similarity first; the LLM only ever sees a few dozen candidates. Removes v1's O(index) prompts and the runaway-filter failure mode entirely.
- **Visual-first indexing (ColPali class)**: embed page images directly, skipping text extraction for layout-heavy sourcebooks (tables, stat blocks). Heavier locally, zero API; worth a spike on the Ars Magica corpus where tables matter.
- **Borrowing whole pipelines**: evaluate LightRAG and RAPTOR's open implementations on a sample KB before reinventing; if one wins on cost/quality, wrap it as a stage implementation rather than porting ideas by hand.

## Evaluation criteria for any candidate method

1. Retrieval quality vs the floor (benchmark score at fixed context budgets).
2. Ingest cost: API tokens + wall-clock local compute per million corpus tokens.
3. Incrementality: cost of adding one document to a large KB (v1's Phase B failed exactly here).
4. Failure surface: what breaks when the LLM misbehaves, and does it break one document or the whole KB.
