# WikiRAG 2

A personal knowledge engine that turns document libraries (PDF-first) into two distinct, independently valuable products:

1. **A retrieval index** — machine-facing: canonical chunks, hybrid search, tight context assembly for LLM answering.
2. **A readable wiki vault** — human-facing: keyword notes with `[[wikilinks]]`, overview maps, browsable in Obsidian.

The defining idea of v2 is that **both products sit on sliders**. How much synthesis a knowledge base gets at ingestion, and how deep retrieval digs at query time, are tunable parameters, not hardwired pipeline choices.

This repository currently contains **the plan only**. No implementation yet. Knowledge bases from v1 will be rebuilt from scratch; there is deliberately no backwards compatibility.

## Why a rewrite (lessons from v1)

WikiRAG v1 ([WikiRAG repo](https://github.com/Wiwass/WikiRAG)) worked, and its benchmark harness produced the key insight that motivates v2:

- The expensive keyword pipeline (LLM summarize → filter → verbatim extract → synthesize) improved answers **only at small context budgets**. At realistic budgets (10k–20k tokens) it tied with plain raw-window retrieval, within noise.
- Asking the LLM to **echo verbatim text** (v1 Phase C) caused most of the system's incidental complexity: JSON escaping failures, repair heuristics, retry machinery, duplicated text requiring dedup at retrieval.
- State lived in 6+ mutually-coupled JSON files, requiring hand-rolled `sync_*` reconciliation everywhere.
- Embeddings silently truncated: unbounded chunk concatenation exceeded the 512-token limit of the embedding model.

Conclusion: the wiki layer's real product is the **readable vault** (and support for global/sensemaking questions), not raw retrieval quality. v2 makes that separation explicit and lets the user choose how much to invest in each.

## Core concepts

### Two axes, two sliders

**Synthesis level `S` — per knowledge base: consolidation eagerness.** How eagerly the KB invests LLM work in readable, conceptually organized artifacts. Above S1, synthesis is **usage-driven** (D11): notes are built lazily when queries need them, plus background consolidation of hot/central concepts.

| Level | Name | What exists |
|---|---|---|
| S0 | Index only | Chunks + embeddings + FTS + native xrefs. No wiki. Cheapest ingest, and the benchmark control. |
| S1 | Gist | S0 + per-document/section summaries (indexed, browsable, entry points for orientation). |
| S2 | Wiki (lazy) | S1 + keyword tagging at ingest; notes consolidated on demand and in background, cached with staleness tracking. |
| S3 | Atlas | S2 + proactive consolidation, overview map, hub pages, link curation. |

**Retrieval depth `R` — per query (user-set or auto-routed).** Default shape: coarse-to-fine with verification (D12). "Retrieve small, read big" and the local reranker are part of the floor.

| Level | Name | What runs |
|---|---|---|
| R0 | Lookup | Hybrid BM25+dense, RRF fusion, cross-encoder rerank, structural expansion. One pass. |
| R1 | Default | R0 + gist-layer orientation (active regions boost) + absolute no-answer threshold. |
| R2 | Deep | R1 + sub-question decomposition + capped repair loop (CRAG-lite) + lazy-consolidation triggers. |
| R3 | Agentic | An agent navigates with tools (search, read note/chunk/section, follow links) until satisfied. |

Levels degrade gracefully: any R works on any S (deeper R simply has fewer structures to exploit on an S0 KB).

### Design principles

1. **Chunks are canonical.** Deterministic, local, structure-aware chunking with stable IDs. Everything (keywords, notes, retrieval results) references chunk IDs. **The LLM never echoes source text back**; verbatim always comes from storage.
2. **One transactional source of truth.** SQLite per KB (content, links, xrefs, FTS index, usage, job state). The markdown vault is a regenerable *view*, never the primary store.
3. **One embedding space.** A single modern embedding model, chunks sized to fit its context. The embedding neighborhood is the system's primary association mechanism (D10).
4. **Retrieval floor = 2026 standard.** BM25 + dense fused with RRF, cross-encoder reranker, retrieve-small-read-big. Anything fancier must beat this in the benchmark.
5. **Consolidation follows use.** Synthesis above the gist layer is lazy + background, driven by the query log (D11). The vault grows around what you actually ask.
6. **Graphs are for navigation, not representation.** Only the corpus's native structure (adjacency, explicit xrefs) is materialized eagerly; semantic links are typed, grounded, and earn retrieval roles only through the benchmark (D10).
7. **Provider-agnostic LLM layer.** Minimal client interface; Gemini first. Cost metering built in, because cheap ingest is an explicit goal.
8. **Benchmark as regression gate.** The v1 harness (blind LLM judge, ablation tiers) is ported early; every claim is measured against the S0/R0 floor. Design bets carry explicit confidence priors (docs/decisions.md).
9. **Resumability is first-class.** Ingestion is a set of idempotent, per-unit jobs tracked in SQLite, not ad-hoc pending files.

## Planned repository layout

```
wikirag2/
├── README.md                  # this file — master overview
├── docs/
│   ├── architecture.md        # memory model, components, data flow, contracts
│   ├── storage.md             # SQLite schema draft, vault-as-view, ID scheme
│   ├── ingestion.md           # stage contracts + OPEN research on cheap ingest methods
│   ├── synthesis.md           # the S slider: memory layers, lazy consolidation, atlas
│   ├── retrieval.md           # the R slider: coarse-to-fine, rerank, verify, agentic
│   ├── llm.md                 # provider-agnostic client, cost metering, key pooling
│   ├── evaluation.md          # benchmark harness, metrics, regression workflow
│   ├── implementation.md      # wave plan + conventions for (sub)agent implementation
│   └── decisions.md           # running log of design decisions (ADR-lite, with priors)
├── wikirag/                   # the package (skeleton: docstrings only, implemented module by module)
│   ├── cli.py                 # CLI: kb management, ingest, query/chat, export, costs, eval
│   ├── config.py              # KB registry, KnowledgeBase handle, S/R enums
│   ├── answer.py              # grounded answering shared by query/chat/eval; depth dispatch
│   ├── storage/               # store.py (sqlite), vectors.py, vault.py (export)
│   ├── llm/                   # client.py (protocol), gemini.py, meter.py, embeddings.py
│   ├── extraction/            # pdf.py: PDF → structured sections (no LLM)
│   ├── ingestion/             # runner.py (jobs), chunking.py, xrefs.py, enrich.py (open)
│   ├── synthesis/             # summaries.py, notes.py, consolidation.py, atlas.py
│   ├── retrieval/             # search, rerank, orient, router, graph, verify, assembly
│   ├── agent/                 # loop.py: R3 agentic navigation
│   └── eval/                  # harness.py: benchmark grid + blind judge
└── tests/
```

## What is deliberately left open

Ingestion **methods** are unspecified on purpose. The stage *contracts* (what goes in, what comes out, what IDs mean) are fixed in [docs/ingestion.md](docs/ingestion.md); the *implementations* are a research area with an explicit goal: **much better cost/performance than v1**, via hybrid local/API approaches, borrowing from existing projects (contextual retrieval, LazyGraphRAG-style deferral, ColPali-style visual indexing, local small-model enrichment). Candidates and evaluation criteria are listed there.

## Roadmap

1. **Plan** (done): structure, contracts, decisions D1–D12.
2. **Skeleton** (done): package layout with per-module specs as docstrings.
3. **Implementation by waves** (next): see [docs/implementation.md](docs/implementation.md) — foundations → ingest floor → retrieval floor (Checkpoint A: measured S0 baseline) → depth (R1/R2) → synthesis (lazy S1–S3) + vault → agent (R3).
4. Continuous: ingest-method research measured against the floor; benchmark standing questions in docs/evaluation.md.
