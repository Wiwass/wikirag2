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

**Synthesis level `S` — per knowledge base, ingestion time.** How much LLM work is invested in making the KB readable and conceptually organized.

| Level | Name | What is built |
|---|---|---|
| S0 | Index only | Chunks + embeddings + lexical index. No wiki. Cheapest possible ingest. |
| S1 | Summaries | S0 + per-document/section summaries (retrievable, browsable). |
| S2 | Wiki | S1 + keyword notes with wikilinks (the v1 vault, rebuilt on chunk references). |
| S3 | Atlas | S2 + overview map note, curated hub pages, link curation passes. |

**Retrieval depth `R` — per query (user-set or auto-routed).** How much work a single question is allowed to cost.

| Level | Name | What runs |
|---|---|---|
| R0 | Lookup | Single hybrid search (BM25 + dense, RRF), top-k straight into context. |
| R1 | Ranked | R0 + cross-encoder reranking. The default. |
| R2 | Multi-hop | R1 + query decomposition into sub-questions + graph-neighbor content expansion. |
| R3 | Agentic | An agent navigates the KB with tools (search, read note, follow links, read chunks) until satisfied. |

Levels degrade gracefully: any R works on any S (R2/R3 simply have fewer structures to exploit on an S0 KB).

### Design principles

1. **Chunks are canonical.** Deterministic, local, structure-aware chunking with stable IDs. Everything (keywords, notes, retrieval results) references chunk IDs. **The LLM never echoes source text back**; verbatim always comes from storage.
2. **One transactional source of truth.** SQLite per KB (content, links, FTS index, job state). The markdown vault is a regenerable *view*, never the primary store.
3. **One embedding space.** A single modern embedding model, chunks sized to fit its context. No parallel embedding tiers.
4. **Retrieval floor = 2026 standard.** BM25 + dense fused with RRF, cross-encoder reranker. Anything fancier must beat this in the benchmark.
5. **Synthesis is decoupled and optional.** Wiki notes are *products* of ingestion and *inputs* to retrieval (indexed like any document), not a privileged hardcoded tier.
6. **Provider-agnostic LLM layer.** Minimal client interface; Gemini is the first implementation. Cost metering built in, because cheap ingest is an explicit goal.
7. **Benchmark as regression gate.** The v1 harness (blind LLM judge, ablation tiers) is ported early; every pipeline claim gets measured against the S0/R0 floor.
8. **Resumability is first-class.** Ingestion is a set of idempotent, per-unit jobs tracked in SQLite, not ad-hoc pending files.

## Planned repository layout

```
wikirag2/
├── README.md                  # this file — master overview
├── docs/
│   ├── architecture.md        # components, data flow, contracts between layers
│   ├── storage.md             # SQLite schema draft, vault-as-view, ID scheme
│   ├── ingestion.md           # stage contracts + OPEN research on cheap ingest methods
│   ├── synthesis.md           # the S slider: wiki notes, overview, link curation
│   ├── retrieval.md           # the R slider: hybrid search, rerank, graph, agentic mode
│   ├── llm.md                 # provider-agnostic client, cost metering, key pooling
│   ├── evaluation.md          # benchmark harness, metrics, regression workflow
│   └── decisions.md           # running log of design decisions (ADR-lite)
├── wikirag/                   # (future) the package
│   ├── cli.py                 # typer/argparse CLI: kb management, ingest, query
│   ├── config.py              # KB registry, S/R profiles, model config
│   ├── storage/               # sqlite store, vector index, vault exporter
│   ├── llm/                   # LLMClient protocol, gemini impl, cost meter
│   ├── extraction/            # pdf → structured text (ports v1 pdf_extractor ideas)
│   ├── ingestion/             # stage runner + pluggable stage implementations
│   ├── synthesis/             # keyword notes, overview, links (S1–S3)
│   ├── retrieval/             # lexical, dense, fusion, rerank, assembly (R0–R2)
│   ├── agent/                 # R3 agentic navigation loop
│   └── eval/                  # benchmark harness (ported from v1)
└── tests/
```

## What is deliberately left open

Ingestion **methods** are unspecified on purpose. The stage *contracts* (what goes in, what comes out, what IDs mean) are fixed in [docs/ingestion.md](docs/ingestion.md); the *implementations* are a research area with an explicit goal: **much better cost/performance than v1**, via hybrid local/API approaches, borrowing from existing projects (contextual retrieval, LazyGraphRAG-style deferral, ColPali-style visual indexing, local small-model enrichment). Candidates and evaluation criteria are listed there.

## Roadmap

1. **Plan** (this repo, now): structure, contracts, decisions.
2. **Skeleton**: package layout, storage schema, LLM client, CLI stubs.
3. **Floor**: extraction → chunking → S0 index → R0/R1 retrieval → chat. Port benchmark, measure the floor.
4. **Synthesis**: S1–S2 on chunk references, vault export.
5. **Depth**: R2 (decomposition + graph), S3 (atlas), then R3 (agentic).
6. Continuous: ingest-method research measured against the floor.
