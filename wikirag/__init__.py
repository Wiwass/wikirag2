"""WikiRAG 2 — personal knowledge engine.

Two products from one substrate:
- a retrieval index (machine-facing): canonical chunks, hybrid search;
- a wiki vault (human-facing): keyword notes, overview, wikilinks.

Both sit on sliders: synthesis level S0-S3 (per-KB, ingestion time) and
retrieval depth R0-R3 (per-query). See README.md and docs/ for the plan.

Package map:
    config      KB registry, S/R profiles, role->model mapping
    storage     SQLite source of truth, vector index, vault export
    llm         provider-agnostic LLM client, cost metering, local models
    extraction  PDF -> structured sections (no LLM)
    ingestion   stage runner + pluggable stage implementations
    synthesis   S1-S3: summaries, keyword notes, atlas/overview
    retrieval   R0-R2: hybrid search, rerank, graph, context assembly
    agent       R3: agentic navigation loop
    eval        benchmark harness (ported from v1)
    cli         user-facing commands
"""
