"""Ingestion: idempotent per-document stages, tracked as jobs in the store.

Stage sequence (contracts fixed in docs/ingestion.md, enrich methods open):

    extract -> chunk -> xrefs -> index [S0] -> summarize [S1+] -> enrich [S2+]

Note synthesis is NOT an ingest stage: L2 notes are consolidated lazily
(wikirag.synthesis.consolidation, D11). Ingest pays only for the layers
above plus keyword tagging.

Non-negotiable rules:
1. No verbatim echo: LLM stages receive numbered chunks, return IDs + new
   prose, never copies of source text.
2. Chunking is deterministic and local; same file => same chunks and IDs.
3. Chunk size fits the embedding model (its max_seq_tokens is an input).
4. Every stage meters its cost.

Modules:
    runner    job scheduling, resume, retry policy
    chunking  deterministic structure-aware chunker
    xrefs     native cross-reference extraction (pattern-based, no LLM)
    enrich    OPEN research area: tagging + contextualization methods
"""
