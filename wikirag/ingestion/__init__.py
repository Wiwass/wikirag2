"""Ingestion: idempotent per-document stages, tracked as jobs in the store.

Stage sequence (contracts fixed in docs/ingestion.md, implementations open):

    extract -> chunk -> index [S0] -> enrich [S1+] -> synth [S2+]

Non-negotiable rules:
1. No verbatim echo: LLM stages receive numbered chunks, return IDs + new
   prose, never copies of source text.
2. Chunking is deterministic and local; same file => same chunks and IDs.
3. Chunk size fits the embedding model (its max_seq_tokens is an input).
4. Every stage meters its cost.

Modules:
    runner    job scheduling, resume, retry policy
    chunking  deterministic structure-aware chunker
    enrich    OPEN research area: tagging + contextualization methods
"""
