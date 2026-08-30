"""LLM layer: provider-agnostic clients, cost metering, local models.

Stages reference *roles* (enrich, synth, answer, judge, classify); config
maps roles to provider+model. Swapping a role to a cheaper provider or a
local model is a config change, not a refactor (docs/llm.md).

Modules:
    client      LLMClient protocol + result/usage types (the contract)
    gemini      first implementation: retries, key pool, quota rotation
    meter       per-call cost recording into the KB's llm_calls table
    embeddings  local sentence-transformers models: embedder + reranker
"""
