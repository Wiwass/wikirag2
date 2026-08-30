"""Cost metering: every LLM call recorded, every method comparison grounded.

Wraps any LLMClient and records (kb, stage, role, model, tokens, cost,
wall time, timestamp) into the KB's llm_calls table. This is core
infrastructure for the ingestion research program (docs/ingestion.md):
candidate methods are compared on measured build cost, not estimates.

Local models meter too (cost 0, wall time still recorded).

Planned API
-----------
- class MeteredClient(LLMClient): wraps a client with (store, stage, role)
- summarize(store, group_by=...) -> cost report rows for `wikirag costs`
"""
