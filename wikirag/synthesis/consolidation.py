"""Consolidation policy: what gets synthesized, when (D11).

The S slider's engine. Sources of demand:
- lazy: consolidation_queue rows pushed by retrieval (R2 verify stage)
  or by an explicit vault request;
- background: hot concepts (query_log frequency) and central ones
  (incoming note_links, chunk_keywords counts), consolidated proactively
  at S2 (hot set only) / S3 (everything central);
- staleness: notes whose source set changed since last synthesis.

Default for first-touch latency: DEFER — the triggering query proceeds
with chunks, the note is built in background and is ready next time.
Config flag for inline synthesis if the user prefers.

Runs under the single-writer lock as ordinary jobs; every synthesis call
is metered (llm_calls) so consolidation cost is always visible.

Planned API
-----------
- run_consolidation(kb, client, budget: ConsolidationBudget) -> Report
- demand(kb) -> ranked list of (keyword_id, reason) awaiting synthesis
"""
