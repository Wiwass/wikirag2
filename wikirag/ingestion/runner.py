"""Job runner: executes ingestion stages, resumably.

Responsibilities
----------------
- For each document, walk the stage sequence up to the KB's synthesis
  level; skip completed jobs (jobs table), retry failed ones with capped
  attempts, record errors per job.
- A batch interrupted anywhere (crash, quota exhaustion) resumes exactly
  where it stopped on the next run — this replaces v1's pending-keyword
  queue, phase-A cache directory, and overview_parts cache with one
  mechanism.
- Raising a KB's synthesis level enqueues only the missing stages for
  already-ingested documents.
- Quota exhaustion (AllKeysExhausted from the LLM layer) stops the batch
  cleanly rather than grinding through doomed retries (v1 lesson).
- Local CPU work (embedding) runs as ordinary jobs interleaved with
  API-bound jobs, not as ad-hoc threads sharing mutable stores.

Planned API
-----------
- run_pending(kb, docs=None, until_stage=None) -> RunReport
"""
