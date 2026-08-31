# Implementation plan — waves, conventions, agent workflow

How to take the skeleton to a working system using parallel subagents. **Each module's spec is its docstring** plus the relevant docs/ file; this document adds ordering, conventions, and acceptance criteria.

## Why waves, not one big fan-out

Modules within a wave are independent given the contracts frozen by earlier waves, so they can be implemented by parallel agents. Modules across waves are not: parallelizing everything at once produces interface drift and integration hell. The orchestrator (main session) integrates and tests between waves; humans gate the checkpoints.

## Waves

**Wave 0 — Foundations** (contracts everything depends on; low parallelism on purpose)

| Module | Notes |
|---|---|
| `config.py` + `storage/store.py` | ONE agent: KnowledgeBase handle and schema are one design. Includes migrations, all tables from docs/storage.md, read/write APIs, tests on a temp DB. |
| `storage/vectors.py` | Interface + npz backend first; sqlite-vec behind the same interface if the micro-benchmark justifies it. |
| `llm/meter.py` + finalize `llm/client.py` | MeteredClient wrapper; fake client for tests. |

**Wave 1 — Ingest floor** (parallel: each module against Wave 0 contracts)

| Module | Notes |
|---|---|
| `extraction/pdf.py` | Port v1 ideas (TOC, font heuristics, boilerplate-by-repetition); output = sections for the store. Standalone debug CLI. |
| `ingestion/chunking.py` | Deterministic; property tests (same input → same IDs; size bounds). |
| `ingestion/xrefs.py` | Pattern set + resolution; tests on fixture text. |
| `llm/embeddings.py` | Embedder + Reranker wrappers; model choice per docs/decisions.md open questions (pick, record in decisions.md). |
| `llm/gemini.py` | Port v1 hardening (retries, key pool, carousel); no domain logic. |
| `ingestion/runner.py` | Job runner over fake stages in tests; wires real stages at integration. |

**Wave 2 — Retrieval floor + CLI** (parallel)

| Module | Notes |
|---|---|
| `retrieval/search.py` | FTS5 + dense + RRF, multi-anchor. |
| `retrieval/rerank.py` | Cross-encoder + threshold plumbing (calibration later). |
| `retrieval/assembly.py` | Pool → context; section-window expansion; ONE token estimator. |
| `answer.py` (R0 path) + `cli.py` | kb create/list, ingest, query --depth R0, costs. |

**→ Checkpoint A (human gate): the S0 floor exists and is measured.**
Ingest a small real corpus sample, run `wikirag query` end-to-end, port enough of `eval/harness.py` to score retrieval (recall@k on a small annotated set) and answers on a subset of the v1 gold dataset. Record the baseline. Nothing speculative gets built before this number exists.

**Wave 3 — Depth** (parallel)

| Module | Notes |
|---|---|
| `retrieval/orient.py` | With ablation flag (evaluation Q3). |
| `retrieval/router.py` | Port v1 classifier; depth auto-routing. |
| `retrieval/verify.py` | Coverage judge + repair (capped) + consolidation triggers. |
| `eval/harness.py` (full) | Grid runner, blind judge, reports. |

**Wave 4 — Synthesis + vault** (parallel)

| Module | Notes |
|---|---|
| `ingestion/enrich.py` | First method: embedding-shortlisted tagging (cheapest candidate). Others behind the registry. |
| `synthesis/summaries.py` | L1 at ingest. |
| `synthesis/notes.py` + `synthesis/consolidation.py` | ONE agent: lazy policy and note synthesis are one design. |
| `synthesis/atlas.py` | Overview map-reduce, hubs. |
| `storage/vault.py` | Export, wikilink rendering. |

**→ Checkpoint B (human gate): S×R grid run vs the Checkpoint A baseline.** Standing questions in docs/evaluation.md get their first answers; weak channels (graph at R2, orientation) live or die here.

**Wave 5 — Agent**

| Module | Notes |
|---|---|
| `retrieval/graph.py` | Only if Checkpoint B kept it. |
| `agent/loop.py` | Tools over store read APIs; caps; trace logging for eval. |

## Conventions (every agent follows these)

1. **The docstring is the spec.** Keep it at the top, update "Planned API" to the real API. Read the linked docs/ file before writing code.
2. **Contracts are frozen.** Do not change another module's public API. If a contract is wrong, STOP and report to the orchestrator instead of working around it.
3. **Tests are part of the module.** `tests/test_<module>.py`, runnable offline: fake `LLMClient` for LLM logic, tiny fixture corpus for storage/retrieval, no network and no model downloads in unit tests (mark model-dependent tests for skip).
4. **Dependencies are closed by default**: stdlib + `numpy`, `sentence-transformers`, `google-genai`, `pymupdf`, `pdfplumber`, `pytest`. Anything else requires an entry in docs/decisions.md.
5. **Style**: type hints everywhere, English docstrings/comments, comments only for non-obvious constraints. No placeholder TODOs — unfinished means report, not stub.
6. **Update `pyproject.toml`** dependencies when a module actually imports something new.
7. **Every LLM call goes through MeteredClient.** No direct provider usage outside `llm/`.
8. **Commit per wave** (orchestrator does it, after integration + green tests), message `Wave N: <summary>`.

## Orchestrator workflow (main session)

1. Pick the next wave; for each module spawn a subagent with: module path, its docstring, the relevant docs file(s), the frozen contracts it consumes (paths of Wave-0 files), and the Conventions section above.
2. On completion: integrate, resolve conflicts yourself (do not respawn for trivial fixes), run `pytest`, run the wave's smoke test.
3. Commit, push, report to the human: what was built, test results, open issues, whether a checkpoint is reached.
4. At checkpoints: stop and wait for the human.
