# Status and roadmap

Implemented by waves (see [implementation.md](implementation.md)); the offline test suite passes (542 passed, 14 skipped in a fresh clone: the skipped tests load real models, run a scale benchmark, or read benchmark files that are not public).

| Part | State |
|---|---|
| Storage (SQLite store, vector index, vault export), LLM layer (Gemini client, cost metering, local embedder and reranker) | done |
| PDF extraction, deterministic chunking, cross-references, resumable ingest runner | done |
| Retrieval floor `R0`: hybrid search, RRF, cross-encoder rerank, context assembly; CLI | done |
| Depth `R1`/`R2`: orientation, question decomposition, verify-and-repair loop | done (depth auto-routing exists in `router.py` but is not wired: `--depth` defaults to `R1`) |
| Synthesis at ingest: `S1` summaries, `S2` keyword tagging; markdown vault export | done |
| Synthesis after ingest: lazy keyword notes, consolidation, atlas (`S2`-`S3`) | implemented and unit-tested, **not wired yet**: no CLI command runs them and `R2` does not queue consolidation yet |
| Evaluation harness (blind judge, depth × budget grid) | done |
| **Checkpoint A**: `S0` baseline measured on a 5-book sample (30 gold questions) | done, results in `benchmarks/` |
| **Checkpoint B**: `S × R` grid against the baseline | prepared, not run yet |
| `R3` agentic retrieval, graph channel (`retrieval/graph.py`) | not implemented (`R3` raises `NotImplementedError`) |

What is measured so far is small: on the 30-question gold set at a 10k-token budget, `R0` scores 18.8 and `R2` scores 19.9 out of 30 (answer and judge model `gemini-3.5-flash-lite`, standard error about ±1.8), which is a difference inside the noise. All six standing questions in [evaluation.md](evaluation.md) are still open; see `benchmarks/arsmagicaV2/runs/*/summary.md`. The benchmark corpus is a commercial RPG library: its PDFs are not in this repository, and neither are the gold questions and recorded answers, which quote it. [benchmarks/README.md](../benchmarks/README.md) describes the gold-set format, so the harness runs on any corpus.

## Roadmap

1. **Plan** (done): structure, contracts, decisions.
2. **Skeleton** (done): package layout with per-module specs as docstrings.
3. **Implementation by waves** (Waves 0-4 implemented; post-ingest synthesis not wired yet, see above): see [implementation.md](implementation.md) — foundations → ingest floor → retrieval floor (**Checkpoint A**: measured S0 baseline, done) → depth (R1/R2) → synthesis (lazy S1–S3) + vault.
4. **Checkpoint B** (next): wire consolidation and the atlas (a `consolidate` command, the `R2` trigger) so `S2`/`S3` knowledge bases can be built, then run the `S × R` grid against the Checkpoint A baseline and answer the standing questions in [evaluation.md](evaluation.md); weak channels (orientation, graph expansion) live or die there.
5. **Wave 5**: agentic retrieval (`R3`), and the graph channel only if Checkpoint B keeps it.
6. Continuous: ingest-method research measured against the floor.
