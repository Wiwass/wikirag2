# Benchmark run: grid_R0_R2_flashlite

- **Knowledge base**: arsmagica-sample
- **Synthesis level**: S0
- **Corpus**: 5 of the arsmagicaV2 PDFs (the ones the 30 gold questions cite)
- **Dataset**: benchmarks/arsmagicaV2/dataset.json
- **Dataset sha256**: dee148828c1b
- **Questions**: 30
- **Answer model**: gemini-3.5-flash-lite
- **Judge model**: gemini-3.5-flash-lite
- **Embedding model**: BAAI/bge-m3
- **Reranker model**: BAAI/bge-reranker-v2-m3
- **Commit**: e0397dc09710fdc100e4c4e970ca9700df433769
- **Started**: 2026-09-01T07:49:22+00:00
- **Finished**: 2026-09-01T11:33:20+00:00
- **Cells with artifacts**: 2

Answers were graded by a blind judge (question, gold answer, key points and candidate answer only — never the depth, the budget or which cell produced it) on a 0-30 scale: correctness (0-12) + completeness (0-12) + groundedness (0-6).

## Simple questions (direct information retrieval)

| Depth | Budget | N | Judged | Avg. context tokens | Avg. score | SEM |
|---|---|---|---|---|---|---|
| R0 | 10000 | 15 | 15 | 7841 | 22.5 | ±2.3 |
| R2 | 10000 | 15 | 15 | 7879 | 23.1 | ±2.3 |

## Complex questions (multi-hop reasoning)

| Depth | Budget | N | Judged | Avg. context tokens | Avg. score | SEM |
|---|---|---|---|---|---|---|
| R0 | 10000 | 15 | 15 | 7841 | 15.1 | ±2.3 |
| R2 | 10000 | 15 | 15 | 7879 | 16.7 | ±2.5 |

## All questions

| Depth | Budget | N | Judged | Avg. context tokens | Avg. score | SEM |
|---|---|---|---|---|---|---|
| R0 | 10000 | 30 | 30 | 7841 | 18.8 | ±1.7 |
| R2 | 10000 | 30 | 30 | 7879 | 19.9 | ±1.8 |

**Note on significance:** with ~15 questions per subset the standard error of the mean is wide; score differences smaller than 3 points on the 0-30 scale should be read as noise, not as a winner (docs/evaluation.md, Practices). The context-token column is what makes cells comparable at a given cost: the nominal budget is an envelope, the tokens actually packed are what the model saw.

## Query cost per cell

From the KB's `llm_calls` meter (answers + judge). A resumed cell was not paid for in this run.

| Depth | Budget | Status | Cost |
|---|---|---|---|
| R0 | 10000 | done | $0.0857 |
| R2 | 10000 | done | $0.1679 |

Total metered this run: $0.2537.

## Standing questions (docs/evaluation.md)

| # | Question | This run |
|---|---|---|
| Q1 | Does the reranker beat plain RRF ordering at equal budget? | no — needs a rerank-ablation run; the grid varies depth and budget only |
| Q2 | Does any S level beat S0 at realistic budgets? | no — needs a second grid on a KB at another synthesis level |
| Q3 | Does orientation (R1) help or hurt vs skipping it? | no — this grid has no R1 cell |
| Q4 | Does the repair loop (R2) pay for its extra call on complex questions? | no — this grid has no R1 cell |
| Q5 | Does graph expansion at R2 add anything over R1 + structural expansion? | no — needs a graph-ablation run at R2, which this grid does not vary |
| Q6 | Is R3 worth its latency on hard questions? | no — this grid has no R1, R3 cell |

The S axis is fixed at S0 here: every S question needs a second grid on a KB ingested at another level (docs/evaluation.md, v2 measurement grid).
