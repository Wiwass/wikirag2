# Evaluation

The v1 benchmark harness is the most valuable thing v1 built and the reason v2's design is evidence-based. It gets ported early (roadmap step 3 in [status.md](status.md)) and becomes the gate for every architectural claim.

## What carries over from v1

- **Gold datasets** per KB: authored questions with gold answers and key points, split simple/complex. The Ars Magica dataset is reusable as-is since the corpus is the same.
- **Blind LLM judge**: sees question, gold answer, key points, candidate answer; never which system/budget produced it. Scores correctness + completeness + groundedness (0–30).
- **Ablation tiers × budget grid**, with SEM reported and a stated noise threshold.

## v2 measurement grid

The two sliders give the ablation structure for free:

| Axis | Cells |
|---|---|
| Synthesis | S0, S1, S2, S3 (same corpus ingested at each level) |
| Retrieval | R0, R1, R2, R3 |
| Budget | ~2.5k, 10k, 25k tokens |

Not the full cube every run: the standing regression suite is one row (S at the KB's production level × all R × budgets); the full grid runs when an ingestion method changes.

Per cell, report **quality** (judge score), **query cost** (latency, API tokens), and — per S level — **build cost** (from the `llm_calls` meter). The v2 promise is stated in these terms: each slider step must buy measurable quality or measurably reduce cost.

## Standing questions the harness must answer

1. Does the reranker beat plain RRF ordering at equal budget? (Expected: clearly yes; if not, the reranker choice is wrong.)
2. Does any S level beat S0 at realistic budgets? (v1's answer was "barely"; D11's honest prior is 45 — if it stays low, S is priced as vault value, not retrieval value.)
3. Does orientation (R1) help or hurt vs skipping it? (D12 prior: 55 — the weakest bet, boost-not-filter caps the downside.)
4. Does the repair loop (R2) pay for its extra call on complex questions? (Prior: 65.)
5. Does graph expansion at R2 add anything over R1 + structural expansion? (D10 prior: low; cut the channel without regret if not.)
6. Is R3 worth its latency on hard questions? (Prior: 70.)

## Practices

- Freeze a baseline run per corpus; regressions compare against it; keep run artifacts (contexts, answers, judgments) as v1 did.
- Judge model pinned per dataset-version; judge changes invalidate cross-run comparisons and force a re-baseline.
- Micro-benchmarks (embedding model choice, vector index choice, reranker choice) use retrieval-only metrics (recall@k against annotated relevant chunks) to avoid paying judge costs for component decisions.
