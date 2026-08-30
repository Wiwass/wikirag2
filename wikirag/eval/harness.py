"""Benchmark harness: run the S x R x budget grid, judge blind, report.

Ported/adapted from v1's benchmark/ package:
- gold datasets per KB (questions + gold answers + key points,
  simple/complex split) — the Ars Magica dataset carries over;
- blind judge (question, gold, key points, candidate; never system
  identity), 0-30 scale: correctness + completeness + groundedness;
- run artifacts kept (contexts, answers, judgments); baseline freezing;
  judge model pinned per dataset version;
- standing regression suite = one S row x all R x budgets; full grid on
  ingest-method changes;
- retrieval-only micro-benchmarks (recall@k on annotated chunks) for
  component choices (embedding model, reranker, vector backend) without
  paying judge costs.

Planned API
-----------
- run_grid(kb, dataset, cells, out_dir) -> RunReport
- judge_run(run_dir, judge_client) / summarize(run_dir)
"""
