"""Eval: benchmark harness — the regression gate (docs/evaluation.md).

Ported from v1 (gold datasets, blind LLM judge, ablation grid, SEM
reporting); v2's grid is S-level x R-depth x budget, plus build-cost
reporting from the llm_calls meter.

Modules:
    harness   dataset loading, grid runner, judge, reports
"""
