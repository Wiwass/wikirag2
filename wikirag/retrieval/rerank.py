"""Second stage: cross-encoder reranking. Part of the R0 floor (D12):
local and cheap, there is no reason to make it a level.

Takes a wide first-stage pool (e.g. top-50), scores each candidate's text
against the query with the local reranker, returns absolute scores.
Highest-leverage upgrade over v1 per the 2026 literature.

Absolute scores matter beyond ordering: a below-threshold top score is
the "nothing relevant found" signal (docs/retrieval.md) — a state v1's
per-query normalization made unrepresentable.

This module only scores and orders. It never drops candidates: inclusion
is assembly's decision, under its budget, and the no-answer gate is the
caller's (R1 coverage gate). One responsibility, one scale.

API
---
- ``rerank(query, candidates, *, reranker=None, batch_size=32)`` — sets
  ``.score`` on every candidate (reranker scale, ``(0, 1)``) and returns
  them sorted by score descending. ``reranker=None`` resolves the shared
  :func:`wikirag.llm.embeddings.get_reranker` instance lazily.
- ``NO_ANSWER_THRESHOLD`` — absolute score below which the pool counts as
  "nothing relevant found".
- ``no_answer(candidates)`` — that check over an already-scored pool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # contracts only; neither import happens at runtime
    from wikirag.llm.embeddings import RerankerLike
    from wikirag.retrieval.search import Candidate

#: Top-score floor on the reranker's sigmoid scale (D14: relevant pairs land
#: above ~0.5, clearly irrelevant ones below ~0.1). Provisional — it gets
#: calibrated on the benchmark at Checkpoint A, per reranker model
#: (docs/evaluation.md, standing question 1).
NO_ANSWER_THRESHOLD: float = 0.25

#: Chunk size for score() calls: the Reranker batches internally, but handing
#: it a whole wide pool in one call builds one big pair list and one long
#: uninterruptible forward run for no gain.
DEFAULT_BATCH_SIZE = 32


def rerank(
    query: str,
    candidates: Sequence["Candidate"],
    *,
    reranker: "RerankerLike | None" = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list["Candidate"]:
    """Score ``candidates`` against ``query`` and order them by relevance.

    Mutates each candidate's ``.score`` (absolute reranker scale, comparable
    across queries and pools) and returns the same objects, sorted by score
    descending, ties broken by first-stage ``rrf_score`` descending and then
    by ``owner_id``. Every other field is left untouched, and nothing is
    filtered out — assembly decides what makes it into the context.

    ``reranker`` defaults to the shared process instance, resolved on the
    first non-empty call so that importing this module (or reranking an
    empty pool) never touches the model registry.
    """
    if not candidates:
        return []
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    if reranker is None:
        from wikirag.llm import embeddings

        reranker = embeddings.get_reranker()

    scores: list[float] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        batch_scores = reranker.score(query, [candidate.text for candidate in batch])
        # A length mismatch would silently pair scores with the wrong
        # candidates, which is invisible downstream — refuse instead.
        if len(batch_scores) != len(batch):
            raise ValueError(
                f"reranker returned {len(batch_scores)} scores for {len(batch)} texts"
            )
        scores.extend(float(score) for score in batch_scores)

    for candidate, score in zip(candidates, scores):
        candidate.score = score

    return sorted(
        candidates,
        key=lambda candidate: (-float(candidate.score), -candidate.rrf_score, candidate.owner_id),
    )


def no_answer(candidates: Sequence["Candidate"]) -> bool:
    """Whether the pool says "nothing relevant found".

    True for an empty pool, or when the best absolute score is below
    :data:`NO_ANSWER_THRESHOLD`. Assumes ``.score`` is set on every
    candidate — i.e. that the pool has been through :func:`rerank`.
    """
    if not candidates:
        return True
    return max(float(candidate.score) for candidate in candidates) < NO_ANSWER_THRESHOLD
