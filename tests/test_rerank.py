"""Tests for wikirag.retrieval.rerank — offline, no model downloads.

The candidate objects are built locally against the frozen
``wikirag.retrieval.search.Candidate`` field contract, so these tests do
not depend on that module being implemented yet; ``rerank`` only ever
duck-types over the attributes below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pytest

from wikirag.llm.embeddings import FakeReranker
from wikirag.retrieval import rerank as rr


@dataclass
class Candidate:
    """Local stand-in with the frozen field set of search.Candidate."""

    owner_kind: str
    owner_id: str
    text: str
    doc_id: str | None = None
    section_id: str | None = None
    rrf_score: float = 0.0
    score: float | None = None


def candidate(owner_id: str, text: str, rrf_score: float = 0.0) -> Candidate:
    return Candidate(
        owner_kind="chunk",
        owner_id=owner_id,
        text=text,
        doc_id="doc-1",
        section_id="sec-1",
        rrf_score=rrf_score,
    )


class StubReranker:
    """Scores by lookup, and records the size of every score() call."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return "stub-reranker"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append(list(texts))
        return [self._scores[text] for text in texts]


def test_sets_scores_and_orders_by_score_desc() -> None:
    weak = candidate("c-weak", "weak")
    strong = candidate("c-strong", "strong")
    middle = candidate("c-middle", "middle")
    reranker = StubReranker({"weak": 0.05, "strong": 0.9, "middle": 0.4})

    ordered = rr.rerank("q", [weak, strong, middle], reranker=reranker)

    assert [c.owner_id for c in ordered] == ["c-strong", "c-middle", "c-weak"]
    assert [c.score for c in ordered] == [0.9, 0.4, 0.05]
    # The same objects come back, with every other field untouched.
    assert ordered[0] is strong
    assert strong.doc_id == "doc-1" and strong.rrf_score == 0.0


def test_ties_break_by_rrf_score_then_owner_id() -> None:
    # Equal reranker scores; deliberately inserted in the wrong order.
    low_rrf = candidate("a", "t-low", rrf_score=0.1)
    high_rrf = candidate("z", "t-high", rrf_score=0.9)
    same_rrf_b = candidate("b", "t-b", rrf_score=0.5)
    same_rrf_a = candidate("a2", "t-a", rrf_score=0.5)
    reranker = StubReranker({"t-low": 0.5, "t-high": 0.5, "t-b": 0.5, "t-a": 0.5})

    ordered = rr.rerank("q", [low_rrf, same_rrf_b, high_rrf, same_rrf_a], reranker=reranker)

    assert [c.owner_id for c in ordered] == ["z", "a2", "b", "a"]


def test_nothing_is_filtered_out() -> None:
    pool = [candidate(f"c-{i}", f"t{i}") for i in range(5)]
    reranker = StubReranker({f"t{i}": 0.0 for i in range(5)})

    ordered = rr.rerank("q", pool, reranker=reranker)

    assert len(ordered) == len(pool)
    assert {c.owner_id for c in ordered} == {c.owner_id for c in pool}
    assert all(c.score == 0.0 for c in ordered)


def test_empty_pool_returns_empty_without_touching_a_reranker(monkeypatch) -> None:
    from wikirag.llm import embeddings

    monkeypatch.setattr(
        embeddings, "get_reranker", lambda *a, **k: pytest.fail("resolved a reranker")
    )

    assert rr.rerank("q", []) == []


def test_works_with_the_fake_reranker() -> None:
    hit = candidate("c-hit", "the capital of France is Paris")
    miss = candidate("c-miss", "unrelated prose about turbines")

    ordered = rr.rerank("capital France", [miss, hit], reranker=FakeReranker())

    assert [c.owner_id for c in ordered] == ["c-hit", "c-miss"]
    assert ordered[0].score == 1.0
    assert ordered[1].score == 0.0


# --------------------------------------------------------------- lazy resolution


def test_reranker_argument_prevents_shared_resolution(monkeypatch) -> None:
    from wikirag.llm import embeddings

    monkeypatch.setattr(
        embeddings, "get_reranker", lambda *a, **k: pytest.fail("get_reranker was called")
    )

    ordered = rr.rerank("q", [candidate("c", "t")], reranker=StubReranker({"t": 0.7}))

    assert ordered[0].score == 0.7


def test_shared_reranker_is_resolved_once_when_omitted(monkeypatch) -> None:
    from wikirag.llm import embeddings

    reranker = StubReranker({"t1": 0.2, "t2": 0.3})
    calls: list[tuple] = []

    def fake_get_reranker(*args, **kwargs):
        calls.append((args, kwargs))
        return reranker

    monkeypatch.setattr(embeddings, "get_reranker", fake_get_reranker)

    rr.rerank("q", [candidate("c1", "t1"), candidate("c2", "t2")], batch_size=1)

    assert len(calls) == 1
    assert len(reranker.calls) == 2  # resolved once, used for every batch


# ----------------------------------------------------------------------- batching


def test_score_calls_are_chunked_at_batch_size() -> None:
    pool = [candidate(f"c-{i}", f"t{i}") for i in range(7)]
    reranker = StubReranker({f"t{i}": i / 10 for i in range(7)})

    ordered = rr.rerank("q", pool, reranker=reranker, batch_size=3)

    assert [len(call) for call in reranker.calls] == [3, 3, 1]
    assert [text for call in reranker.calls for text in call] == [f"t{i}" for i in range(7)]
    # Scores stay paired with their own candidate across batch boundaries.
    assert [c.owner_id for c in ordered] == [f"c-{i}" for i in reversed(range(7))]


def test_single_batch_when_pool_fits() -> None:
    pool = [candidate(f"c-{i}", f"t{i}") for i in range(4)]
    reranker = StubReranker({f"t{i}": 0.5 for i in range(4)})

    rr.rerank("q", pool, reranker=reranker, batch_size=32)

    assert [len(call) for call in reranker.calls] == [4]


def test_rejects_a_non_positive_batch_size() -> None:
    with pytest.raises(ValueError):
        rr.rerank("q", [candidate("c", "t")], reranker=StubReranker({"t": 0.5}), batch_size=0)


def test_rejects_a_score_count_mismatch() -> None:
    class ShortReranker:
        model_name = "short"

        def score(self, query: str, texts: Sequence[str]) -> list[float]:
            return [0.5] * (len(texts) - 1)

    with pytest.raises(ValueError):
        rr.rerank("q", [candidate("a", "t1"), candidate("b", "t2")], reranker=ShortReranker())


# ---------------------------------------------------------------------- no_answer


def test_no_answer_is_true_for_an_empty_pool() -> None:
    assert rr.no_answer([]) is True


def test_no_answer_around_the_threshold() -> None:
    below = candidate("c-below", "t")
    below.score = rr.NO_ANSWER_THRESHOLD - 0.01
    at = candidate("c-at", "t")
    at.score = rr.NO_ANSWER_THRESHOLD

    assert rr.no_answer([below]) is True
    assert rr.no_answer([at]) is False  # the threshold itself passes


def test_no_answer_uses_the_best_score_in_the_pool() -> None:
    weak = candidate("c-weak", "weak")
    strong = candidate("c-strong", "strong")
    ordered = rr.rerank(
        "q", [weak, strong], reranker=StubReranker({"weak": 0.01, "strong": 0.8})
    )

    assert rr.no_answer(ordered) is False
    assert rr.no_answer([weak]) is True


def test_threshold_sits_on_the_sigmoid_scale() -> None:
    assert 0.0 < rr.NO_ANSWER_THRESHOLD < 1.0
