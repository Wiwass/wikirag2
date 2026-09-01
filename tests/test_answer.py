"""Answering tests: pipeline wiring, the no-answer gate, prompts, logging.

Fully offline. The retrieval pipeline is faked by monkeypatching the names
`answer.py` imported (hybrid_search, rerank, no_answer, assemble), so no
retrieval module, reranker model or vector index is involved; the KB is a
real (empty) store, because query logging is part of the contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from wikirag import answer as answer_module
from wikirag.answer import (
    HISTORY_WINDOW,
    NO_ANSWER_TEXT,
    RESERVED_ANSWER_TOKENS,
    Answer,
    answer,
)
from wikirag.config import KnowledgeBase, RetrievalDepth, create_kb
from wikirag.llm.client import FakeLLMClient, LLMResult, Usage
from wikirag.retrieval.verify import DEFAULT_MAX_ROUNDS


@pytest.fixture()
def kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterable[KnowledgeBase]:
    monkeypatch.setenv("WIKIRAG_HOME", str(tmp_path / "kbs"))
    with create_kb("test") as handle:
        yield handle


@dataclass
class FakeCandidate:
    """The retrieval.search.Candidate shape, without the module."""

    owner_kind: str = "chunk"
    owner_id: str = "c1"
    text: str = "some retrieved text"
    doc_id: str | None = "doc1"
    section_id: str | None = "doc1:1"
    rrf_score: float = 0.5
    score: float | None = None


@dataclass
class FakeEntry:
    """The assembly.ContextEntry shape."""

    ref: int = 1
    owner_kind: str = "chunk"
    owner_id: str = "c1"
    doc_id: str | None = "doc1"
    section_id: str | None = "doc1:1"
    section_title: str | None = "Spells"
    doc_title: str | None = "Grimoire"
    chunk_ids: list[str] = field(default_factory=lambda: ["c1"])
    score: float = 0.9
    text: str = "some retrieved text"


@dataclass
class FakeContext:
    text: str = "[1] Grimoire — Spells\nsome retrieved text"
    entries: list[FakeEntry] = field(default_factory=lambda: [FakeEntry()])
    token_count: int = 42


@dataclass
class FakeRegions:
    """The orient.Regions shape: an identity boost, as on an S0 KB."""

    def is_empty(self) -> bool:
        return True

    def boost(self, candidates: Any) -> list[Any]:
        return list(candidates)


@dataclass
class FakeClassification:
    """The router.Classification shape."""

    is_complex: bool = False
    subquestions: list[str] = field(default_factory=list)
    retrieval_guidance: str = ""
    fallback: bool = False


@dataclass
class FakeCoverage:
    """The verify.Coverage shape."""

    ok: bool = True
    missing_aspects: list[str] = field(default_factory=list)
    repair_queries: list[str] = field(default_factory=list)
    fallback: bool = False


class FakePipeline:
    """Records every stage call and returns scripted results."""

    def __init__(
        self,
        *,
        candidates: list[FakeCandidate] | None = None,
        no_answer: bool = False,
        context: FakeContext | None = None,
    ) -> None:
        self.candidates = candidates if candidates is not None else [FakeCandidate()]
        self._no_answer = no_answer
        self.context = context if context is not None else FakeContext()
        self.calls: dict[str, list[dict[str, Any]]] = {
            "hybrid_search": [],
            "rerank": [],
            "no_answer": [],
            "assemble": [],
            "orient": [],
            "classify": [],
            "check_coverage": [],
        }
        #: scripted Wave-3 stage results; overwritten per test
        self.classification = FakeClassification()
        self.coverages: list[FakeCoverage] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "FakePipeline":
        monkeypatch.setattr(answer_module, "hybrid_search", self.hybrid_search)
        monkeypatch.setattr(answer_module, "rerank", self.rerank)
        monkeypatch.setattr(answer_module, "no_answer", self.no_answer)
        monkeypatch.setattr(answer_module, "assemble", self.assemble)
        monkeypatch.setattr(answer_module, "orient", self.orient)
        monkeypatch.setattr(answer_module, "classify", self.classify)
        monkeypatch.setattr(answer_module, "check_coverage", self.check_coverage)
        monkeypatch.setattr(answer_module, "anchors", self.anchors)
        return self

    def anchors(self, query: str, classification: Any) -> list[str]:
        return [query, *classification.subquestions]

    def orient(self, kb: Any, query: str, *, enabled: bool = True) -> "FakeRegions":
        self.calls["orient"].append({"query": query, "enabled": enabled})
        return FakeRegions()

    def classify(self, client: Any, query: str, **kwargs: Any) -> "FakeClassification":
        self.calls["classify"].append({"query": query})
        return self.classification

    def check_coverage(self, client: Any, query: str, context: Any, **kwargs: Any) -> "FakeCoverage":
        self.calls["check_coverage"].append({"query": query})
        if self.coverages:
            return self.coverages.pop(0)
        return FakeCoverage(ok=True)

    def hybrid_search(self, kb: Any, queries: Any, k: int = 50, **kwargs: Any) -> list[FakeCandidate]:
        self.calls["hybrid_search"].append({"queries": list(queries), "k": k})
        return list(self.candidates)

    def rerank(self, query: str, candidates: Any, **kwargs: Any) -> list[FakeCandidate]:
        self.calls["rerank"].append({"query": query, "candidates": list(candidates)})
        for candidate in candidates:
            candidate.score = 0.9
        return list(candidates)

    def no_answer(self, candidates: Any) -> bool:
        self.calls["no_answer"].append({"candidates": list(candidates)})
        return self._no_answer

    def assemble(self, kb: Any, scored: Any, budget_tokens: int, **kwargs: Any) -> FakeContext:
        self.calls["assemble"].append({"scored": list(scored), "budget_tokens": budget_tokens})
        return self.context


def make_client(text: str = "The spell needs vis [1].") -> FakeLLMClient:
    return FakeLLMClient(
        LLMResult(
            text=text,
            usage=Usage(prompt_tokens=120, output_tokens=17, total_tokens=137),
            cost_estimate=0.0001,
        )
    )


def test_happy_path_builds_context_and_answers(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = FakePipeline().install(monkeypatch)
    client = make_client()

    reply = answer(kb, "How does the spell work?", client=client, depth=RetrievalDepth.R0)

    assert isinstance(reply, Answer)
    assert reply.text == "The spell needs vis [1]."
    assert reply.provenance == pipeline.context.entries
    assert reply.usage.prompt_tokens == 120
    assert reply.usage.output_tokens == 17
    assert reply.no_answer is False
    assert reply.depth is RetrievalDepth.R0
    assert reply.context_tokens == 42

    assert pipeline.calls["hybrid_search"][0]["queries"] == ["How does the spell work?"]
    call = client.calls[0]
    assert "[1] Grimoire — Spells" in call["user"]
    assert "Question: How does the spell work?" in call["user"]
    assert "[1]" in call["system"]  # the citation rule reaches the model


def test_query_is_logged(kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch) -> None:
    FakePipeline().install(monkeypatch)

    answer(kb, "what is vis?", client=make_client(), depth="R2")

    logged = kb.store.get_queries()
    assert len(logged) == 1
    assert logged[0]["query"] == "what is vis?"
    assert logged[0]["depth"] == "R2"
    assert logged[0]["no_answer"] is False


def test_no_answer_skips_the_model_and_is_logged(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = FakePipeline(no_answer=True).install(monkeypatch)
    client = make_client()

    reply = answer(kb, "who won in 1998?", client=client)

    assert reply.no_answer is True
    assert reply.text == NO_ANSWER_TEXT
    assert reply.provenance == []
    assert reply.context_tokens == 0
    assert client.calls == []  # no generate call at all
    assert pipeline.calls["assemble"] == []  # and no context assembled either

    logged = kb.store.get_queries()
    assert len(logged) == 1
    assert logged[0]["no_answer"] is True


def test_history_is_windowed_and_marked_as_history(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakePipeline().install(monkeypatch)
    client = make_client()
    history = [(f"question {i}", f"reply {i}") for i in range(HISTORY_WINDOW + 3)]

    answer(kb, "and then?", client=client, history=history)

    user = client.calls[0]["user"]
    for i in range(3):  # the oldest exchanges fall out of the window
        assert f"question {i}" not in user
    for i in range(3, HISTORY_WINDOW + 3):
        assert f"question {i}" in user
        assert f"reply {i}" in user
    assert "not evidence" in user


def test_no_history_block_when_there_is_no_history(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakePipeline().install(monkeypatch)
    client = make_client()

    answer(kb, "first question", client=client)

    assert "Conversation history" not in client.calls[0]["user"]


def test_budget_reserves_output_tokens(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = FakePipeline().install(monkeypatch)
    client = make_client()

    answer(kb, "q", client=client, budget_tokens=10_000)

    assert pipeline.calls["assemble"][0]["budget_tokens"] == 10_000 - RESERVED_ANSWER_TOKENS
    assert client.calls[0]["max_output_tokens"] == RESERVED_ANSWER_TOKENS


def test_small_budget_still_retrieves(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = FakePipeline().install(monkeypatch)

    answer(kb, "q", client=make_client(), budget_tokens=1_000)

    # The reservation is proportional below the cap, so context still gets most.
    assert pipeline.calls["assemble"][0]["budget_tokens"] == 800


def test_r0_skips_orientation_and_the_no_answer_gate(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R0 is the benchmark control: it answers from whatever it found."""
    pipeline = FakePipeline(no_answer=True).install(monkeypatch)
    client = make_client()

    reply = answer(kb, "q", client=client, depth="R0")

    assert reply.no_answer is False  # the gate belongs to R1+
    assert pipeline.calls["orient"] == []
    assert pipeline.calls["classify"] == []
    assert pipeline.calls["check_coverage"] == []
    assert len(client.calls) == 1


def test_r1_orients_and_applies_the_gate(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = FakePipeline(no_answer=True).install(monkeypatch)

    reply = answer(kb, "q", client=make_client(), depth="R1")

    assert reply.no_answer is True
    assert pipeline.calls["orient"] == [{"query": "q", "enabled": True}]
    assert pipeline.calls["classify"] == []  # decomposition is R2


def test_orientation_can_be_ablated(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evaluation.md Q3: orientation must be switchable off for the ablation."""
    pipeline = FakePipeline().install(monkeypatch)

    answer(kb, "q", client=make_client(), depth="R1", orient_enabled=False)

    assert pipeline.calls["orient"] == [{"query": "q", "enabled": False}]


def test_r2_searches_subquestions_as_extra_anchors(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = FakePipeline().install(monkeypatch)
    pipeline.classification = FakeClassification(
        is_complex=True,
        subquestions=["what is vis?", "how is it gathered?"],
        retrieval_guidance="Compare the two chapters.",
    )

    client = make_client()
    reply = answer(kb, "q", client=client, depth="R2")

    assert pipeline.calls["hybrid_search"][0]["queries"] == [
        "q",
        "what is vis?",
        "how is it gathered?",
    ]
    assert reply.anchors == ["q", "what is vis?", "how is it gathered?"]
    # The guidance reaches the answering model, marked as guidance not fact.
    assert "Compare the two chapters." in client.calls[-1]["user"]


def test_r2_repairs_once_when_coverage_falls_short(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = FakePipeline().install(monkeypatch)
    pipeline.coverages = [
        FakeCoverage(ok=False, missing_aspects=["cost"], repair_queries=["vis cost"]),
        FakeCoverage(ok=True),
    ]

    reply = answer(kb, "q", client=make_client(), depth="R2")

    assert reply.repair_rounds == 1
    # One extra retrieval round on the judge's queries, and a re-assembly.
    assert [c["queries"] for c in pipeline.calls["hybrid_search"]] == [["q"], ["vis cost"]]
    assert len(pipeline.calls["assemble"]) == 2


def test_r2_repair_is_hard_capped(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A judge that is never satisfied must not loop: the cap is the contract."""
    pipeline = FakePipeline().install(monkeypatch)
    pipeline.coverages = [
        FakeCoverage(ok=False, repair_queries=[f"q{i}"]) for i in range(5)
    ]

    reply = answer(kb, "q", client=make_client(), depth="R2")

    assert reply.repair_rounds == DEFAULT_MAX_ROUNDS == 1
    assert len(pipeline.calls["check_coverage"]) == 1


def test_r2_does_not_repair_when_coverage_is_ok(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = FakePipeline().install(monkeypatch)
    pipeline.coverages = [FakeCoverage(ok=True)]

    reply = answer(kb, "q", client=make_client(), depth="R2")

    assert reply.repair_rounds == 0
    assert len(pipeline.calls["hybrid_search"]) == 1


def test_r3_is_not_implemented(kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch) -> None:
    FakePipeline().install(monkeypatch)

    with pytest.raises(NotImplementedError, match="Wave 5"):
        answer(kb, "q", client=make_client(), depth="R3")

    assert kb.store.get_queries() == []


def test_unknown_depth_is_rejected(kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch) -> None:
    FakePipeline().install(monkeypatch)

    with pytest.raises(ValueError, match="RetrievalDepth"):
        answer(kb, "q", client=make_client(), depth="R9")


def test_pool_size_reaches_search(kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = FakePipeline().install(monkeypatch)

    answer(kb, "q", client=make_client(), pool_size=12)

    assert pipeline.calls["hybrid_search"][0]["k"] == 12


def test_empty_context_is_stated_not_faked(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakePipeline(context=FakeContext(text="", entries=[], token_count=0)).install(monkeypatch)
    client = make_client()

    answer(kb, "q", client=client)

    assert "No knowledge-base context" in client.calls[0]["user"]
