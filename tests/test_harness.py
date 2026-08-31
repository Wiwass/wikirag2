"""Tests for wikirag.eval.harness — offline: temp KB, FTS only, fake clients.

No network, no model downloads: retrieval runs lexically (no vector index is
written, so hybrid_search never resolves an embedder) and is reranked by
FakeReranker, whose score is the fraction of query tokens present in a
text — which makes every hit in these tests hand-verifiable.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from wikirag.answer import Answer
from wikirag.config import WIKIRAG_HOME_ENV, KnowledgeBase, RetrievalDepth, create_kb
from wikirag.eval import harness
from wikirag.eval.harness import (
    JUDGE_RESPONSE_SCHEMA,
    GoldQuestion,
    Judgment,
    _validate_judgment,
    answer_run,
    judge_answer,
    judge_answers,
    load_dataset,
    match_doc_ids,
    parse_cited_pages,
    retrieval_eval,
    slugify,
    summarize,
)
from wikirag.llm.client import AllKeysExhausted, FakeLLMClient, LLMResult, Usage
from wikirag.llm.embeddings import FakeReranker
from wikirag.retrieval.assembly import ContextEntry

DATASET_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "arsmagicaV2" / "dataset.json"

DOC_A = "ag0205-ars-magica"
DOC_B = "ag0280-covenants"


# --------------------------------------------------------------- citations


@pytest.mark.parametrize(
    ("citation", "expected"),
    [
        ("page 95", [95]),
        ("page 7", [7]),
        ("pages 64, 110, 111", [64, 110, 111]),
        ("pages 10-11, 25", [10, 11, 25]),
        ("pages 121, 130, 90", [90, 121, 130]),
        ("pages 10–12", [10, 11, 12]),  # en dash, as a word processor writes it
        ("Pages 5", [5]),
        ("page 12, page 12", [12]),  # duplicates collapse
    ],
)
def test_parse_cited_pages_formats(citation: str, expected: list[int]) -> None:
    assert parse_cited_pages(citation) == expected


@pytest.mark.parametrize(
    "citation",
    ["", None, "   ", "see the appendix", "chapter 3", "Ars Magica, 5th edition", "p. 12"],
)
def test_parse_cited_pages_garbage_is_empty(citation: str | None) -> None:
    assert parse_cited_pages(citation) == []


def test_parse_cited_pages_absurd_range_keeps_endpoints_only() -> None:
    # A 1..900 "range" is a typo, not an annotation of 900 relevant pages.
    assert parse_cited_pages("pages 1-900") == [1, 900]
    assert parse_cited_pages("pages 30-20") == [20, 30]


def test_load_dataset_reads_the_committed_gold_set() -> None:
    questions = load_dataset(DATASET_PATH)
    assert len(questions) == 30
    assert sum(q.type == "simple" for q in questions) == 15
    assert sum(q.type == "complex" for q in questions) == 15
    first = questions[0]
    assert first.id == "q01"
    assert first.cited_pages == [7]
    assert first.source_docs == ["AG0205 - Ars Magica"]
    assert first.key_points
    # Every citation in the approved dataset parses; a regression here would
    # silently turn page_hit into doc_hit for the affected questions.
    assert all(q.cited_pages for q in questions)


def test_load_dataset_rejects_a_broken_question(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps({"questions": [{"id": "q1", "type": "tricky", "question": "a", "gold_answer": "b"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="'type' must be one of"):
        load_dataset(path)


# ------------------------------------------------------------ doc matching


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("AG0205 - Ars Magica", "ag0205-ars-magica"),
        ("AG0279 - The Mysteries [revised ed]", "ag0279-the-mysteries-revised-ed"),
        ("AG0277 - Houses of Hermes - True Lineages (1)", "ag0277-houses-of-hermes-true-lineages-1"),
    ],
)
def test_slugify_matches_ingestion_doc_ids(title: str, expected: str) -> None:
    assert slugify(title) == expected


def test_match_doc_ids_exact_prefix_and_missing(kb: KnowledgeBase) -> None:
    matched = match_doc_ids(
        kb,
        [
            "AG0205 - Ars Magica",  # exact slug
            "AG0280 - Covenants [revised]",  # slug prefix of the dataset title
            "AG9999 - Missing Book",
        ],
    )
    assert matched == {
        "AG0205 - Ars Magica": DOC_A,
        "AG0280 - Covenants [revised]": DOC_B,
        "AG9999 - Missing Book": None,
    }


# ------------------------------------------------------------------ fixture

# Chunk texts are chosen so FakeReranker's score is a fraction with a small
# denominator: every ordering below is arithmetic, not model behaviour.
CHUNKS: tuple[tuple[str, str, str, str], ...] = (
    (DOC_A, "a1", "s1", "botch dice table chance percentage"),
    (DOC_A, "a2", "s2", "certamen duel of wizards"),
    (DOC_B, "b1", "s3", "library summae tractatus"),
    (DOC_B, "b2", "s4", "appendix of tables"),
)


@pytest.fixture()
def kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[KnowledgeBase]:
    monkeypatch.setenv(WIKIRAG_HOME_ENV, str(tmp_path))
    knowledge_base = create_kb("bench")
    store = knowledge_base.store
    store.add_document(doc_id=DOC_A, filename="AG0205 - Ars Magica.pdf")
    store.add_document(doc_id=DOC_B, filename="AG0280 - Covenants.pdf")
    store.add_sections(
        DOC_A,
        [
            {"id": "s1", "ordinal": 0, "title": "Botch Chance", "page_start": 5, "page_end": 8},
            {"id": "s2", "ordinal": 1, "title": "Certamen", "page_start": 40, "page_end": 44},
        ],
    )
    store.add_sections(
        DOC_B,
        [
            {"id": "s3", "ordinal": 0, "title": "Library", "page_start": 90, "page_end": 95},
            {"id": "s4", "ordinal": 1, "title": "Appendix"},  # no page range recorded
        ],
    )
    store.add_chunks(
        [
            {"id": cid, "doc_id": doc, "section_id": sid, "ordinal": i, "text": text}
            for i, (doc, cid, sid, text) in enumerate(CHUNKS)
        ]
    )
    with knowledge_base:
        yield knowledge_base


def gold(
    qid: str,
    question: str,
    docs: list[str],
    citation: str,
    qtype: str = "simple",
) -> GoldQuestion:
    return GoldQuestion(
        id=qid,
        type=qtype,
        question=question,
        gold_answer=f"gold for {qid}",
        key_points=[f"kp for {qid}"],
        source_docs=docs,
        citation=citation,
        cited_pages=parse_cited_pages(citation),
    )


# --------------------------------------------------------- retrieval_eval


def test_retrieval_eval_doc_and_page_hits(kb: KnowledgeBase) -> None:
    reranker = FakeReranker()
    questions = [
        # a1 is the only chunk with any of these tokens: rank 1, score 3/3.
        gold("q1", "botch dice chance", ["AG0205 - Ars Magica"], "page 7"),
        # b1 scores 2/3 (library, summae), a1 scores 1/3 (botch): gold doc A
        # is rank 2, so k=1 misses and k=2 hits.
        gold("q2", "library summae botch", ["AG0205 - Ars Magica"], "page 7", qtype="complex"),
        # right document, cited page far outside section s3 (90-95).
        gold("q3", "library summae", ["AG0280 - Covenants"], "page 200", qtype="complex"),
        # the source document is not in this KB at all.
        gold("q4", "certamen duel", ["AG9999 - Missing Book"], "no page here"),
    ]

    report = retrieval_eval(kb, questions, ks=(1, 2), reranker=reranker)
    by_id = {q.id: q for q in report.questions}

    assert by_id["q1"].doc_hit == {1: True, 2: True}
    assert by_id["q1"].page_hit == {1: True, 2: True}
    assert by_id["q1"].top_score == pytest.approx(1.0)
    assert by_id["q1"].would_no_answer is False

    assert by_id["q2"].doc_hit == {1: False, 2: True}
    assert by_id["q2"].page_hit == {1: False, 2: True}

    assert by_id["q3"].doc_hit == {1: True, 2: True}
    assert by_id["q3"].page_hit == {1: False, 2: False}

    assert by_id["q4"].gold_doc_ids == []
    assert by_id["q4"].unmatched_source_docs == ["AG9999 - Missing Book"]
    assert by_id["q4"].doc_hit == {1: False, 2: False}
    # No parsable citation: the question is scored doc-only.
    assert by_id["q4"].cited_pages == []
    assert by_id["q4"].page_hit == by_id["q4"].doc_hit


def test_retrieval_eval_page_tolerance_absorbs_the_offset(kb: KnowledgeBase) -> None:
    question = gold("q", "library summae", ["AG0280 - Covenants"], "page 97")  # s3 = 90-95

    strict = retrieval_eval(kb, [question], ks=(1,), reranker=FakeReranker(), page_tolerance=1)
    loose = retrieval_eval(kb, [question], ks=(1,), reranker=FakeReranker(), page_tolerance=2)

    assert strict.questions[0].page_hit == {1: False}
    assert loose.questions[0].page_hit == {1: True}


def test_retrieval_eval_section_without_pages_cannot_page_hit(kb: KnowledgeBase) -> None:
    # b2 lives in section s4, which has no recorded page range.
    question = gold("q", "appendix tables", ["AG0280 - Covenants"], "page 91")
    report = retrieval_eval(kb, [question], ks=(1,), reranker=FakeReranker())
    assert report.questions[0].doc_hit == {1: True}
    assert report.questions[0].page_hit == {1: False}


def test_retrieval_eval_empty_pool_is_a_no_answer(kb: KnowledgeBase) -> None:
    question = gold("q", "zzzz nothing matches", ["AG0205 - Ars Magica"], "page 7")
    report = retrieval_eval(kb, [question], ks=(1,), reranker=FakeReranker())
    outcome = report.questions[0]
    assert outcome.n_candidates == 0
    assert outcome.top_score is None
    assert outcome.would_no_answer is True
    assert outcome.doc_hit == {1: False}


def test_retrieval_report_rows_split_simple_complex_overall(kb: KnowledgeBase) -> None:
    questions = [
        gold("q1", "botch dice chance", ["AG0205 - Ars Magica"], "page 7"),
        gold("q2", "library summae", ["AG0280 - Covenants"], "page 200", qtype="complex"),
    ]
    rows = retrieval_eval(kb, questions, ks=(2,), reranker=FakeReranker()).rows()

    assert [row["subset"] for row in rows] == ["simple", "complex", "overall"]
    assert {row["subset"]: row["doc_hit"] for row in rows} == {
        "simple": 1.0,
        "complex": 1.0,
        "overall": 1.0,
    }
    assert {row["subset"]: row["page_hit"] for row in rows} == {
        "simple": 1.0,
        "complex": 0.0,
        "overall": 0.5,
    }
    overall = rows[-1]
    assert overall["n"] == 2
    assert overall["n_with_pages"] == 2
    assert overall["n_without_gold_docs"] == 0


# --------------------------------------------------------------- answer_run


def stub_answer(text: str = "an answer", *, no_answer: bool = False) -> Answer:
    return Answer(
        text=text,
        provenance=[
            ContextEntry(
                ref=1,
                owner_kind="chunk",
                owner_id="a1",
                doc_id=DOC_A,
                section_id="s1",
                section_title="Botch Chance",
                doc_title="Ars Magica",
                chunk_ids=["a1"],
                score=0.9,
                text="botch dice table",
            )
        ],
        usage=Usage(prompt_tokens=100, output_tokens=20, total_tokens=120),
        no_answer=no_answer,
        context_tokens=42,
    )


def test_answer_run_writes_the_artifact_shape(
    kb: KnowledgeBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake_answer(_kb, query, **kwargs):
        seen.append(query)
        return stub_answer(f"answer to {query}")

    monkeypatch.setattr(harness, "answer", fake_answer)
    questions = [
        gold("q1", "botch dice chance", ["AG0205 - Ars Magica"], "page 7"),
        gold("q2", "library summae", ["AG0280 - Covenants"], "page 95", qtype="complex"),
    ]
    out = tmp_path / "runs" / "run.json"

    artifact = answer_run(
        kb,
        questions,
        client=FakeLLMClient(LLMResult(text="unused"), model="fake-model"),
        depth=RetrievalDepth.R0,
        budget_tokens=2500,
        out_path=out,
    )

    assert seen == ["botch dice chance", "library summae"]
    assert artifact == json.loads(out.read_text(encoding="utf-8"))
    meta = artifact["meta"]
    assert meta["kb"] == "bench"
    assert meta["depth"] == "R0"
    assert meta["budget_tokens"] == 2500
    assert meta["model"] == "fake-model"
    assert meta["ts"]
    first = artifact["results"][0]
    assert first["id"] == "q1"
    assert first["type"] == "simple"
    assert first["answer"] == "answer to botch dice chance"
    assert first["no_answer"] is False
    assert first["context_tokens"] == 42
    assert first["usage"] == {
        "prompt_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "cached_tokens": 0,
    }
    assert first["provenance"] == [{"doc_id": DOC_A, "section_title": "Botch Chance", "ref": 1}]


def test_answer_run_records_one_failure_and_continues(
    kb: KnowledgeBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_answer(_kb, query, **kwargs):
        if query == "boom":
            raise RuntimeError("provider hiccup")
        return stub_answer()

    monkeypatch.setattr(harness, "answer", fake_answer)
    questions = [
        gold("q1", "boom", ["AG0205 - Ars Magica"], "page 7"),
        gold("q2", "library summae", ["AG0280 - Covenants"], "page 95"),
    ]

    artifact = answer_run(
        kb,
        questions,
        client=FakeLLMClient(LLMResult(text="unused")),
        out_path=tmp_path / "run.json",
    )

    assert [r["id"] for r in artifact["results"]] == ["q1", "q2"]
    assert artifact["results"][0]["error"] == "RuntimeError: provider hiccup"
    assert "answer" not in artifact["results"][0]
    assert artifact["results"][1]["answer"] == "an answer"
    assert "aborted" not in artifact["meta"]


def test_answer_run_aborts_on_exhausted_keys_and_saves_partial(
    kb: KnowledgeBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_answer(_kb, query, **kwargs):
        if query == "second":
            raise AllKeysExhausted("no quota left")
        return stub_answer()

    monkeypatch.setattr(harness, "answer", fake_answer)
    questions = [
        gold("q1", "first", ["AG0205 - Ars Magica"], "page 7"),
        gold("q2", "second", ["AG0205 - Ars Magica"], "page 7"),
        gold("q3", "third", ["AG0205 - Ars Magica"], "page 7"),
    ]
    out = tmp_path / "run.json"

    artifact = answer_run(
        kb, questions, client=FakeLLMClient(LLMResult(text="unused")), out_path=out
    )

    assert [r["id"] for r in artifact["results"]] == ["q1"]
    assert artifact["meta"]["aborted"].startswith("AllKeysExhausted:")
    assert json.loads(out.read_text(encoding="utf-8"))["results"] == artifact["results"]


# ------------------------------------------------------------- blind judge


def test_validate_judgment_clamps_each_field_and_recomputes_total() -> None:
    judgment = _validate_judgment(
        {
            "correctness": 99,
            "completeness": -4,
            "groundedness": 10,
            "total": 30,  # the model's own total is ignored
            "rationale": "  solid  ",
        }
    )
    assert (judgment.correctness, judgment.completeness, judgment.groundedness) == (12, 0, 6)
    assert judgment.total == 18
    assert judgment.rationale == "solid"
    assert judgment.fallback is False


def test_validate_judgment_survives_non_integer_garbage() -> None:
    judgment = _validate_judgment(
        {"correctness": "abc", "completeness": None, "groundedness": [1], "rationale": 7}
    )
    assert (judgment.correctness, judgment.completeness, judgment.groundedness) == (0, 0, 0)
    assert judgment.total == 0
    assert judgment.rationale == ""


def test_validate_judgment_accepts_numeric_strings_and_missing_fields() -> None:
    judgment = _validate_judgment({"correctness": "9", "completeness": 3.7})
    assert judgment.correctness == 9
    assert judgment.completeness == 3  # int() truncates, then clamps
    assert judgment.groundedness == 0
    assert judgment.total == 12


def judge_json(correctness: int, completeness: int, groundedness: int, rationale: str = "ok") -> str:
    return json.dumps(
        {
            "correctness": correctness,
            "completeness": completeness,
            "groundedness": groundedness,
            "rationale": rationale,
        }
    )


def test_judge_answer_is_blind_and_uses_the_json_schema() -> None:
    client = FakeLLMClient(LLMResult(text=judge_json(10, 8, 5)))
    judgment = judge_answer(
        client,
        "How many botch dice?",
        "Five botch dice give 4.1%.",
        ["4.1% for five botch dice"],
        "The chance is 4.1%.",
    )

    assert judgment == Judgment(
        correctness=10, completeness=8, groundedness=5, total=23, rationale="ok"
    )
    call = client.calls[0]
    assert call["json_schema"] == JUDGE_RESPONSE_SCHEMA
    assert call["system"] == harness.JUDGE_SYSTEM_PROMPT
    assert "How many botch dice?" in call["user"]
    assert "Five botch dice give 4.1%." in call["user"]
    assert "- 4.1% for five botch dice" in call["user"]
    assert "The chance is 4.1%." in call["user"]
    # Blind: nothing about retrieval, depth, budget or which system answered.
    for leak in ("provenance", "context", "depth", "budget", "wikirag"):
        assert leak not in call["user"].lower()


def test_judge_answer_strips_markdown_fences() -> None:
    client = FakeLLMClient(LLMResult(text=f"```json\n{judge_json(12, 12, 6)}\n```"))
    assert judge_answer(client, "q", "gold", [], "candidate").total == 30


def test_judge_answer_retries_malformed_json_then_succeeds() -> None:
    client = FakeLLMClient(
        LLMResult(text="not json at all"),
        LLMResult(text=judge_json(6, 6, 3)),
    )
    judgment = judge_answer(client, "q", "gold", ["kp"], "candidate")
    assert len(client.calls) == 2
    assert judgment.total == 15
    assert judgment.fallback is False


def test_judge_answer_falls_back_after_max_attempts() -> None:
    client = FakeLLMClient(LLMResult(text="still not json"))
    judgment = judge_answer(client, "q", "gold", ["kp"], "candidate", max_attempts=2)
    assert len(client.calls) == 2
    assert judgment == Judgment(fallback=True)
    assert judgment.total == 0


def test_judge_answer_falls_back_on_a_non_object_payload() -> None:
    client = FakeLLMClient(LLMResult(text="[1, 2, 3]"))
    assert judge_answer(client, "q", "gold", [], "candidate").fallback is True


def test_judge_answer_falls_back_on_a_provider_error() -> None:
    class BoomClient:
        model = "boom"

        def generate(self, **kwargs):
            raise RuntimeError("500 from provider")

    judgment = judge_answer(BoomClient(), "q", "gold", [], "candidate")
    assert judgment.fallback is True
    assert judgment.total == 0


def test_judge_answer_propagates_exhausted_keys() -> None:
    class ExhaustedClient:
        model = "spent"

        def generate(self, **kwargs):
            raise AllKeysExhausted("quota")

    with pytest.raises(AllKeysExhausted):
        judge_answer(ExhaustedClient(), "q", "gold", [], "candidate")


def artifact_file(tmp_path: Path) -> Path:
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "meta": {"kb": "bench", "depth": "R1", "budget_tokens": 10000},
                "results": [
                    {"id": "q1", "type": "simple", "question": "q1?", "answer": "a1"},
                    {"id": "q2", "type": "complex", "question": "q2?", "error": "RuntimeError: x"},
                    {"id": "q3", "type": "complex", "question": "q3?", "answer": "a3"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_judge_answers_writes_judgments_back_and_skips_errors(tmp_path: Path) -> None:
    path = artifact_file(tmp_path)
    questions = [
        gold("q1", "q1?", ["AG0205 - Ars Magica"], "page 7"),
        gold("q2", "q2?", ["AG0205 - Ars Magica"], "page 7", qtype="complex"),
        gold("q3", "q3?", ["AG0205 - Ars Magica"], "page 7", qtype="complex"),
    ]
    client = FakeLLMClient(LLMResult(text=judge_json(12, 6, 6)))

    judgments = judge_answers(path, questions, judge_client=client)

    assert set(judgments) == {"q1", "q3"}
    assert judgments["q1"].total == 24
    assert len(client.calls) == 2  # the errored question is never judged
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["results"][0]["judgment"]["total"] == 24
    assert "judgment" not in saved["results"][1]
    assert saved["results"][2]["judgment"]["correctness"] == 12


def test_judge_answers_stops_when_keys_are_exhausted(tmp_path: Path) -> None:
    path = artifact_file(tmp_path)
    questions = [
        gold("q1", "q1?", ["AG0205 - Ars Magica"], "page 7"),
        gold("q3", "q3?", ["AG0205 - Ars Magica"], "page 7", qtype="complex"),
    ]

    class ExhaustAfterFirst:
        model = "spent"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            if self.calls > 1:
                raise AllKeysExhausted("quota")
            return LLMResult(text=judge_json(3, 3, 3))

    judgments = judge_answers(path, questions, judge_client=ExhaustAfterFirst())

    assert set(judgments) == {"q1"}
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["meta"]["judge_aborted"].startswith("AllKeysExhausted:")
    assert saved["results"][0]["judgment"]["total"] == 9
    assert "judgment" not in saved["results"][2]


def test_judge_answers_accepts_an_in_memory_artifact(tmp_path: Path) -> None:
    artifact = {"meta": {}, "results": [{"id": "q1", "type": "simple", "answer": "a"}]}
    judge_answers(
        artifact,
        [gold("q1", "q1?", ["AG0205 - Ars Magica"], "page 7")],
        judge_client=FakeLLMClient(LLMResult(text=judge_json(1, 2, 3))),
    )
    assert artifact["results"][0]["judgment"]["total"] == 6


# ---------------------------------------------------------------- summarize


def judged_artifact() -> dict:
    def result(qid: str, qtype: str, total_parts: tuple[int, int, int], **extra) -> dict:
        correctness, completeness, groundedness = total_parts
        return {
            "id": qid,
            "type": qtype,
            "question": f"{qid}?",
            "answer": "text",
            "no_answer": extra.pop("no_answer", False),
            "judgment": {
                "correctness": correctness,
                "completeness": completeness,
                "groundedness": groundedness,
                "total": correctness + completeness + groundedness,
                "rationale": "",
                "fallback": extra.pop("fallback", False),
            },
            **extra,
        }

    return {
        "meta": {"kb": "bench"},
        "results": [
            result("q1", "simple", (10, 8, 2)),  # total 20
            result("q2", "simple", (12, 10, 4)),  # total 26
            result("q3", "complex", (4, 4, 2), no_answer=True, fallback=True),  # total 10
            {"id": "q4", "type": "complex", "question": "q4?", "error": "RuntimeError: x"},
        ],
    }


def test_summarize_means_and_sem(tmp_path: Path) -> None:
    path = tmp_path / "judged.json"
    path.write_text(json.dumps(judged_artifact()), encoding="utf-8")

    rows = {row["subset"]: row for row in summarize(path)}
    assert set(rows) == {"simple", "complex", "overall"}

    simple = rows["simple"]
    assert simple["n"] == 2 and simple["judged"] == 2
    # totals 20 and 26: mean 23, stdev sqrt(18) = 3*sqrt(2), SEM = 3*sqrt(2)/sqrt(2) = 3.
    assert simple["mean_total"] == pytest.approx(23.0)
    assert simple["sem_total"] == pytest.approx(3.0)
    assert simple["mean_correctness"] == pytest.approx(11.0)
    assert simple["mean_completeness"] == pytest.approx(9.0)
    assert simple["mean_groundedness"] == pytest.approx(3.0)
    assert simple["no_answer"] == 0 and simple["fallback"] == 0 and simple["errors"] == 0

    complex_row = rows["complex"]
    assert complex_row["n"] == 2  # the errored question still counts as asked
    assert complex_row["judged"] == 1
    assert complex_row["mean_total"] == pytest.approx(10.0)
    assert complex_row["sem_total"] == 0.0  # one sample has no spread
    assert complex_row["no_answer"] == 1 and complex_row["fallback"] == 1
    assert complex_row["errors"] == 1

    overall = rows["overall"]
    assert overall["n"] == 4 and overall["judged"] == 3
    # totals 20, 26, 10: mean 56/3, stdev 14/sqrt(3), SEM = 14/3.
    assert overall["mean_total"] == pytest.approx(56 / 3)
    assert overall["sem_total"] == pytest.approx(14 / 3)


def test_summarize_unjudged_run_reports_no_means() -> None:
    rows = summarize({"meta": {}, "results": [{"id": "q1", "type": "simple", "answer": "a"}]})
    assert rows[0]["judged"] == 0
    assert rows[0]["mean_total"] is None
    assert rows[0]["sem_total"] is None
