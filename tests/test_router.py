"""Tests for wikirag.retrieval.router — offline: FakeLLMClient, canned JSON.

No network: every classification comes from a scripted LLMResult, and the
normalization/routing helpers are pure.
"""

from __future__ import annotations

import json

import pytest

from wikirag.config import RetrievalDepth
from wikirag.llm.client import AllKeysExhausted, FakeLLMClient, LLMResult
from wikirag.retrieval.router import (
    CLASSIFIER_HISTORY_TURNS,
    CLASSIFIER_RESPONSE_SCHEMA,
    CLASSIFIER_SYSTEM_PROMPT,
    MAX_SUBQUESTIONS,
    Classification,
    _validate_classification,
    anchors,
    classify,
    route_depth,
)


def payload(
    is_complex: bool = False,
    subquestions: list | None = None,
    guidance: str = "",
) -> str:
    return json.dumps(
        {
            "is_complex": is_complex,
            "subquestions": subquestions if subquestions is not None else [],
            "retrieval_guidance": guidance,
        }
    )


class BoomClient:
    """A provider that fails the way a 500 does."""

    model = "boom"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls: list[dict] = []

    def generate(self, **kwargs) -> LLMResult:
        self.calls.append(kwargs)
        raise self.exc


# ------------------------------------------------------------------ classify


def test_classify_simple_question() -> None:
    client = FakeLLMClient(LLMResult(text=payload(False)))
    result = classify(client, "How many botch dice?")

    assert result == Classification(is_complex=False, subquestions=[], retrieval_guidance="")
    assert result.fallback is False
    assert len(client.calls) == 1


def test_classify_complex_question_keeps_subquestions_and_guidance() -> None:
    client = FakeLLMClient(
        LLMResult(
            text=payload(
                True,
                ["What is a covenant?", "What is a magus?"],
                "the previous turn only covered covenants",
            )
        )
    )
    result = classify(client, "Compare covenants and magi.")

    assert result.is_complex is True
    assert result.subquestions == ["What is a covenant?", "What is a magus?"]
    assert result.retrieval_guidance == "the previous turn only covered covenants"
    assert result.fallback is False


def test_classify_passes_system_prompt_and_json_schema() -> None:
    client = FakeLLMClient(LLMResult(text=payload(False)))
    classify(client, "a question")

    call = client.calls[0]
    assert call["json_schema"] is CLASSIFIER_RESPONSE_SCHEMA
    assert call["system"] == CLASSIFIER_SYSTEM_PROMPT
    assert "Current question: a question" in call["user"]


def test_classify_strips_markdown_fences() -> None:
    client = FakeLLMClient(LLMResult(text=f"```json\n{payload(True, ['sub'])}\n```"))
    assert classify(client, "q").subquestions == ["sub"]


def test_classify_retries_malformed_json_then_succeeds() -> None:
    client = FakeLLMClient(
        LLMResult(text="not json at all"),
        LLMResult(text=payload(True, ["sub one"])),
    )
    result = classify(client, "q")

    assert len(client.calls) == 2
    assert result.is_complex is True
    assert result.fallback is False


def test_classify_falls_back_after_max_attempts() -> None:
    client = FakeLLMClient(LLMResult(text="still not json"))
    result = classify(client, "q", max_attempts=2)

    assert len(client.calls) == 2
    assert result == Classification(fallback=True)
    assert result.is_complex is False
    assert result.subquestions == []


def test_classify_falls_back_on_a_non_object_payload() -> None:
    client = FakeLLMClient(LLMResult(text="[1, 2, 3]"))
    result = classify(client, "q")

    assert len(client.calls) == 2  # a list is retried like malformed JSON
    assert result.fallback is True


def test_classify_degrades_on_a_provider_error() -> None:
    client = BoomClient(RuntimeError("500 from provider"))
    result = classify(client, "q")

    assert result == Classification(fallback=True)
    assert len(client.calls) == 1  # a provider error is not retried here


def test_classify_propagates_exhausted_keys() -> None:
    client = BoomClient(AllKeysExhausted("quota"))
    with pytest.raises(AllKeysExhausted):
        classify(client, "q")


# ------------------------------------------------------------ prompt assembly


def test_history_is_windowed_to_the_last_turns() -> None:
    history = [(f"question {i}", f"answer {i}") for i in range(8)]
    client = FakeLLMClient(LLMResult(text=payload(False)))
    classify(client, "current", history=history)

    user = client.calls[0]["user"]
    assert "question 2" not in user  # 8 exchanges, only the last 5 survive
    for i in range(3, 8):
        assert f"question {i}" in user
        assert f"answer {i}" in user
    assert user.count("User: ") == CLASSIFIER_HISTORY_TURNS


def test_history_absent_leaves_no_transcript_block() -> None:
    client = FakeLLMClient(LLMResult(text=payload(False)))
    classify(client, "current")

    assert "Recent conversation history" not in client.calls[0]["user"]


def test_overview_text_is_included_when_present_and_noted_when_absent() -> None:
    with_overview = FakeLLMClient(LLMResult(text=payload(False)))
    classify(with_overview, "q", overview_text="The KB covers Ars Magica.")
    assert "Knowledge base overview:" in with_overview.calls[0]["user"]
    assert "The KB covers Ars Magica." in with_overview.calls[0]["user"]

    without = FakeLLMClient(LLMResult(text=payload(False)))
    classify(without, "q")
    assert "(No knowledge base overview is available.)" in without.calls[0]["user"]

    blank = FakeLLMClient(LLMResult(text=payload(False)))
    classify(blank, "q", overview_text="   ")
    assert "(No knowledge base overview is available.)" in blank.calls[0]["user"]


# -------------------------------------------------------------- normalization


def test_validate_caps_subquestions() -> None:
    data = {
        "is_complex": True,
        "subquestions": [f"sub {i}" for i in range(MAX_SUBQUESTIONS + 3)],
        "retrieval_guidance": "",
    }
    assert len(_validate_classification(data).subquestions) == MAX_SUBQUESTIONS


def test_validate_drops_blank_and_non_string_subquestions() -> None:
    data = {
        "is_complex": True,
        "subquestions": ["  keep me  ", "", "   ", None, 42, ["nested"], "keep me too"],
    }
    assert _validate_classification(data).subquestions == ["keep me", "keep me too"]


def test_validate_deduplicates_subquestions() -> None:
    data = {"is_complex": True, "subquestions": ["same", "same ", "other"]}
    assert _validate_classification(data).subquestions == ["same", "other"]


def test_validate_forces_empty_subquestions_when_simple() -> None:
    data = {"is_complex": False, "subquestions": ["a", "b"], "retrieval_guidance": "note"}
    result = _validate_classification(data)

    assert result.is_complex is False
    assert result.subquestions == []
    assert result.retrieval_guidance == "note"  # guidance survives the simple case


def test_validate_coerces_junk_types() -> None:
    result = _validate_classification(
        {"is_complex": 1, "subquestions": "not a list", "retrieval_guidance": 7}
    )
    assert result.is_complex is True
    assert result.subquestions == []
    assert result.retrieval_guidance == ""


def test_validate_of_an_empty_payload_is_the_simple_default() -> None:
    assert _validate_classification({}) == Classification()


def test_classify_simple_answer_with_subquestions_is_normalized() -> None:
    client = FakeLLMClient(LLMResult(text=payload(False, ["stray sub"])))
    assert classify(client, "q").subquestions == []


# --------------------------------------------------------------- route_depth


def test_route_depth_maps_both_ways() -> None:
    assert route_depth(Classification(is_complex=False)) is RetrievalDepth.R1
    assert route_depth(Classification(is_complex=True)) is RetrievalDepth.R2


def test_route_depth_honors_overrides() -> None:
    assert (
        route_depth(
            Classification(is_complex=False),
            default_simple=RetrievalDepth.R0,
            default_complex=RetrievalDepth.R3,
        )
        is RetrievalDepth.R0
    )
    assert (
        route_depth(
            Classification(is_complex=True),
            default_simple=RetrievalDepth.R0,
            default_complex=RetrievalDepth.R3,
        )
        is RetrievalDepth.R3
    )


def test_route_depth_of_a_fallback_is_the_simple_depth() -> None:
    assert route_depth(Classification(fallback=True)) is RetrievalDepth.R1


# -------------------------------------------------------------------- anchors


def test_anchors_of_a_simple_classification_is_the_query_alone() -> None:
    assert anchors("What is a magus?", Classification()) == ["What is a magus?"]


def test_anchors_puts_the_query_first_then_subquestions() -> None:
    classification = Classification(is_complex=True, subquestions=["sub a", "sub b"])
    assert anchors("original", classification) == ["original", "sub a", "sub b"]


def test_anchors_deduplicates_case_insensitively_and_strips() -> None:
    classification = Classification(
        is_complex=True, subquestions=["  Original  ", "sub", "SUB", ""]
    )
    assert anchors("original", classification) == ["original", "sub"]


def test_anchors_of_a_blank_query_keeps_only_the_subquestions() -> None:
    classification = Classification(is_complex=True, subquestions=["sub"])
    assert anchors("   ", classification) == ["sub"]
