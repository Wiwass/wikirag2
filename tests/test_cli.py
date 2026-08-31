"""CLI tests: `main([...])` driven directly, fully offline.

The KB root is a tmp_path via WIKIRAG_HOME. `query --fake` runs the real
retrieval pipeline over a hand-seeded, FTS-only KB (no vector index, so no
embedding model is ever loaded) with only the cross-encoder step replaced,
and the fake LLM client instead of a provider. No network, no downloads,
no PDF ingestion.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from wikirag import answer as answer_module
from wikirag.cli import FAKE_ANSWER_TEXT, main
from wikirag.config import create_kb, open_kb


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "kbs"
    monkeypatch.setenv("WIKIRAG_HOME", str(root))
    return root


@pytest.fixture(autouse=True)
def stage_registry() -> Iterator[None]:
    """`ingest` wires the real stages into a module-level registry; put it back."""
    from wikirag.ingestion.runner import STAGES

    saved = dict(STAGES)
    yield
    STAGES.clear()
    STAGES.update(saved)


@pytest.fixture()
def seeded() -> str:
    """A KB with one document, one section and two chunks. Name: 'kb'."""
    with create_kb("kb") as kb:
        kb.store.add_document(
            doc_id="grimoire", filename="grimoire.pdf", title="Grimoire", status="ingested"
        )
        kb.store.add_sections(
            "grimoire", [{"id": "grimoire:1", "ordinal": 1, "title": "Spells", "level": 1}]
        )
        kb.store.add_chunks(
            [
                {
                    "id": "grimoire:1:0",
                    "doc_id": "grimoire",
                    "section_id": "grimoire:1",
                    "ordinal": 0,
                    "text": "A dragon breathes fire over the tower of the covenant.",
                    "token_count": 12,
                },
                {
                    "id": "grimoire:1:1",
                    "doc_id": "grimoire",
                    "section_id": "grimoire:1",
                    "ordinal": 1,
                    "text": "Binding a dragon requires vis and a prepared circle.",
                    "token_count": 11,
                },
            ]
        )
    return "kb"


@pytest.fixture()
def local_rerank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Score every candidate above the gate, without a cross-encoder model."""

    def fake_rerank(query: str, candidates: Any, **kwargs: Any) -> list[Any]:
        for candidate in candidates:
            candidate.score = 0.9
        return list(candidates)

    monkeypatch.setattr(answer_module, "rerank", fake_rerank)


# ------------------------------------------------------------------ kb command


def test_kb_create_and_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["kb", "create", "mykb"]) == 0
    capsys.readouterr()
    assert main(["kb", "list"]) == 0
    assert "mykb" in capsys.readouterr().out


def test_kb_list_empty_root(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["kb", "list"]) == 0
    assert "no knowledge bases" in capsys.readouterr().out


def test_kb_create_twice_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["kb", "create", "mykb"]) == 0
    assert main(["kb", "create", "mykb"]) == 2
    assert "already exists" in capsys.readouterr().err


def test_kb_info_reports_contents(seeded: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["kb", "info", seeded]) == 0
    out = capsys.readouterr().out
    assert "documents:      1 (1 ingested)" in out
    assert "chunks:         2" in out
    assert "synthesis:      S0" in out
    assert "default_depth = R1" in out


def test_unknown_kb_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["kb", "info", "nope"]) == 2
    assert "no knowledge base 'nope'" in capsys.readouterr().err


# --------------------------------------------------------------------- queries


def test_query_fake_end_to_end(
    seeded: str, local_rerank: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["query", seeded, "how do you bind a dragon?", "--fake"]) == 0
    out = capsys.readouterr().out
    assert FAKE_ANSWER_TEXT in out
    assert "Sources:" in out
    assert "[1] Grimoire — Spells" in out
    assert "R1" in out and "tok" in out

    with open_kb(seeded) as kb:
        logged = kb.store.get_queries()
        assert [row["query"] for row in logged] == ["how do you bind a dragon?"]
        assert logged[0]["no_answer"] is False


def test_query_depth_and_budget_are_honored(
    seeded: str, local_rerank: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["query", seeded, "dragon", "--fake", "--depth", "R0", "--budget", "500"]) == 0
    assert "[R0 |" in capsys.readouterr().out


def test_query_with_no_match_takes_the_no_answer_path(
    seeded: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # Nothing matches lexically and there is no vector index, so the pool is
    # empty and the gate fires before any model would be needed.
    assert main(["query", seeded, "zzzqqq unrelated", "--fake"]) == 0
    out = capsys.readouterr().out
    assert "could not find anything relevant" in out
    assert "Sources:" not in out

    with open_kb(seeded) as kb:
        assert kb.store.get_queries()[0]["no_answer"] is True
        assert kb.store.get_llm_calls() == []


def test_query_r3_exits_2(seeded: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["query", seeded, "dragon", "--fake", "--depth", "R3"]) == 2
    assert "Wave 5" in capsys.readouterr().err


def test_query_unknown_kb_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["query", "nope", "dragon", "--fake"]) == 2
    assert "no knowledge base" in capsys.readouterr().err


# ----------------------------------------------------------------------- costs


def test_costs_after_a_fake_query(
    seeded: str, local_rerank: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["query", seeded, "dragon", "--fake"]) == 0
    capsys.readouterr()

    assert main(["costs", seeded]) == 0
    out = capsys.readouterr().out
    assert "fake-model" in out
    assert "query" in out and "answer" in out
    assert "total: $0.0000 over 1 call(s)" in out


def test_costs_grouping_and_empty_kb(
    seeded: str, local_rerank: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["costs", seeded]) == 0
    assert "no LLM calls recorded" in capsys.readouterr().out

    assert main(["query", seeded, "dragon", "--fake"]) == 0
    capsys.readouterr()
    assert main(["costs", seeded, "--by", "role"]) == 0
    out = capsys.readouterr().out
    assert "role" in out and "fake-model" not in out


def test_costs_rejects_unknown_grouping(seeded: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["costs", seeded, "--by", "wizard"]) == 2
    assert "cannot group by" in capsys.readouterr().err


# ------------------------------------------------------- level, forget, ingest


def test_set_level(seeded: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["set-level", seeded, "S1"]) == 0
    out = capsys.readouterr().out
    assert "synthesis level S1" in out
    # The seeded document is marked ingested but was never run through the
    # stages, so every stage of the S1 sequence is owed.
    assert "5 job(s) added" in out
    with open_kb(seeded) as kb:
        assert kb.synthesis_level.value == "S1"


def test_set_level_rejects_garbage(seeded: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["set-level", seeded, "S9"]) == 2
    assert "unknown SynthesisLevel" in capsys.readouterr().err


def test_forget(seeded: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["forget", seeded, "grimoire"]) == 0
    assert main(["forget", seeded, "grimoire"]) == 1
    assert "no document 'grimoire'" in capsys.readouterr().err
    with open_kb(seeded) as kb:
        assert kb.store.list_documents() == []


def test_ingest_skips_non_pdf_inputs(
    seeded: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("not a pdf")

    assert main(["ingest", seeded, str(notes), str(tmp_path / "missing.pdf")]) == 1
    captured = capsys.readouterr()
    assert "notes.txt: not a PDF, skipped" in captured.err
    assert "missing.pdf: no such file, skipped" in captured.err
    assert "inbox: 0 new document(s)" in captured.out
    with open_kb(seeded) as kb:
        assert list(kb.raw_dir.iterdir()) == []


def test_ingest_empty_inbox_is_a_no_op(seeded: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ingest", seeded]) == 0
    assert "inbox: 0 new document(s)" in capsys.readouterr().out
