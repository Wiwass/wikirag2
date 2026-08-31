"""Runner tests: discovery, stage sequencing, resume, caps, quota stop.

Fully offline: stages are fakes recording (doc_id, stage) calls, and the
"PDFs" are empty files in the KB's raw/ inbox.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from wikirag.config import KnowledgeBase, SynthesisLevel, create_kb
from wikirag.ingestion.runner import (
    BASE_STAGES,
    STAGES,
    RunReport,
    discover_inbox,
    doc_id_for,
    enqueue_missing,
    register_stage,
    run_pending,
    stage_sequence,
)
from wikirag.llm.client import AllKeysExhausted


@pytest.fixture()
def kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterable[KnowledgeBase]:
    monkeypatch.setenv("WIKIRAG_HOME", str(tmp_path / "kbs"))
    with create_kb("test") as handle:
        yield handle


def add_pdf(kb: KnowledgeBase, name: str) -> Path:
    path = kb.raw_dir / name
    path.write_bytes(b"")
    return path


class FakeStages:
    """Stage registry recording every call, with per-stage failure hooks."""

    def __init__(self, stages: Iterable[str]) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail: dict[tuple[str, str], Exception] = {}
        self.mapping: dict[str, Callable[[KnowledgeBase, str], None]] = {
            stage: self._make(stage) for stage in stages
        }

    def _make(self, stage: str) -> Callable[[KnowledgeBase, str], None]:
        def run(kb: KnowledgeBase, doc_id: str) -> None:
            self.calls.append((doc_id, stage))
            error = self.fail.get((doc_id, stage))
            if error is not None:
                raise error

        return run


@pytest.fixture()
def fakes(kb: KnowledgeBase) -> FakeStages:
    return FakeStages(stage_sequence(SynthesisLevel.S3))


# ------------------------------------------------------------- sequence + ids


def test_stage_sequence_per_level() -> None:
    assert stage_sequence(SynthesisLevel.S0) == list(BASE_STAGES)
    assert stage_sequence("S1") == [*BASE_STAGES, "summarize"]
    assert stage_sequence(SynthesisLevel.S2) == [*BASE_STAGES, "summarize", "enrich"]
    assert stage_sequence(SynthesisLevel.S3) == stage_sequence(SynthesisLevel.S2)


def test_doc_id_slugging() -> None:
    assert doc_id_for("Ars Magica 5th Ed.pdf") == "ars-magica-5th-ed"
    assert doc_id_for("/lib/Houses_of_Hermes--True Lineages.PDF") == (
        "houses-of-hermes-true-lineages"
    )
    assert doc_id_for("--weird--.pdf") == "weird"
    assert doc_id_for("Covenants (2004).pdf") == "covenants-2004"


def test_stage_registry_is_empty_until_wired() -> None:
    assert STAGES == {}
    register_stage("extract", lambda kb, doc_id: None)
    try:
        assert set(STAGES) == {"extract"}
    finally:
        STAGES.clear()


# ------------------------------------------------------------------ discovery


def test_discover_inbox_enqueues_sequence_for_level(kb: KnowledgeBase) -> None:
    add_pdf(kb, "Alpha Book.pdf")
    add_pdf(kb, "beta.pdf")
    (kb.raw_dir / "notes.txt").write_text("ignored")

    assert discover_inbox(kb) == ["alpha-book", "beta"]

    document = kb.store.get_document("alpha-book")
    assert document is not None
    assert document["filename"] == "Alpha Book.pdf"
    assert document["status"] == "pending"
    stages = [job["stage"] for job in kb.store.pending_jobs("alpha-book")]
    assert stages == list(BASE_STAGES)
    assert kb.store.get_document("notes") is None

    # Re-running discovery is a no-op: nothing new, no duplicate jobs.
    assert discover_inbox(kb) == []
    assert len(kb.store.pending_jobs("alpha-book")) == len(BASE_STAGES)


def test_discover_inbox_at_s2(kb: KnowledgeBase) -> None:
    kb.synthesis_level = SynthesisLevel.S2
    add_pdf(kb, "tome.pdf")
    assert discover_inbox(kb) == ["tome"]
    assert [job["stage"] for job in kb.store.pending_jobs("tome")] == [
        *BASE_STAGES,
        "summarize",
        "enrich",
    ]


def test_discover_inbox_slug_collision_skips_second(
    kb: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    add_pdf(kb, "Grand Tribunal.pdf")
    add_pdf(kb, "grand_tribunal.pdf")
    with caplog.at_level(logging.WARNING):
        added = discover_inbox(kb)

    assert added == ["grand-tribunal"]
    document = kb.store.get_document("grand-tribunal")
    assert document is not None
    assert document["filename"] == "Grand Tribunal.pdf"  # not overwritten
    assert "grand_tribunal.pdf" in caplog.text


# ----------------------------------------------------------------- happy path


def test_run_pending_runs_sequence_once_then_archives(
    kb: KnowledgeBase, fakes: FakeStages
) -> None:
    add_pdf(kb, "Alpha Book.pdf")
    discover_inbox(kb)

    report = run_pending(kb, stages=fakes.mapping)

    assert fakes.calls == [("alpha-book", stage) for stage in BASE_STAGES]
    assert report.done == {stage: 1 for stage in BASE_STAGES}
    assert report.failed == {} and report.errors == []
    assert report.stopped_early is False
    assert report.docs_completed == ["alpha-book"]

    document = kb.store.get_document("alpha-book")
    assert document is not None and document["status"] == "ingested"
    assert not (kb.raw_dir / "Alpha Book.pdf").exists()
    assert (kb.archive_dir / "Alpha Book.pdf").is_file()

    # Idempotent: a second run finds nothing to do.
    again = run_pending(kb, stages=fakes.mapping)
    assert len(fakes.calls) == len(BASE_STAGES)
    assert again.done == {} and again.docs_completed == []


def test_until_stage_stops_the_sequence(kb: KnowledgeBase, fakes: FakeStages) -> None:
    add_pdf(kb, "alpha.pdf")
    discover_inbox(kb)

    report = run_pending(kb, until_stage="chunk", stages=fakes.mapping)

    assert fakes.calls == [("alpha", "extract"), ("alpha", "chunk")]
    assert report.docs_completed == []
    document = kb.store.get_document("alpha")
    assert document is not None and document["status"] == "pending"
    assert (kb.raw_dir / "alpha.pdf").is_file()

    # The rest of the sequence is still pending and runs on the next call.
    report = run_pending(kb, stages=fakes.mapping)
    assert fakes.calls[2:] == [("alpha", "xrefs"), ("alpha", "index")]
    assert report.docs_completed == ["alpha"]

    with pytest.raises(ValueError):
        run_pending(kb, until_stage="summarize", stages=fakes.mapping)


# ------------------------------------------------------------------- failures


def test_failed_stage_blocks_its_document_only(
    kb: KnowledgeBase, fakes: FakeStages
) -> None:
    add_pdf(kb, "alpha.pdf")
    add_pdf(kb, "beta.pdf")
    discover_inbox(kb)
    fakes.fail[("alpha", "chunk")] = RuntimeError("bad pdf")

    report = run_pending(kb, stages=fakes.mapping)

    assert ("alpha", "xrefs") not in fakes.calls
    assert [c for c in fakes.calls if c[0] == "beta"] == [
        ("beta", stage) for stage in BASE_STAGES
    ]
    assert report.failed == {"chunk": 1}
    assert report.skipped == {"xrefs": 1, "index": 1}
    assert report.errors == [("alpha", "chunk", "bad pdf")]
    assert report.docs_completed == ["beta"]

    job = kb.store.get_job("alpha", "chunk")
    assert job is not None
    assert job["status"] == "failed" and job["attempts"] == 1
    assert job["last_error"] == "bad pdf"
    document = kb.store.get_document("alpha")
    assert document is not None and document["status"] == "pending"


def test_resume_reruns_only_unfinished_jobs(
    kb: KnowledgeBase, fakes: FakeStages
) -> None:
    add_pdf(kb, "alpha.pdf")
    discover_inbox(kb)
    fakes.fail[("alpha", "chunk")] = RuntimeError("transient")
    run_pending(kb, stages=fakes.mapping)

    fakes.calls.clear()
    fakes.fail.clear()
    report = run_pending(kb, stages=fakes.mapping)

    assert fakes.calls == [("alpha", "chunk"), ("alpha", "xrefs"), ("alpha", "index")]
    assert report.done == {"chunk": 1, "xrefs": 1, "index": 1}
    assert report.skipped == {"extract": 1}  # already done, not re-run
    assert report.docs_completed == ["alpha"]
    assert (kb.archive_dir / "alpha.pdf").is_file()


def test_attempts_cap_reports_instead_of_retrying(
    kb: KnowledgeBase, fakes: FakeStages
) -> None:
    add_pdf(kb, "alpha.pdf")
    discover_inbox(kb)
    fakes.fail[("alpha", "extract")] = RuntimeError("boom")

    for _ in range(2):
        run_pending(kb, stages=fakes.mapping, max_attempts=2)
    assert fakes.calls == [("alpha", "extract")] * 2

    report = run_pending(kb, stages=fakes.mapping, max_attempts=2)

    assert fakes.calls == [("alpha", "extract")] * 2  # not retried a third time
    assert report.done == {} and report.failed == {}
    assert report.skipped["extract"] == 1
    assert len(report.errors) == 1
    doc_id, stage, message = report.errors[0]
    assert (doc_id, stage) == ("alpha", "extract")
    assert "out of retries" in message and "boom" in message


def test_missing_implementation_is_reported_without_spending_an_attempt(
    kb: KnowledgeBase, fakes: FakeStages
) -> None:
    add_pdf(kb, "alpha.pdf")
    discover_inbox(kb)
    partial = {"extract": fakes.mapping["extract"]}

    report = run_pending(kb, stages=partial)

    assert report.done == {"extract": 1}
    assert report.errors == [("alpha", "chunk", "no implementation registered")]
    job = kb.store.get_job("alpha", "chunk")
    assert job is not None and job["attempts"] == 0 and job["status"] == "pending"


def test_all_keys_exhausted_stops_the_batch(
    kb: KnowledgeBase, fakes: FakeStages
) -> None:
    add_pdf(kb, "alpha.pdf")
    add_pdf(kb, "beta.pdf")
    discover_inbox(kb)
    fakes.fail[("alpha", "xrefs")] = AllKeysExhausted("quota gone")

    report = run_pending(kb, doc_ids=["alpha", "beta"], stages=fakes.mapping)

    assert report.stopped_early is True
    assert fakes.calls == [("alpha", "extract"), ("alpha", "chunk"), ("alpha", "xrefs")]
    assert report.failed == {"xrefs": 1}
    assert report.errors == [("alpha", "xrefs", "quota gone")]
    assert report.docs_completed == []

    # beta was never touched: its jobs are untouched and rerunnable.
    beta_jobs = kb.store.pending_jobs("beta")
    assert [job["stage"] for job in beta_jobs] == list(BASE_STAGES)
    assert all(job["attempts"] == 0 for job in beta_jobs)

    # And the next run picks both up where they stopped.
    fakes.fail.clear()
    fakes.calls.clear()
    resumed = run_pending(kb, stages=fakes.mapping)
    assert fakes.calls == [
        ("alpha", "xrefs"),
        ("alpha", "index"),
        *[("beta", stage) for stage in BASE_STAGES],
    ]
    assert sorted(resumed.docs_completed) == ["alpha", "beta"]


# ------------------------------------------------------------ raising S later


def test_enqueue_missing_after_raising_synthesis_level(
    kb: KnowledgeBase, fakes: FakeStages
) -> None:
    add_pdf(kb, "alpha.pdf")
    discover_inbox(kb)
    run_pending(kb, stages=fakes.mapping)
    fakes.calls.clear()

    kb.synthesis_level = SynthesisLevel.S2
    assert enqueue_missing(kb) == 2  # summarize + enrich, nothing re-enqueued
    assert enqueue_missing(kb) == 0  # already queued, not duplicated

    report = run_pending(kb, stages=fakes.mapping)

    assert fakes.calls == [("alpha", "summarize"), ("alpha", "enrich")]
    assert report.done == {"summarize": 1, "enrich": 1}
    assert report.skipped == {stage: 1 for stage in BASE_STAGES}
    assert report.docs_completed == ["alpha"]
    document = kb.store.get_document("alpha")
    assert document is not None and document["status"] == "ingested"


def test_enqueue_missing_ignores_documents_still_ingesting(
    kb: KnowledgeBase, fakes: FakeStages
) -> None:
    add_pdf(kb, "alpha.pdf")
    discover_inbox(kb)
    run_pending(kb, until_stage="chunk", stages=fakes.mapping)

    kb.synthesis_level = SynthesisLevel.S1
    assert enqueue_missing(kb) == 0
    assert kb.store.get_job("alpha", "summarize") is None


# ----------------------------------------------------------------- edge cases


def test_unknown_document_is_reported(kb: KnowledgeBase, fakes: FakeStages) -> None:
    report = run_pending(kb, doc_ids=["ghost"], stages=fakes.mapping)
    assert report.errors == [("ghost", "*", "unknown document")]
    assert fakes.calls == []


def test_completion_repairs_an_interrupted_archive(
    kb: KnowledgeBase, fakes: FakeStages
) -> None:
    """All jobs done but the side-effect missed (crash): the next run fixes it."""
    add_pdf(kb, "alpha.pdf")
    discover_inbox(kb)
    for stage in BASE_STAGES:
        job = kb.store.get_job("alpha", stage)
        assert job is not None
        kb.store.finish_job(int(job["id"]), status="done")

    report = run_pending(kb, stages=fakes.mapping)

    assert fakes.calls == []
    assert report.docs_completed == ["alpha"]
    assert (kb.archive_dir / "alpha.pdf").is_file()
    document = kb.store.get_document("alpha")
    assert document is not None and document["status"] == "ingested"


def test_report_defaults() -> None:
    report = RunReport()
    assert report.done == {} and report.failed == {} and report.skipped == {}
    assert report.errors == [] and report.docs_completed == []
    assert report.stopped_early is False
