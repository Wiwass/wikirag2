"""Job runner: executes ingestion stages, resumably.

Responsibilities
----------------
- For each document, walk the stage sequence up to the KB's synthesis
  level; skip completed jobs (jobs table), retry failed ones with capped
  attempts, record errors per job.
- A batch interrupted anywhere (crash, quota exhaustion) resumes exactly
  where it stopped on the next run — this replaces v1's pending-keyword
  queue, phase-A cache directory, and overview_parts cache with one
  mechanism.
- Raising a KB's synthesis level enqueues only the missing stages for
  already-ingested documents.
- Quota exhaustion (AllKeysExhausted from the LLM layer) stops the batch
  cleanly rather than grinding through doomed retries (v1 lesson).
- Local CPU work (embedding) runs as ordinary jobs interleaved with
  API-bound jobs, not as ad-hoc threads sharing mutable stores.

Stage sequence (docs/ingestion.md)
----------------------------------
S0 runs extract → chunk → xrefs → index; S1 adds summarize; S2 and S3 add
enrich. Stages of one document run in that order and a failed stage blocks
that document's later stages for the rest of the run — never other
documents.

A stage implementation is `Callable[[KnowledgeBase, str], None]` taking
(kb, doc_id) and raising on failure. They are wired into `STAGES` by the
orchestrator at integration time via `register_stage`; this module never
imports them, which is what keeps the runner testable offline.

Completion side-effect
----------------------
When every stage of the KB's sequence has a `done` job for a document, the
runner marks the document `ingested` and moves its PDF from `raw/` (the
inbox) to `archive/` (docs/storage.md). Both steps are idempotent: an
already-archived file or an already-ingested status is not an error, so a
crash between them is repaired by the next run.

Public API
----------
- StageFn: Callable[[KnowledgeBase, str], None]
- BASE_STAGES, STAGES, register_stage(name, fn)
- stage_sequence(level) -> list[str]
- doc_id_for(filename) -> str
- discover_inbox(kb) -> list[str]
- run_pending(kb, doc_ids=None, until_stage=None, stages=None,
  max_attempts=3) -> RunReport
- enqueue_missing(kb) -> int
- class RunReport: done, failed, skipped (per-stage counts), errors,
  stopped_early, docs_completed
"""

from __future__ import annotations

import logging
import re
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from wikirag.config import KnowledgeBase, SynthesisLevel
from wikirag.llm.client import AllKeysExhausted

log = logging.getLogger(__name__)

StageFn = Callable[[KnowledgeBase, str], None]

# docs/ingestion.md: the S0 floor, all local and LLM-free.
BASE_STAGES: tuple[str, ...] = ("extract", "chunk", "xrefs", "index")
_LEVEL_EXTRAS: dict[SynthesisLevel, tuple[str, ...]] = {
    SynthesisLevel.S0: (),
    SynthesisLevel.S1: ("summarize",),
    SynthesisLevel.S2: ("summarize", "enrich"),
    SynthesisLevel.S3: ("summarize", "enrich"),
}

# Wired by the orchestrator at integration; empty here on purpose.
STAGES: dict[str, StageFn] = {}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def register_stage(name: str, fn: StageFn) -> None:
    """Bind a stage name to its implementation in the module-level registry."""
    STAGES[name] = fn


def stage_sequence(level: SynthesisLevel | str) -> list[str]:
    """The stages a document owes at this synthesis level, in run order."""
    return [*BASE_STAGES, *_LEVEL_EXTRAS[SynthesisLevel.parse(level)]]


def doc_id_for(filename: str) -> str:
    """Slugified filename stem: lowercase, non-alphanumeric runs to '-'."""
    return _NON_ALNUM.sub("-", Path(filename).stem.lower()).strip("-")


@dataclass
class RunReport:
    """What one `run_pending` call did, per stage and per document.

    `done`/`failed` count stage executions of this run; `skipped` counts
    stages that were not executed — already done, blocked by an earlier
    failure in the same document, out of retries, or with no implementation
    registered. `errors` holds (doc_id, stage, error) for everything worth
    a human's attention, including the jobs that were skipped because they
    hit the attempts cap.
    """

    done: Counter[str] = field(default_factory=Counter)
    failed: Counter[str] = field(default_factory=Counter)
    skipped: Counter[str] = field(default_factory=Counter)
    errors: list[tuple[str, str, str]] = field(default_factory=list)
    stopped_early: bool = False
    docs_completed: list[str] = field(default_factory=list)


def discover_inbox(kb: KnowledgeBase) -> list[str]:
    """Register every new PDF in `raw/` and enqueue its stages.

    Returns the doc_ids added. Files whose slug is already taken (a second
    file slugging onto an existing document) are skipped and logged rather
    than overwriting the registered document.
    """
    sequence = stage_sequence(kb.synthesis_level)
    added: list[str] = []
    for path in sorted(kb.raw_dir.glob("*.pdf")):
        doc_id = doc_id_for(path.name)
        existing = kb.store.get_document(doc_id)
        if existing is not None:
            if existing["filename"] != path.name:
                log.warning(
                    "skipping %s: doc_id %r already taken by %s",
                    path.name,
                    doc_id,
                    existing["filename"],
                )
            continue
        kb.store.add_document(doc_id=doc_id, filename=path.name, status="pending")
        kb.store.enqueue_jobs(doc_id, sequence)
        added.append(doc_id)
    return added


def enqueue_missing(kb: KnowledgeBase) -> int:
    """Enqueue the stages ingested documents still owe at the current level.

    Raising a KB's synthesis level adds stages to the sequence; only those
    without a `done` job are enqueued, so nothing already paid for is
    redone. Returns the number of job rows created.
    """
    sequence = stage_sequence(kb.synthesis_level)
    created = 0
    for doc in kb.store.list_documents(status="ingested"):
        doc_id = str(doc["id"])
        jobs = {stage: kb.store.get_job(doc_id, stage) for stage in sequence}
        missing = [
            stage
            for stage, job in jobs.items()
            if job is None or job["status"] != "done"
        ]
        if not missing:
            continue
        kb.store.enqueue_jobs(doc_id, missing)
        created += sum(1 for stage in missing if jobs[stage] is None)
    return created


def run_pending(
    kb: KnowledgeBase,
    doc_ids: Sequence[str] | None = None,
    until_stage: str | None = None,
    stages: Mapping[str, StageFn] | None = None,
    max_attempts: int = 3,
) -> RunReport:
    """Run the pending stages of the given documents (default: all of them).

    Documents are processed one at a time, stages in sequence order. A stage
    that raises fails its job and blocks the rest of that document's
    sequence for this run; other documents still run. `AllKeysExhausted`
    stops the whole batch (`stopped_early`), leaving every remaining job
    untouched for the next run.
    """
    sequence = stage_sequence(kb.synthesis_level)
    if until_stage is not None:
        if until_stage not in sequence:
            raise ValueError(
                f"unknown stage {until_stage!r} at level {kb.synthesis_level.value}: "
                f"expected one of {', '.join(sequence)}"
            )
        run_stages = sequence[: sequence.index(until_stage) + 1]
    else:
        run_stages = list(sequence)

    registry: Mapping[str, StageFn] = STAGES if stages is None else stages
    report = RunReport()

    for doc_id in _target_docs(kb, doc_ids, max_attempts):
        document = kb.store.get_document(doc_id)
        if document is None:
            report.errors.append((doc_id, "*", "unknown document"))
            continue
        blocked = False
        ran_any = False
        for stage in run_stages:
            job = kb.store.get_job(doc_id, stage)
            if job is None or job["status"] == "done" or blocked:
                report.skipped[stage] += 1
                continue
            if job["attempts"] >= max_attempts:
                report.skipped[stage] += 1
                report.errors.append(
                    (
                        doc_id,
                        stage,
                        f"out of retries after {job['attempts']} attempts: "
                        f"{job['last_error'] or 'no error recorded'}",
                    )
                )
                blocked = True
                continue
            fn = registry.get(stage)
            if fn is None:
                # A wiring gap, not a document failure: no attempt is spent.
                report.skipped[stage] += 1
                report.errors.append((doc_id, stage, "no implementation registered"))
                blocked = True
                continue
            kb.store.start_job(int(job["id"]))
            try:
                fn(kb, doc_id)
            except AllKeysExhausted as exc:
                kb.store.finish_job(int(job["id"]), status="failed", error=str(exc))
                report.failed[stage] += 1
                report.errors.append((doc_id, stage, str(exc)))
                report.stopped_early = True
                return report
            except Exception as exc:
                kb.store.finish_job(int(job["id"]), status="failed", error=str(exc))
                report.failed[stage] += 1
                report.errors.append((doc_id, stage, str(exc)))
                blocked = True
                continue
            kb.store.finish_job(int(job["id"]), status="done")
            report.done[stage] += 1
            ran_any = True
        _complete_if_done(kb, doc_id, str(document["status"]), sequence, ran_any, report)
    return report


def _target_docs(
    kb: KnowledgeBase, doc_ids: Sequence[str] | None, max_attempts: int
) -> list[str]:
    """Documents to visit: the caller's, else every unfinished one.

    Documents with runnable jobs come first, in job order. Documents that
    are not yet `ingested` but have no runnable job are visited too: they
    are either out of retries (which the report must say) or finished but
    interrupted before the completion side-effect (which this repairs).
    """
    if doc_ids is not None:
        return list(doc_ids)
    seen: dict[str, None] = {}
    for job in kb.store.pending_jobs(max_attempts=max_attempts):
        seen.setdefault(str(job["doc_id"]), None)
    for document in kb.store.list_documents():
        if document["status"] != "ingested":
            seen.setdefault(str(document["id"]), None)
    return list(seen)


def _complete_if_done(
    kb: KnowledgeBase,
    doc_id: str,
    previous_status: str,
    sequence: Sequence[str],
    ran_any: bool,
    report: RunReport,
) -> None:
    """Mark a fully-processed document ingested and archive its source file."""
    for stage in sequence:
        job = kb.store.get_job(doc_id, stage)
        if job is None or job["status"] != "done":
            return
    if previous_status != "ingested":
        kb.store.set_document_status(doc_id, "ingested")
    document = kb.store.get_document(doc_id)
    assert document is not None  # checked by the caller before any stage ran
    source = kb.raw_dir / str(document["filename"])
    if source.is_file():
        kb.archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(kb.archive_dir / source.name))
    if ran_any or previous_status != "ingested":
        report.docs_completed.append(doc_id)
