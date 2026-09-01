"""Consolidation policy: what gets synthesized, when (D11).

The S slider's engine. Sources of demand:
- lazy: consolidation_queue rows pushed by retrieval (R2 verify stage)
  or by an explicit vault request;
- background: hot concepts (query_log frequency) and central ones
  (incoming note_links, chunk_keywords counts), consolidated proactively
  at S2 (hot set only) / S3 (everything central);
- staleness: notes whose source set changed since last synthesis.

Default for first-touch latency: DEFER — the triggering query proceeds
with chunks, the note is built in background and is ready next time.
Config flag for inline synthesis if the user prefers.

Runs under the single-writer lock as ordinary jobs; every synthesis call
is metered (llm_calls) so consolidation cost is always visible.

Budget
------
`ConsolidationBudget` counts notes, not tokens: one note is the unit of
work the caller can reason about, and its token cost is bounded by
`notes.select_sources` anyway. `max_notes` caps a run; `max_failures`
stops a run that is failing systematically (a bad prompt, a dead model)
instead of burning the whole queue on it. Quota exhaustion always stops
immediately and reports what was done — the ingestion runner's discipline,
for the same reason: doomed retries cost real money and real time.

Demand ranking
--------------
Queued keywords come first (someone actually asked for them), oldest
first, because a lazily-triggered note is a promise made to a query that
already happened. Background demand is ranked by `hot` (how often the
keyword was hit in query_log) plus `central` (how many chunks carry the
tag, and how many notes link to it), so the vault grows around what the
user actually asks (D11) rather than around what happens to be big.

Public API
----------
- class ConsolidationBudget: max_notes, max_failures
- class Demand: keyword_id, name, reason, score, queued
- class ConsolidationReport: built, refreshed, failed, skipped, errors,
  stopped_early, keyword_ids
- demand(kb, *, limit=50, include_background=True) -> list[Demand]
- consolidate_pending(kb, *, client, budget=..., embedder=None) -> Report
- refresh_stale(kb, *, client, limit=..., embedder=None) -> Report
- run_consolidation(kb, client, budget=..., embedder=None) -> Report
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Sequence

from wikirag.config import KnowledgeBase
from wikirag.llm.client import AllKeysExhausted, LLMClient
from wikirag.llm.embeddings import EmbedderLike
from wikirag.synthesis.notes import NOTE_KIND, note_is_stale, synthesize_note

#: How many query_log rows count as "recent" when scoring hot concepts.
USAGE_WINDOW = 200

#: Relative weight of the three background signals. Usage dominates: D11's
#: whole claim is that the vault should grow around what gets asked.
_HOT_WEIGHT = 3.0
_TAG_WEIGHT = 1.0
_LINK_WEIGHT = 2.0

#: A queued keyword always outranks background demand, whatever its score.
_QUEUED_BONUS = 1_000.0

DEFAULT_MAX_NOTES = 20
DEFAULT_MAX_FAILURES = 3


@dataclass
class ConsolidationBudget:
    """What one consolidation run is allowed to spend."""

    max_notes: int = DEFAULT_MAX_NOTES
    max_failures: int = DEFAULT_MAX_FAILURES


@dataclass
class Demand:
    """One concept awaiting synthesis, with why it deserves it."""

    keyword_id: int
    name: str
    reason: str
    score: float
    queued: bool = False


@dataclass
class ConsolidationReport:
    """What one run did. `stopped_early` means quota, not failure."""

    built: int = 0
    refreshed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[tuple[int, str]] = field(default_factory=list)
    stopped_early: bool = False
    keyword_ids: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.built + self.refreshed


def demand(
    kb: KnowledgeBase, *, limit: int = 50, include_background: bool = True
) -> list[Demand]:
    """Concepts ranked by how much they deserve consolidation.

    Queued rows first (a query already asked for them), then background
    demand from usage and centrality. Keywords that already have a fresh
    note are left out: nothing to do for them.
    """
    queued_rows = kb.store.consolidation_demand(limit=limit)
    queued_ids = {int(row["keyword_id"]) for row in queued_rows}

    ranked: list[Demand] = [
        Demand(
            keyword_id=int(row["keyword_id"]),
            name=str(row["name"]),
            reason=str(row["reason"]),
            score=_QUEUED_BONUS,
            queued=True,
        )
        for row in queued_rows
    ]

    if include_background:
        for entry in _background_demand(kb):
            if entry.keyword_id in queued_ids:
                continue
            note = kb.store.get_note(kind=NOTE_KIND, keyword_id=entry.keyword_id)
            if note is not None and not note["stale"]:
                continue  # already consolidated and still fresh
            ranked.append(entry)

    ranked.sort(key=lambda d: (-d.score, d.keyword_id))
    return ranked[:limit]


def _background_demand(kb: KnowledgeBase) -> list[Demand]:
    """Hot (asked about) and central (well-connected) concepts."""
    keywords = kb.store.list_keywords()
    if not keywords:
        return []

    hits: collections.Counter[int] = collections.Counter()
    for row in kb.store.get_queries(limit=USAGE_WINDOW):
        for keyword_id in row["hit_keywords"]:
            hits[int(keyword_id)] += 1

    incoming: collections.Counter[int] = collections.Counter()
    for keyword in keywords:
        note = kb.store.get_note(kind=NOTE_KIND, keyword_id=int(keyword["id"]))
        if note is None:
            continue
        for link in kb.store.note_links_from(int(note["id"])):
            incoming[int(link["to_keyword_id"])] += 1

    out: list[Demand] = []
    for keyword in keywords:
        keyword_id = int(keyword["id"])
        hot = hits.get(keyword_id, 0)
        tags = len(kb.store.chunks_for_keyword(keyword_id))
        links = incoming.get(keyword_id, 0)
        if not tags and not hot:
            continue  # a keyword nothing points at is not demand
        score = _HOT_WEIGHT * hot + _TAG_WEIGHT * tags + _LINK_WEIGHT * links
        out.append(
            Demand(
                keyword_id=keyword_id,
                name=str(keyword["name"]),
                reason="hot" if hot else "central",
                score=score,
            )
        )
    return out


def consolidate_pending(
    kb: KnowledgeBase,
    *,
    client: LLMClient,
    budget: ConsolidationBudget | None = None,
    embedder: EmbedderLike | None = None,
    include_background: bool = True,
) -> ConsolidationReport:
    """Synthesize the most deserving notes until the budget is spent."""
    budget = budget or ConsolidationBudget()
    report = ConsolidationReport()
    if budget.max_notes <= 0:
        return report

    for entry in demand(kb, limit=budget.max_notes, include_background=include_background):
        if report.total >= budget.max_notes:
            break
        existed = kb.store.get_note(kind=NOTE_KIND, keyword_id=entry.keyword_id) is not None
        try:
            synthesize_note(kb, entry.keyword_id, client=client, embedder=embedder)
        except AllKeysExhausted as exc:
            # Quota: stop the run, leave the queue intact for the next one.
            report.stopped_early = True
            report.errors.append((entry.keyword_id, str(exc)))
            return report
        except Exception as exc:
            report.failed += 1
            report.errors.append((entry.keyword_id, str(exc)))
            if report.failed >= budget.max_failures:
                # Failing systematically: stop rather than burn the queue.
                return report
            continue
        # Dequeued only after the note exists, so a crash re-does the work
        # instead of losing the request.
        if entry.queued:
            kb.store.dequeue_consolidation(entry.keyword_id)
        report.keyword_ids.append(entry.keyword_id)
        if existed:
            report.refreshed += 1
        else:
            report.built += 1
    return report


def refresh_stale(
    kb: KnowledgeBase,
    *,
    client: LLMClient,
    limit: int = DEFAULT_MAX_NOTES,
    embedder: EmbedderLike | None = None,
) -> ConsolidationReport:
    """Rebuild keyword notes whose source set changed since synthesis."""
    report = ConsolidationReport()
    for note in _stale_keyword_notes(kb, limit=limit):
        keyword_id = int(note["keyword_id"])
        try:
            synthesize_note(kb, keyword_id, client=client, embedder=embedder)
        except AllKeysExhausted as exc:
            report.stopped_early = True
            report.errors.append((keyword_id, str(exc)))
            return report
        except Exception as exc:
            report.failed += 1
            report.errors.append((keyword_id, str(exc)))
            continue
        report.refreshed += 1
        report.keyword_ids.append(keyword_id)
    return report


def _stale_keyword_notes(kb: KnowledgeBase, *, limit: int) -> list[dict[str, Any]]:
    """Notes flagged stale in the store, plus those whose sources drifted.

    The flag catches what `forget_document` invalidated; `note_is_stale`
    catches what re-ingestion changed without anyone saying so.
    """
    seen: dict[int, dict[str, Any]] = {}
    for note in kb.store.stale_notes():
        if note["kind"] == NOTE_KIND and note["keyword_id"] is not None:
            seen[int(note["id"])] = note
    for keyword in kb.store.list_keywords():
        if len(seen) >= limit:
            break
        note = kb.store.get_note(kind=NOTE_KIND, keyword_id=int(keyword["id"]))
        if note is None or int(note["id"]) in seen:
            continue
        if note_is_stale(kb, int(note["id"])):
            seen[int(note["id"])] = note
    return list(seen.values())[:limit]


def run_consolidation(
    kb: KnowledgeBase,
    client: LLMClient,
    budget: ConsolidationBudget | None = None,
    *,
    embedder: EmbedderLike | None = None,
) -> ConsolidationReport:
    """One background pass: refresh what drifted, then build what is due."""
    budget = budget or ConsolidationBudget()
    report = refresh_stale(kb, client=client, limit=budget.max_notes, embedder=embedder)
    if report.stopped_early:
        return report
    remaining = budget.max_notes - report.total
    if remaining <= 0:
        return report
    second = consolidate_pending(
        kb,
        client=client,
        budget=ConsolidationBudget(max_notes=remaining, max_failures=budget.max_failures),
        embedder=embedder,
    )
    return _merge(report, second)


def _merge(first: ConsolidationReport, second: ConsolidationReport) -> ConsolidationReport:
    return ConsolidationReport(
        built=first.built + second.built,
        refreshed=first.refreshed + second.refreshed,
        failed=first.failed + second.failed,
        skipped=first.skipped + second.skipped,
        errors=[*first.errors, *second.errors],
        stopped_early=first.stopped_early or second.stopped_early,
        keyword_ids=[*first.keyword_ids, *second.keyword_ids],
    )
