"""Verification (R2): coverage judge + repair queries (CRAG-lite).

After assembly, a light LLM call (role: classify) judges whether the
assembled evidence covers the question; if not, it emits 1-2 targeted
queries for the missing aspects, and ONE more retrieval round runs.
Hard-capped (default: a single repair iteration) so cost and latency stay
predictable. Confidence prior 65 (D12); benchmark question Q4.

Also owns the consolidation trigger: concepts touched by the query that
lack a fresh L2 note are pushed to consolidation_queue (D11) — deep
queries leave the KB better than they found it.

The judge never sees a reason to quote the corpus: it returns a verdict,
short descriptions of what is missing, and new search queries (D3, the
no-verbatim-echo rule). Entries are referred to by their assembly number.

Failure discipline
------------------
The coverage check is an *optimization*, never a gate: a broken judge
must not block an answer that already has context. Malformed JSON is
retried up to ``max_attempts``; exhausted attempts, a non-object payload
or any provider exception degrade to ``Coverage(ok=True, fallback=True)``
— i.e. "good enough", no repair round. ``AllKeysExhausted`` is the one
exception that propagates: the answer call right after would fail too, so
the caller must see it.

Normalization (``_validate_coverage``, pure)
--------------------------------------------
``ok`` is coerced to bool; blank and non-string list items are dropped;
``repair_queries`` is capped at ``MAX_REPAIR_QUERIES``; and when ``ok`` is
true ``repair_queries`` is forced empty — a judge that says the evidence
suffices must not be able to spend a retrieval round anyway, and that is
the only field the caller acts on. ``missing_aspects`` is kept as given
(a satisfied judge may still note a minor gap; it is reporting only).

Consolidation trigger (D11)
---------------------------
``trigger_consolidation`` queues the query's keywords that have no fresh
L2 note (missing or stale). The store's ``consolidation_queue`` is keyed
by (keyword_id, reason) with ON CONFLICT DO NOTHING, so re-queueing is
idempotent for a given reason; ids repeated within one call are
deduplicated here. The return value counts the keywords this call pushed,
not the growth of the queue. Nothing is synthesized inline: the default
is defer (D11), the current query proceeds with chunks.

Public API
----------
- ``MAX_REPAIR_QUERIES``, ``DEFAULT_MAX_ROUNDS``, ``MAX_COVERAGE_ATTEMPTS``
- ``COVERAGE_SYSTEM_PROMPT``, ``COVERAGE_RESPONSE_SCHEMA``
- ``class Coverage``: ok, missing_aspects, repair_queries, fallback
- ``check_coverage(client, query, context, *, max_attempts=2) -> Coverage``
- ``trigger_consolidation(kb, hit_keywords, *, reason="query") -> int``
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from wikirag.config import KnowledgeBase
from wikirag.llm.client import AllKeysExhausted, LLMClient
from wikirag.retrieval.assembly import Context

__all__ = [
    "COVERAGE_RESPONSE_SCHEMA",
    "COVERAGE_SYSTEM_PROMPT",
    "DEFAULT_MAX_ROUNDS",
    "MAX_COVERAGE_ATTEMPTS",
    "MAX_REPAIR_QUERIES",
    "Coverage",
    "check_coverage",
    "trigger_consolidation",
]

MAX_REPAIR_QUERIES = 2
DEFAULT_MAX_ROUNDS = 1  # hard cap: one repair round, so cost stays predictable
MAX_COVERAGE_ATTEMPTS = 2

COVERAGE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "missing_aspects": {"type": "array", "items": {"type": "string"}},
        "repair_queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ok", "missing_aspects", "repair_queries"],
}

COVERAGE_SYSTEM_PROMPT = """You check whether retrieved evidence is enough to answer a question, for a knowledge-base retrieval system. You do not answer the question.

You receive the QUESTION and numbered EVIDENCE entries ([1], [2], ...) retrieved from the knowledge base.

Decide whether the entries, taken together, contain the information needed for a complete answer:

- "ok": true when the evidence suffices, false when a needed aspect of the question has no support in it.
- "missing_aspects": when ok is false, short descriptions (a few words each) of what the evidence does not cover. Describe the gap; do not copy or quote text from the entries, and refer to entries by their number only.
- "repair_queries": when ok is false, at most 2 short standalone search queries (keywords, not questions) that would retrieve the missing information. They must target the gaps, not repeat the original question.

When ok is true, "missing_aspects" and "repair_queries" must both be empty arrays.

Judge only coverage, not correctness: evidence that is on-topic and complete is ok even if you disagree with it. If the question is broad and the evidence covers its substance, that is ok — repair rounds are expensive, ask for one only for a real gap.

Output the JSON object only, no markdown fences, no commentary."""


@dataclass
class Coverage:
    """The judge's verdict on one assembled context."""

    ok: bool
    missing_aspects: list[str] = field(default_factory=list)
    repair_queries: list[str] = field(default_factory=list)
    fallback: bool = False  # the judge failed; treated as ok, never blocks an answer


_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _parse_json_object(text: str) -> Any:
    """Parse the judge's output, tolerating markdown fences a model adds anyway."""
    return json.loads(_JSON_FENCE.sub("", text or "").strip())


def _clean_strings(value: Any) -> list[str]:
    """Non-empty stripped strings from a list-ish payload; anything else drops."""
    if not isinstance(value, (list, tuple)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _validate_coverage(data: Mapping[str, Any]) -> Coverage:
    """Normalize the judge's raw JSON into a coherent Coverage. Pure."""
    ok = bool(data.get("ok", False))
    repair_queries = _clean_strings(data.get("repair_queries"))[:MAX_REPAIR_QUERIES]
    return Coverage(
        ok=ok,
        missing_aspects=_clean_strings(data.get("missing_aspects")),
        # A verdict of "covered" cannot also spend a retrieval round.
        repair_queries=[] if ok else repair_queries,
    )


def _build_coverage_message(query: str, context: Context) -> str:
    evidence = context.text.strip() or "(no evidence was retrieved)"
    return f"QUESTION:\n{query}\n\nEVIDENCE:\n{evidence}"


def check_coverage(
    client: LLMClient,
    query: str,
    context: Context,
    *,
    max_attempts: int = MAX_COVERAGE_ATTEMPTS,
) -> Coverage:
    """Judge whether the assembled context covers the question.

    Returns ``ok=True`` with empty repair queries when the evidence
    suffices; otherwise the missing aspects and up to
    ``MAX_REPAIR_QUERIES`` targeted search queries for one repair round.

    Malformed JSON is retried up to ``max_attempts``; exhausted attempts
    or any provider exception degrade to ``Coverage(ok=True,
    fallback=True)`` so a broken judge never blocks an answer.
    ``AllKeysExhausted`` propagates.
    """
    user_message = _build_coverage_message(query, context)

    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            result = client.generate(
                system=COVERAGE_SYSTEM_PROMPT,
                user=user_message,
                json_schema=COVERAGE_RESPONSE_SCHEMA,
            )
            data = _parse_json_object(result.text)
        except json.JSONDecodeError:
            if attempt < max_attempts:
                continue
            return Coverage(ok=True, fallback=True)
        except AllKeysExhausted:
            raise
        except Exception:
            return Coverage(ok=True, fallback=True)

        if not isinstance(data, dict):
            if attempt < max_attempts:
                continue
            return Coverage(ok=True, fallback=True)
        return _validate_coverage(data)

    return Coverage(ok=True, fallback=True)


def trigger_consolidation(
    kb: KnowledgeBase, hit_keywords: Sequence[int], *, reason: str = "query"
) -> int:
    """Queue the query's concepts that lack a fresh L2 note (D11).

    A keyword needs consolidation when it has no keyword note, or its note
    is stale (its source set changed). Returns how many keywords this call
    pushed to ``consolidation_queue``; queueing is idempotent per
    (keyword, reason), so a repeat costs nothing and grows no demand.
    On an S0 KB there are no keywords, so there is nothing to queue.
    """
    store = kb.store
    queued = 0
    seen: set[int] = set()
    for raw in hit_keywords:
        keyword_id = int(raw)
        if keyword_id in seen:
            continue
        seen.add(keyword_id)
        note = store.get_note(kind="keyword", keyword_id=keyword_id)
        if note is not None and not note["stale"]:
            continue
        store.queue_consolidation(keyword_id, reason)
        queued += 1
    return queued
